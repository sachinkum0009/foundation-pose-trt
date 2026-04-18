#include "foundation_pose/trt_engine.hpp"
#include <fstream>

using namespace nvinfer1;

namespace foundation_pose {

// Logger for TensorRT
class Logger : public ILogger {
  void log(Severity serverity, const char *msg) noexcept override {
    if (serverity <= Severity::kWARNING)
      std::cout << "[TensorRT]: " << msg << "\n";
  }
} gLogger;

TrtEngine::TrtEngine(std::string &model_path) : S(7), B(3), C(20), threshold(0.52f) {
  build(model_path);
  cudaStreamCreate(&stream);
}
TrtEngine::~TrtEngine() {
  for (void *buf : buffers)
    cudaFree(buf);
  if (context)
    delete context;
  if (engine)
    delete engine;
  if (runtime)
    delete runtime;
}

void TrtEngine::build(const std::string &model_path) {
    // 1. Initialize Runtime first (needed for both paths)
        runtime = nvinfer1::createInferRuntime(gLogger);

        bool is_engine = (model_path.substr(model_path.find_last_of(".") + 1) == "engine");

        if (is_engine) {
            // --- LOAD ENGINE ---
            std::ifstream file(model_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open engine file: " << model_path << std::endl;
                return;
            }

            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> engine_data(size);
            file.read(engine_data.data(), size);
            file.close();

            engine = runtime->deserializeCudaEngine(engine_data.data(), size);
            std::cout << "Successfully loaded engine: " << model_path << std::endl;
        }
        else {
            // --- BUILD FROM ONNX ---
            auto builder = nvinfer1::createInferBuilder(gLogger);
            auto network = builder->createNetworkV2(0U);
            auto parser = nvonnxparser::createParser(*network, gLogger);

            if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
                std::cerr << "Failed to parse ONNX file!" << std::endl;
                return;
            }

            auto config = builder->createBuilderConfig();
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

            // Performance optimization
            if (builder->platformHasFastFp16()) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
                std::cout << "FP16 mode enabled." << std::endl;
            }

            auto plan = builder->buildSerializedNetwork(*network, *config);

            // Save the engine for next time
            std::string engine_path = model_path.substr(0, model_path.find_last_of(".")) + ".engine";
            std::ofstream p(engine_path, std::ios::binary);
            p.write(reinterpret_cast<const char*>(plan->data()), plan->size());
            p.close();
            std::cout << "Engine built and saved to: " << engine_path << std::endl;

            engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

            // Clean up builder resources
            delete parser;
            delete network;
            delete builder;
            delete config;
        }

        if (engine) {
            context = engine->createExecutionContext();
        }
        setup_buffers();
}

void TrtEngine::setup_buffers() {
  // Setup Buffers
  int nbBindings = engine->getNbIOTensors();
  for (int i = 0; i < nbBindings; ++i) {
    auto name = engine->getIOTensorName(i);
    auto dims = engine->getTensorShape(name);
    int64_t size = 1;
    for (int j = 0; j < dims.nbDims; ++j)
      size *= dims.d[j];

    void *gpu_ptr;
    cudaMalloc(&gpu_ptr, size * sizeof(float));
    buffers.push_back(gpu_ptr);
    buffer_sizes.push_back(size);

    if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
      input_index = i;
    else {
      output_index = i;
      output_data.resize(size);
    }
  }
}

void TrtEngine::infer(const cv::Mat &img) {
  // Preprocess
  cv::Mat resized, float_img;
  cv::resize(img, resized, cv::Size(448, 448));
  resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  // HWC to CHW (TensorRT expects NCHW)
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);
  std::vector<float> input_blob;
  for (auto &c : channels) {
    input_blob.insert(input_blob.end(), (float *)c.data,
                      (float *)c.data + 448 * 448);
  }

  // 1. Upload to GPU Asynchronously
  cudaMemcpyAsync(buffers[input_index], input_blob.data(),
                  buffer_sizes[input_index] * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  // 2. Run Inference on the specific stream
  context->setInputTensorAddress(engine->getIOTensorName(input_index),
                                 buffers[input_index]);
  context->setOutputTensorAddress(engine->getIOTensorName(output_index),
                                  buffers[output_index]);
  context->enqueueV3(stream);

  // 3. Download Results Asynchronously
  cudaMemcpyAsync(output_data.data(), buffers[output_index],
                  buffer_sizes[output_index] * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  // 4. Wait for the stream to finish before post-processing
  cudaStreamSynchronize(stream);

  // 5. Post Process Image
  post_processing(img);
}

void TrtEngine::post_processing(const cv::Mat &img) {
  // Post Processing
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < S; ++j) {
      for (int a = 0; a < B; ++a) {
        // Calculate index in the flat output_data array
        int base = ((i * S + j) * B + a) * 25;

        float raw_conf = output_data[base + 4];
        float conf = 1.0f / (1.0f + std::exp(-raw_conf)); // Sigmoid

        if (conf > threshold) {
          // Find Class Probabilities (Softmax)
          int class_id = 0;
          float max_prob = -1e10;
          float sum_exp = 0;

          // Simple max for class ID (similar to your np.argmax)
          for (int c = 0; c < C; ++c) {
            float p = std::exp(output_data[base + 5 + c]);
            if (p > max_prob) {
              max_prob = p;
              class_id = c;
            }
          }

          // Coordinates
          float x_cell = output_data[base + 0];
          float y_cell = output_data[base + 1];
          float w_rel = output_data[base + 2];
          float h_rel = output_data[base + 3];

          float x_relative = (j + x_cell) / S;
          float y_relative = (i + y_cell) / S;

          // Convert to Pixel Coordinates
          int width = static_cast<int>(w_rel * img.cols);
          int height = static_cast<int>(h_rel * img.rows);
          int xmin = static_cast<int>((x_relative * img.cols) - (width / 2.0));
          int ymin = static_cast<int>((y_relative * img.rows) - (height / 2.0));

          // 6. Draw on Image using OpenCV
          cv::Rect box(xmin, ymin, width, height);
          cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

          std::string label = "ID:" + std::to_string(class_id) + " " +
                              std::to_string(conf).substr(0, 4);
          cv::putText(img, label, cv::Point(xmin, ymin - 5),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                      1);
        }
      }
    }
  }

}

} // namespace foundation_pose
