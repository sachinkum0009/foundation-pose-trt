#ifndef FOUNDATION_POSE_TRT_ENGINE_HPP_
#define FOUNDATION_POSE_TRT_ENGINE_HPP_

#include <driver_types.h>
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv4/opencv2/opencv.hpp>

namespace foundation_pose {

/// @brief TrtEngine
/// Inference the model
class TrtEngine {
public:
  TrtEngine(std::string& onnx_path);
  ~TrtEngine();

  void infer(const cv::Mat& img);
private:
    void build(const std::string& onnx_path);
    // TensorRT objects
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    // CUDA buffers
    std::vector<void*> buffers;
    std::vector<int64_t> buffer_sizes;
    int input_index, output_index;

    // CPU output buffer
    std::vector<float> output_data;

    // Cuda Stream
    cudaStream_t stream;
    // std::string _model_path;
};

} // namespace foundation_pose

#endif // FOUNDATION_POSE_TRT_ENGINE_HPP_
