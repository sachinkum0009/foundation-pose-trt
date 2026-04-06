#include "foundation_pose/trt_engine.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

int main() {
  std::string onnx_path = "foundation-pose-trt/onnx/model.engine"; // "foundation-pose-trt/onnx/yolo_model_100_static.onnx";
  std::string img_path = "foundation-pose-trt/images/2012_004148.jpg";
  foundation_pose::TrtEngine trt_engine(onnx_path);

  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    std::cerr << "Could not load image!\n";
    return -1;
  }

  trt_engine.infer(img);

  cv::imwrite("output_result2.jpg", img);
  std::cout << "Inference complete. Saved files\n";
  return 0;
}
