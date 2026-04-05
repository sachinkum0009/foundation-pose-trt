#ifndef FOUNDATION_POSE_HPP_
#define FOUNDATION_POSE_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

using namespace nvinfer1;

namespace foundation_pose {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT]" << msg << std::endl;
  }
};

class FoundationPose {
public:
  FoundationPose();
  ~FoundationPose();
  void runInteference();

private:
  Logger gLogger;
  struct TRTDeleter {
    template <typename T> void operator()(T *obj) const {
      if (obj) {
        delete obj;
      }
    }
  };
  float my_variable_;
  std::unique_ptr<IRuntime, TRTDeleter> _runtime;
};

} // namespace foundation_pose

#endif // FOUNDATION_POSE_HPP_
