#include "foundation_pose/foundation_pose.hpp"

namespace foundation_pose {
FoundationPose::FoundationPose() {
  nvinfer1::IRuntime *rawRuntime = nvinfer1::createInferRuntime(gLogger);

  if (!rawRuntime) {
    throw std::runtime_error("Failed to create TensorRT Runtime");
  }

  _runtime.reset(rawRuntime);
}
FoundationPose::~FoundationPose() {}

void FoundationPose::runInteference() {}
} // namespace foundation_pose
