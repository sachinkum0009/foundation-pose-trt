#include "foundation_pose/yolo.hpp"

namespace yolo
{
    Yolo::Yolo(const std::string &model_path) {
        auto _trt_engine = std::make_unique<foundation_pose::TrtEngine>(model_path);
    }
    Yolo::~Yolo() {}

    void Yolo::run_inference(const cv::Mat &img) {
        _trt_engine->infer(img);
    }
} // namespace yolo
