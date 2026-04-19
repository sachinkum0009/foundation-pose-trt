#ifndef FOUNDATION_POSE_YOLO_HPP_
#define FOUNDATION_POSE_YOLO_HPP_

#include "foundation_pose/trt_engine.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <memory>

namespace yolo
{
    class Yolo
    {
    public:
        Yolo(const std::string &model_path);
        ~Yolo();

        /**
         * @brief Run inference to detect objects
         */
        void run_inference(const cv::Mat &img);

    private:
        std::unique_ptr<foundation_pose::TrtEngine> _trt_engine;
    };
} // namespace yolo

#endif
