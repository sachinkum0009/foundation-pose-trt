#ifndef FOUNDATION_POSE_YOLO_HPP_
#define FOUNDATION_POSE_YOLO_HPP_

#include "foundation_pose/trt_engine.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>

namespace yolo
{
    class Yolo : public rclcpp::Node
    {
    public:
        /**
         * @brief Constructor for the YOLO
         * * Initializes the class
         * @param model_path Path to the exported YOLO model file (.onnx, or .engine)
         */
        Yolo(const std::string &model_path);
        ~Yolo();

        /**
         * @brief Performs object detection on the input image.
         * @param img The input frame (typically BGR from OpenCV). The function
         * * automatically handles resizing and normalization.
         */
        void run_inference(const cv::Mat &img);

    private:
        void image_cb(sensor_msgs::msg::Image::ConstSharedPtr msg);
        std::unique_ptr<foundation_pose::TrtEngine> _trt_engine;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _img_raw_sub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _img_res_pub;
    };
} // namespace yolo

#endif
