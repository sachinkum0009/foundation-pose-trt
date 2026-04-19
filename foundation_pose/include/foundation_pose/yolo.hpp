// Copyright 2026 Sachin Kumar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FOUNDATION_POSE_YOLO_HPP_
#define FOUNDATION_POSE_YOLO_HPP_

#include <cv_bridge/cv_bridge.hpp>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "foundation_pose/trt_engine.hpp"

namespace yolo {
class Yolo : public rclcpp::Node {
 public:
  /**
   * @brief Constructor for the YOLO
   * * Initializes the class
   * @param model_path Path to the exported YOLO model file (.onnx, or .engine)
   */
  Yolo(const std::string& model_path);
  ~Yolo() = default;

  /**
   * @brief Performs object detection on the input image.
   * @param img The input frame (typically BGR from OpenCV). The function
   * * automatically handles resizing and normalization.
   */
  void run_inference(const cv::Mat& img);

 private:
  /**
   * @brief Image Callback function
   * @param msg Image from the ros2 topic
   */
  void image_cb(sensor_msgs::msg::Image::ConstSharedPtr msg);
  std::unique_ptr<foundation_pose::TrtEngine> _trt_engine;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _img_raw_sub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _img_res_pub;
};
}  // namespace yolo

#endif
