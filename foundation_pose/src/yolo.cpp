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

#include "foundation_pose/yolo.hpp"

namespace yolo {
Yolo::Yolo(const std::string& model_path) : Node("yolo_ros_node") {
  _trt_engine = std::make_unique<foundation_pose::TrtEngine>(model_path);
  _img_raw_sub = this->create_subscription<sensor_msgs::msg::Image>(
      "img_in", 10, std::bind(&Yolo::image_cb, this, std::placeholders::_1));
  _img_res_pub = this->create_publisher<sensor_msgs::msg::Image>("img_out", 10);
}

void Yolo::run_inference(const cv::Mat& img) {
  if (_trt_engine) {
    _trt_engine->infer(img);
  }
}

void Yolo::image_cb(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  // convert img to opencv
  cv_bridge::CvImagePtr cv_ptr;
  try {
    // Use bgr8 for standard OpenCV color format
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  run_inference(cv_ptr->image);
}
}  // namespace yolo

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  std::string model_path =
      "/home/asus/backup/zzzzz/ros2/personal_projects/vision-flow/"
      "foundation-pose-trt/yolo26l_rtx3060.engine";
  auto node = std::make_shared<yolo::Yolo>(model_path);
  try {
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Node crashed: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}