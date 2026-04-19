#include "foundation_pose/yolo.hpp"

namespace yolo
{
    Yolo::Yolo(const std::string &model_path) : Node("yolo_ros_node")
    {
        _trt_engine = std::make_unique<foundation_pose::TrtEngine>(model_path);
        _img_raw_sub = this->create_subscription<sensor_msgs::msg::Image>("img_in", 10, std::bind(&Yolo::image_cb, this, std::placeholders::_1));
        _img_res_pub = this->create_publisher<sensor_msgs::msg::Image>("img_out", 10);
    }
    Yolo::~Yolo() {}

    void Yolo::run_inference(const cv::Mat &img)
    {
        _trt_engine->infer(img);
    }

    void Yolo::image_cb(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        // convert img to opencv
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            // Use bgr8 for standard OpenCV color format
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        run_inference(cv_ptr->image);
    }
} // namespace yolo

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::string model_path = "/home/asus/backup/zzzzz/ros2/personal_projects/vision-flow/foundation-pose-trt/yolo26l_rtx3060.engine";
    auto node = std::make_shared<yolo::Yolo>(model_path);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}