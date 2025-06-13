#ifndef COLOR_TRANSFER_HPP
#define COLOR_TRANSFER_HPP

#include <opencv2/opencv.hpp>

class ColorTransfer {
public:
    ColorTransfer(const cv::Mat& styleImage, const cv::Mat& targetImage);
    cv::Mat applyTransfer();

private:
    cv::Mat styleBGR;
    cv::Mat targetBGR;
    cv::Mat styleLab;
    cv::Mat targetLab;
};

#endif // COLOR_TRANSFER_HPP
