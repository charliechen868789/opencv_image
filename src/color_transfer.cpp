#include "color_transfer.hpp"

using namespace cv;

ColorTransfer::ColorTransfer(const Mat& styleImage, const Mat& targetImage)
    : styleBGR(styleImage.clone()), targetBGR(targetImage.clone())
{
    // Convert to Lab space and float type
    cvtColor(styleBGR, styleLab, COLOR_BGR2Lab);
    cvtColor(targetBGR, targetLab, COLOR_BGR2Lab);
    styleLab.convertTo(styleLab, CV_32FC3);
    targetLab.convertTo(targetLab, CV_32FC3);
}

cv::Mat ColorTransfer::applyTransfer()
{
    // Compute mean and stddev for each channel
    Mat meanStyle, stdStyle, meanTarget, stdTarget;
    meanStdDev(styleLab, meanStyle, stdStyle);
    meanStdDev(targetLab, meanTarget, stdTarget);

    // Split channels
    std::vector<Mat> styleChs, targetChs;
    split(targetLab, targetChs); // Only target is modified

    for (int i = 0; i < 3; ++i) {
        targetChs[i] -= meanTarget.at<double>(i);
        targetChs[i] /= stdTarget.at<double>(i);
        targetChs[i] *= stdStyle.at<double>(i);
        targetChs[i] += meanStyle.at<double>(i);
    }

    Mat merged, result;
    merge(targetChs, merged);
    merged.convertTo(merged, CV_8UC3);
    cvtColor(merged, result, COLOR_Lab2BGR);
    return result;
}
