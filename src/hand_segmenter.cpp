#include "hand_segmenter.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

HandSegmenter::HandSegmenter(const cv::Mat& inputImage)
    : img(inputImage.clone())  // clone to own the image data
{
    if (img.empty()) {
        throw std::runtime_error("Error: Input image is empty");
    }
}

void HandSegmenter::segmentAndShow(const std::string& windowName) {
    // Step 1: Blur
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(5, 5), 0.67);

    // Step 2: Convert to HSV
    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // Step 3: Threshold HSV range to segment skin tone
    cv::Mat segmented;
    cv::inRange(hsv, cv::Scalar(0, 10, 60), cv::Scalar(20, 150, 255), segmented);

    // Step 4: Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(segmented, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cerr << "Warning: No contours found." << std::endl;
        cv::imshow(windowName, cv::Mat::zeros(img.size(), CV_8UC1));
        return;
    }

    // Step 5: Find largest contour index
    int largestIdx = findLargestContour(contours);

    // Step 6: Draw largest contour filled on blank image
    cv::Mat drawing = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::drawContours(drawing, contours, largestIdx, cv::Scalar(255), cv::FILLED, 8, hierarchy);

    // Step 7: Dilate to smooth edges
    int dilationSize = 6;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_CROSS,
        cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
        cv::Point(dilationSize, dilationSize));
    cv::dilate(drawing, drawing, element);

    // Step 8: Show result
    cv::imshow("Original Image", img);
    cv::imshow(windowName, drawing);
    cv::waitKey(0);
}

int HandSegmenter::findLargestContour(const std::vector<std::vector<cv::Point>>& contours) {
    int maxIdx = -1;
    size_t maxSize = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() > maxSize) {
            maxSize = contours[i].size();
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}
