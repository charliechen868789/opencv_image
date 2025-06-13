#ifndef HAND_SEGMENTER_HPP
#define HAND_SEGMENTER_HPP

#include <opencv2/core.hpp>
#include <string>

class HandSegmenter {
public:
    explicit HandSegmenter(const cv::Mat& inputImage);

    // Segments the hand and displays the result window with the given name
    void segmentAndShow(const std::string& windowName = "Hand Segmentation");

private:
    cv::Mat img;

    int findLargestContour(const std::vector<std::vector<cv::Point>>& contours);
};

#endif // HAND_SEGMENTER_HPP
