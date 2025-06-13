#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <string>

class EdgeDetector {
public:
    explicit EdgeDetector(const cv::Mat& inputImage);

    void applyLaplacian(const std::string& windowName = "Laplacian Edges");
    void applyCanny(const std::string& windowName = "Canny Edges");

private:
    cv::Mat img;
};

#endif // EDGE_DETECTOR_HPP
