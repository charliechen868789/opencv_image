#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <opencv2/core.hpp>

class EdgeDetector {
public:
    EdgeDetector(const cv::Mat& inputImage);
    void applyLaplacian();
    void applyCanny();

private:
    cv::Mat img;
};

#endif // EDGE_DETECTOR_HPP
