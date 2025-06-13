#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <string>

class EdgeDetector {
public:
    EdgeDetector(const cv::Mat& inputImage);

    void applyLaplacian(const std::string& windowName = "Laplacian Edges");
    void applyCanny(const std::string& windowName = "Canny Edges");
    void applySobel(const std::string& windowName = "Sobel Edges");
    void applyScharr(const std::string& windowName = "Scharr Edges");
    void applyPrewitt(const std::string& windowName = "Prewitt Edges");
    void applyRoberts(const std::string& windowName = "Roberts Edges");
    void applyLoG(const std::string& windowName = "LoG Edges");

private:
    cv::Mat img;
};

#endif // EDGE_DETECTOR_HPP
