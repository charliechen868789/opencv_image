#include "edge_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

EdgeDetector::EdgeDetector(const Mat& inputImage)
{
    img = inputImage.clone();
}

void EdgeDetector::applyLaplacian(const std::string& windowName)
{
    Mat blur, laplace;
    GaussianBlur(img, blur, Size(3, 3), 0);

    std::vector<Mat> channels(3), edges(3);
    split(blur, channels);

    for (int i = 0; i < 3; ++i) {
        Laplacian(channels[i], edges[i], CV_32F);
        convertScaleAbs(edges[i], edges[i]);
        threshold(edges[i], edges[i], 20, 255, THRESH_BINARY);
    }

    bitwise_or(edges[0], edges[1], laplace);
    bitwise_or(laplace, edges[2], laplace);

    imshow(windowName, laplace);
}

void EdgeDetector::applyCanny(const std::string& windowName)
{
    Mat cannyEdges;
    Canny(img, cannyEdges, 200, 230);
    imshow(windowName, cannyEdges);
}
