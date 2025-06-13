#include "edge_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

EdgeDetector::EdgeDetector(const cv::Mat& inputImage) {
    // Convert to grayscale if needed
    if (inputImage.channels() == 3)
        cvtColor(inputImage, img, COLOR_BGR2GRAY);
    else
        img = inputImage.clone();
}

void EdgeDetector::applyLaplacian(const std::string& windowName) {
    Mat laplacian;
    Laplacian(img, laplacian, CV_16S, 3);
    convertScaleAbs(laplacian, laplacian);
    imshow(windowName, laplacian);
}

void EdgeDetector::applyCanny(const std::string& windowName) {
    Mat edges;
    Canny(img, edges, 100, 200);
    imshow(windowName, edges);
}

void EdgeDetector::applySobel(const std::string& windowName) {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;

    Sobel(img, grad_x, CV_16S, 1, 0, 3);
    Sobel(img, grad_y, CV_16S, 0, 1, 3);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    imshow(windowName, grad);
}

void EdgeDetector::applyScharr(const std::string& windowName) {
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;

    Scharr(img, grad_x, CV_16S, 1, 0);
    Scharr(img, grad_y, CV_16S, 0, 1);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    imshow(windowName, grad);
}

void EdgeDetector::applyPrewitt(const std::string& windowName) {
    // Prewitt kernels
    Mat kernelx = (Mat_<float>(3,3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);
    Mat kernely = (Mat_<float>(3,3) <<
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1);

    Mat grad_x, grad_y;
    filter2D(img, grad_x, CV_32F, kernelx);
    filter2D(img, grad_y, CV_32F, kernely);

    Mat abs_grad_x, abs_grad_y, grad;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    imshow(windowName, grad);
}

void EdgeDetector::applyRoberts(const std::string& windowName) {
    // Roberts cross uses 2x2 kernels
    Mat kernelx = (Mat_<float>(2,2) <<
        1, 0,
        0,-1);
    Mat kernely = (Mat_<float>(2,2) <<
        0, 1,
       -1, 0);

    Mat grad_x, grad_y;
    filter2D(img, grad_x, CV_32F, kernelx);
    filter2D(img, grad_y, CV_32F, kernely);

    Mat abs_grad_x, abs_grad_y, grad;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    imshow(windowName, grad);
}

void EdgeDetector::applyLoG(const std::string& windowName) {
    Mat blurred, log;
    GaussianBlur(img, blurred, Size(3,3), 0);
    Laplacian(blurred, log, CV_16S, 3);
    convertScaleAbs(log, log);
    imshow(windowName, log);
}
