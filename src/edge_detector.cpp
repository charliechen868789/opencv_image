#include "edge_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

EdgeDetector::EdgeDetector(const Mat& inputImage) {
    img = inputImage.clone();
}

void EdgeDetector::applyLaplacian() {
    Mat blur, imgLaplacian;
    vector<Mat> rgbColourEdges(3);

    GaussianBlur(img, blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
    vector<Mat> channels(3);
    split(blur, channels);

    double T = 20;
    double maxval = 255;

    for (int i = 0; i < 3; i++) {
        Laplacian(channels[i], rgbColourEdges[i], CV_32F);
        convertScaleAbs(rgbColourEdges[i], rgbColourEdges[i]);
        threshold(rgbColourEdges[i], rgbColourEdges[i], T, maxval, THRESH_BINARY);
    }

    Mat tempEdges;
    bitwise_or(rgbColourEdges[0], rgbColourEdges[1], tempEdges);
    bitwise_or(tempEdges, rgbColourEdges[2], imgLaplacian);
    imgLaplacian.convertTo(imgLaplacian, CV_8U);

    imshow("Laplacian Edges", imgLaplacian);
    imwrite("Laplacian_T20.jpg", imgLaplacian);
}

void EdgeDetector::applyCanny() {
    Mat imgCanny;
    Canny(img, imgCanny, 200, 230);

    imshow("Canny Edges", imgCanny);
    imwrite("Canny_T1_200_T2_230.jpg", imgCanny);
}
