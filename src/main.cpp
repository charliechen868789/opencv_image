#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "edge_detector.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("../assets/tulips.png");
    if (img.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }

    EdgeDetector detector(img);

    cout << "Choose Edge Detection Method:\n";
    cout << "1. Laplacian Edge Detection\n";
    cout << "2. Canny Edge Detection\n";
    cout << "3. Both\n";
    cout << "Enter your choice (1/2/3): ";

    int choice;
    cin >> choice;

    imshow("Original Image", img);

    switch (choice) {
        case 1:
            detector.applyLaplacian();
            break;
        case 2:
            detector.applyCanny();
            break;
        case 3:
            detector.applyLaplacian();
            detector.applyCanny();
            break;
        default:
            cout << "Invalid choice!" << endl;
            return -1;
    }

    waitKey(0);
    return 0;
}
