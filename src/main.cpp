#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <mutex>
#include "edge_detector.hpp"
#include "thread_pool.hpp"

using namespace cv;
using namespace std;

std::mutex gui_mutex;

void showWithLock(const string& windowName, function<void()> showFunc) {
    std::lock_guard<std::mutex> lock(gui_mutex);
    showFunc();
    waitKey(0);
    destroyWindow(windowName);
}

void runLaplacian(Mat img, int instance) {
    EdgeDetector detector(img);
    string windowName = "Laplacian Result " + to_string(instance);
    // Run processing without GUI lock
    detector.applyLaplacian(windowName);

    // Protect waitKey & destroyWindow
    showWithLock(windowName, [](){ /* nothing extra here, imshow called already */ });
}

void runCanny(Mat img, int instance) {
    EdgeDetector detector(img);
    string windowName = "Canny Result " + to_string(instance);
    detector.applyCanny(windowName);

    showWithLock(windowName, [](){});
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    string imagePath = argv[1];
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "Error: Could not load image from: " << imagePath << endl;
        return -1;
    }

    int laplacianCount = 1, cannyCount = 1;
    ThreadPool pool(thread::hardware_concurrency()); // e.g. 4 or 8 threads

    while (true) {
        cout << "\n=== Edge Detection Menu ===\n";
        cout << "1. Laplacian Edge Detection\n";
        cout << "2. Canny Edge Detection\n";
        cout << "3. Both\n";
        cout << "4. Exit\n";
        cout << "Enter your choice (1-4): ";

        int choice;
        cin >> choice;

        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a number between 1 and 4.\n";
            continue;
        }

        switch (choice) {
            case 1:
                pool.enqueue(runLaplacian, img.clone(), laplacianCount++);
                break;
            case 2:
                pool.enqueue(runCanny, img.clone(), cannyCount++);
                break;
            case 3:
                pool.enqueue(runLaplacian, img.clone(), laplacianCount++);
                pool.enqueue(runCanny, img.clone(), cannyCount++);
                break;
            case 4:
                cout << "Exiting...\n";
                pool.shutdown();
                return 0;
            default:
                cout << "Invalid choice! Please try again.\n";
                break;
        }

        cout << "Detection launched. You can enter another option.\n";
    }
}
