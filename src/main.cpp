#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <mutex>
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

#include "edge_detector.hpp"
#include "hand_segmenter.hpp"
#include "color_transfer.hpp"
#include "contrast_enhancer.hpp"

#include "thread_pool.hpp"

using namespace cv;
using namespace std;

// GUI task queue
std::queue<std::function<void()>> guiTasks;
std::mutex guiQueueMutex;
std::condition_variable guiQueueCond;
std::atomic<bool> guiRunning{true};

// GUI thread function
void guiThreadFunc() {
    while (guiRunning) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(guiQueueMutex);
            guiQueueCond.wait(lock, [] { return !guiTasks.empty() || !guiRunning; });
            if (!guiRunning && guiTasks.empty())
                break;
            task = std::move(guiTasks.front());
            guiTasks.pop();
        }
        task();  // Show window, waitKey, destroyWindow
    }
}

// Enqueue a GUI task
void enqueueGuiTask(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(guiQueueMutex);
        guiTasks.push(std::move(task));
    }
    guiQueueCond.notify_one();
}

// Background edge detection function
void runEdgeDetection(function<void(EdgeDetector&, const string&)> func, Mat img, int instance, const string& methodName) {
    EdgeDetector detector(img);
    string windowName = methodName + " Result " + to_string(instance);

    // Prepare GUI operation as task
    enqueueGuiTask([detector, windowName, func]() mutable {
        func(detector, windowName);
        int key = 0;
        while (key == 0) {
            key = waitKey(30);  // Responsive wait
        }
        destroyWindow(windowName);
    });
}

void runColorTransfer(const cv::Mat& targetImg, int instance) {
    // Load style image from fixed path
    std::string stylePath = "../assets/paint.jpg";  // Or set to your known style image location
    cv::Mat styleImg = cv::imread(stylePath);

    if (styleImg.empty()) {
        std::cerr << "Error: Could not load style image from " << stylePath << std::endl;
        return;
    }

    ColorTransfer colorTransfer(styleImg, targetImg);
    cv::Mat result = colorTransfer.applyTransfer();

    enqueueGuiTask([result, instance]() {
        std::string windowName = "Color Transfer Result " + std::to_string(instance);
        cv::namedWindow(windowName);
        cv::imshow(windowName, result);
        cv::waitKey(0);  // Wait for user to press a key
        cv::destroyWindow(windowName);
    });
}

void runHandSegmentation(const cv::Mat& img, int instance) {
    // Create segmenter once here
    HandSegmenter segmenter(img);

    // Enqueue GUI task that calls segmentAndShow (shows images + waitKey)
    enqueueGuiTask([segmenter, instance]() mutable {
        std::string windowName = "Hand Segmentation Result " + std::to_string(instance);
        segmenter.segmentAndShow(windowName);
    });
}

void runContrastEnhancement(const cv::Mat& img, int instance) {
    ContrastEnhancer enhancer(img);
    enqueueGuiTask([enhancer, instance]() mutable {
        std::string windowPrefix = "Contrast Result " + std::to_string(instance);
        enhancer.enhanceAndShow(windowPrefix);

        // ✅ Wait for key press to keep windows open
        cv::waitKey(0);

        // ✅ Clean up all windows this task opened
        cv::destroyAllWindows();
    });
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

    ThreadPool pool(thread::hardware_concurrency());
    std::thread guiThread(guiThreadFunc);  // Start GUI thread
    int instanceCount = 1;

    while (true) {
        cout << "\n=== Main Menu ===\n";
        cout << "1. Edge Detection\n";
        cout << "2. Hand Segmentation\n";
        cout << "3. Color Transfer\n";
        cout << "4. Contrast Enhancer\n";
        cout << "20. Exit\n";
        cout << "Enter choice: ";

        int choiceMain;
        cin >> choiceMain;
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input!\n";
            continue;
        }

        if (choiceMain == 20) {
            cout << "Exiting...\n";
            break;
        }

        if (choiceMain == 1) {
            while (true) {
                cout << "\n--- Edge Detection Methods ---\n";
                cout << "1. Laplacian\n";
                cout << "2. Canny\n";
                cout << "3. Sobel\n";
                cout << "4. Scharr\n";
                cout << "5. Prewitt\n";
                cout << "6. Roberts\n";
                cout << "7. LoG\n";
                cout << "8. Back to Main Menu\n";
                cout << "Enter method choice: ";

                int methodChoice;
                cin >> methodChoice;
                if (cin.fail()) {
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    cout << "Invalid input!\n";
                    continue;
                }

                if (methodChoice == 8)
                    break;

                function<void(EdgeDetector&, const string&)> func;
                string methodName;

                switch(methodChoice) {
                    case 1: func = &EdgeDetector::applyLaplacian; methodName = "Laplacian"; break;
                    case 2: func = &EdgeDetector::applyCanny; methodName = "Canny"; break;
                    case 3: func = &EdgeDetector::applySobel; methodName = "Sobel"; break;
                    case 4: func = &EdgeDetector::applyScharr; methodName = "Scharr"; break;
                    case 5: func = &EdgeDetector::applyPrewitt; methodName = "Prewitt"; break;
                    case 6: func = &EdgeDetector::applyRoberts; methodName = "Roberts"; break;
                    case 7: func = &EdgeDetector::applyLoG; methodName = "LoG"; break;
                    default:
                        cout << "Invalid method choice!\n";
                        continue;
                }

                pool.enqueue(runEdgeDetection, func, img.clone(), instanceCount++, methodName);
                cout << "Task launched. You can select another method or return.\n";
            }
        } else if (choiceMain == 2) {
            static int handInstanceCount = 1;
            pool.enqueue(runHandSegmentation, img.clone(), handInstanceCount++);
            std::cout << "Hand segmentation task launched.\n";
        }else if (choiceMain == 3) {
            static int _handInstanceCount = 1;
            pool.enqueue(runColorTransfer, img.clone(), _handInstanceCount++);
            cout << "Color transfer task launched.\n";
        } else if (choiceMain == 4) {
            pool.enqueue(runContrastEnhancement, img.clone(), instanceCount++);
            cout << "Contrast enhancement task launched.\n";
        }
        else 
        {
            cout << "Invalid choice!\n";
        }
    }

    // Shutdown
    pool.shutdown();
    {
        std::lock_guard<std::mutex> lock(guiQueueMutex);
        guiRunning = false;
    }
    guiQueueCond.notify_all();
    guiThread.join();

    return 0;
}
