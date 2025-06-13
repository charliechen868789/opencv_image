cmake_minimum_required(VERSION 3.5)
project( opencv_demo )

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_demo
    src/main.cpp
    src/edge_detector.cpp
    src/hand_segmenter.cpp
    src/color_transfer.cpp
    src/contrast_enhancer.cpp
)

target_link_libraries(opencv_demo ${OpenCV_LIBS})
