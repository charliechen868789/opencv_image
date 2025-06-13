cmake_minimum_required(VERSION 2.8)
project( opencv_demo )

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_demo
    src/main.cpp
    src/edge_detector.cpp
)

target_link_libraries(opencv_demo ${OpenCV_LIBS})
