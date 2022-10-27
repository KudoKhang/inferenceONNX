cmake_minimum_required(VERSION 3.23)
project(inferenceONNX)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /opt/homebrew/Cellar/onnxruntime/1.12.1/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_executable(inferenceONNX test.cpp)
target_include_directories(inferenceONNX PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inferenceONNX PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
