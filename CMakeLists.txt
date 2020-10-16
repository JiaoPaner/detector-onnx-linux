cmake_minimum_required(VERSION 3.15)
project(detector)

set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_C_FLAGS "-Wwrite-strings")

find_package(OpenCV REQUIRED)
include_directories(include include/onnx ${OpenCV_INCLUDE_DIRS})

set(GPU 0)
# add onnxruntime-gpu lib to gpu/lib
#cuda10.1|cudnn7+ ----> onnxruntime-gpu-1.4.0+
#cuda10.2|cudnn8+ ----> onnxruntime-gpu-1.5.0+
if (${GPU})
    link_directories(lib/gpu)
    add_definitions(-D GPU)
else()
    link_directories(lib)
endif ()

file(GLOB src src/*.cpp include/*.h src/*.c)
set(SRC ${src})

add_executable(detector ${SRC})
#add_library(detector SHARED ${SRC})

target_link_libraries(detector ${OpenCV_LIBS} onnxruntime)