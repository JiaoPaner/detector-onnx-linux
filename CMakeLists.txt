cmake_minimum_required(VERSION 3.15)
project(detector)

set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_C_FLAGS "-Wwrite-strings")

find_package(OpenCV REQUIRED)
include_directories(include include/onnx ${OpenCV_INCLUDE_DIRS})
link_directories(lib)

file(GLOB src src/*.cpp include/*.h src/*.c)
set(SRC ${src})

#add_executable(detector ${SRC})
add_library(detector SHARED ${SRC})

target_link_libraries(detector ${OpenCV_LIBS} onnxruntime)