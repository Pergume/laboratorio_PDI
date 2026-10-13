cmake_minimum_required(VERSION 3.16)

project(Laboratorio_PDI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs highgui imgproc)

add_executable(laboratorio_PDI
    src/main.cpp
    src/frontend.cpp
    src/image_io.cpp
    src/intensity.cpp
    src/histogram.cpp
    src/convolution.cpp
    src/convolution_tests.cpp
    src/spatial_filters.cpp
    src/edge_detection.cpp
    src/sharpening.cpp
    src/dissolve.cpp
)

target_include_directories(laboratorio_PDI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)


target_link_libraries(laboratorio_PDI PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(laboratorio_PDI PRIVATE /W4)
else()
    target_compile_options(laboratorio_PDI PRIVATE -Wall -Wextra -Wpedantic)
endif()
