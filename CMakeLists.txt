cmake_minimum_required(VERSION 3.16)

project(LaboratorioPDI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs)

add_executable(laboratorio_pdi
    src/main.cpp
    src/image_io.cpp
    src/intensity.cpp
    src/histogram.cpp
    src/convolution.cpp
    src/spatial_filters.cpp
    src/edge_detection.cpp
    src/sharpening.cpp
    src/dissolve.cpp
)

target_include_directories(laboratorio_pdi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(laboratorio_pdi PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(laboratorio_pdi PRIVATE /W4)
else()
    target_compile_options(laboratorio_pdi PRIVATE -Wall -Wextra -Wpedantic)
endif()
