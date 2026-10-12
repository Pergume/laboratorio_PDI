#pragma once

#include <opencv2/core.hpp>

namespace pdi {

enum class BorderMode {
    ZeroPadding, // Fora da imagem vale 0.
    Replicate,   // Replica o pixel mais proximo da borda.
    Reflect      // Espelha os pixels em relacao a borda.
};

cv::Mat convolveRaw(
    const cv::Mat& image,
    const cv::Mat& kernel,
    double offset = 0.0,
    BorderMode borderMode = BorderMode::Replicate
);

cv::Mat convolve(
    const cv::Mat& image,
    const cv::Mat& kernel,
    double offset = 0.0,
    BorderMode borderMode = BorderMode::Replicate
);

} 
