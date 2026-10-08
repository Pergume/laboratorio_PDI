#pragma once

#include <opencv2/core.hpp>

namespace pdi {

enum class BorderMode {
    ZeroPadding,
    Replicate,
    Reflect
};

cv::Mat convolve(
    const cv::Mat& image,
    const cv::Mat& kernel,
    double offset = 0.0,
    BorderMode borderMode = BorderMode::Replicate
);

} // namespace pdi
