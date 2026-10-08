#pragma once

#include <opencv2/core.hpp>
#include "convolution.hpp"

namespace pdi {

cv::Mat meanFilter(
    const cv::Mat& image,
    int neighborhoodSize,
    BorderMode borderMode = BorderMode::Replicate
);

cv::Mat medianFilter(
    const cv::Mat& image,
    int neighborhoodSize,
    BorderMode borderMode = BorderMode::Replicate
);

} // namespace pdi
