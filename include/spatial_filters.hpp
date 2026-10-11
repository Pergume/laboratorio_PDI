#pragma once

#include <opencv2/core.hpp>

namespace pdi {

cv::Mat meanFilter(
    const cv::Mat& image,
    int neighborhoodSize
);


cv::Mat medianFilter(
    const cv::Mat& image,
    int neighborhoodSize
);

}
