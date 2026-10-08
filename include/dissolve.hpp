#pragma once

#include <opencv2/core.hpp>

namespace pdi {

cv::Mat uniformDissolve(
    const cv::Mat& imageA,
    const cv::Mat& imageB,
    double alpha
);

cv::Mat nonUniformDissolve(
    const cv::Mat& imageA,
    const cv::Mat& imageB,
    const cv::Mat& weightMap
);

} // namespace pdi
