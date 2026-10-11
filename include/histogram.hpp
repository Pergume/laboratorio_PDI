#pragma once

#include <array>
#include <opencv2/core.hpp>

namespace pdi {

using Histogram = std::array<unsigned long long, 256>;

Histogram calculateHistogram(const cv::Mat& image);
cv::Mat histogramExpansion(const cv::Mat& image);
cv::Mat histogramEqualization(const cv::Mat& image);

cv::Mat adaptiveContrast(
    const cv::Mat& image,
    double c,
    int neighborhoodSize
);

} // namespace pdi
