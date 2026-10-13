#pragma once

#include <array>
#include <opencv2/core.hpp>

namespace pdi {

using Histogram = std::array<unsigned long long, 256>;

Histogram calculateHistogram(const cv::Mat& image);
cv::Mat histogramExpansion(const cv::Mat& image);
cv::Mat histogramEqualization(const cv::Mat& image);

struct LocalStatistics {
    cv::Mat mean; // CV_64FC1, intensidades de 0 a 255.
    cv::Mat standardDeviation; // CV_64FC1, desvio padrao populacional.
};

cv::Mat adaptiveContrast(
    const cv::Mat& image,
    double c,
    int neighborhoodSize,
    LocalStatistics* statistics = nullptr
);

} // namespace pdi
