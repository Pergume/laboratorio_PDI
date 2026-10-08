#pragma once

#include <opencv2/core.hpp>

namespace pdi {

struct LaplacianSharpenResult {
    cv::Mat laplacianResponse;
    cv::Mat sharpenedImage;
};

LaplacianSharpenResult laplacianSharpen(const cv::Mat& image);
cv::Mat highBoost(const cv::Mat& image, double k, int neighborhoodSize);

} // namespace pdi
