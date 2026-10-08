#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace pdi {

cv::Mat loadGrayscale8Bit(const std::string& path);
void saveImage(const std::string& path, const cv::Mat& image);
bool isGrayscale8Bit(const cv::Mat& image) noexcept;

} // namespace pdi
