#pragma once

#include <opencv2/core.hpp>

namespace pdi {

//magnitude contem sqrt(Gx^2 + Gy^2), tambem limitada a [0, 255].

struct GradientResult {
    cv::Mat gx;
    cv::Mat gy;
    cv::Mat magnitude;
};

struct KirschResult {
    cv::Mat responses[8];
    cv::Mat finalResponse;
};

GradientResult roberts(const cv::Mat& image);

GradientResult sobel(const cv::Mat& image);

KirschResult kirsch(const cv::Mat& image);

cv::Mat laplacian(const cv::Mat& image);

}
