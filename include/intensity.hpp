#pragma once

#include <opencv2/core.hpp>

namespace pdi {

// Transformacoes pontuais exigidas pela atividade.
// As implementacoes serao adicionadas nas proximas etapas.
cv::Mat negative(const cv::Mat& image);
cv::Mat threshold(const cv::Mat& image, unsigned char thresholdValue);
cv::Mat gammaTransform(const cv::Mat& image, double gamma, double c = 1.0);
cv::Mat logTransform(const cv::Mat& image, double c = 1.0);
cv::Mat contrastStretch(const cv::Mat& image, unsigned char rMin, unsigned char rMax);

} // namespace pdi
