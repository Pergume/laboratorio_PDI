#pragma once

#include <opencv2/core.hpp>

namespace pdi {

// Transformacoes pontuais exigidas pela atividade.
// Todas recebem uma imagem em escala de cinza de 8 bits (CV_8UC1)
// e retornam uma nova imagem no mesmo formato.
cv::Mat negative(const cv::Mat& image);
cv::Mat threshold(const cv::Mat& image, unsigned char thresholdValue);
cv::Mat gammaTransform(const cv::Mat& image, double gamma, double c = 1.0);
cv::Mat logTransform(const cv::Mat& image, double c = 1.0);
cv::Mat contrastStretch(const cv::Mat& image, unsigned char rMin, unsigned char rMax);

} // namespace pdi
