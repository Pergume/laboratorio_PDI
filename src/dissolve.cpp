#include "dissolve.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void validateImage(
    const cv::Mat& image,
    const char* imageName
) {
    if (image.empty()) {
        throw std::invalid_argument(
            std::string(imageName) + ": imagem vazia."
        );
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            std::string(imageName) +
            ": esperado formato CV_8UC1 (escala de cinza, 8 bits)."
        );
    }
}

void validateImagePair(
    const cv::Mat& imageA,
    const cv::Mat& imageB
) {
    validateImage(imageA, "imageA");
    validateImage(imageB, "imageB");

    if (imageA.rows != imageB.rows || imageA.cols != imageB.cols) {
        throw std::invalid_argument(
            "As duas imagens do dissolve devem possuir as mesmas dimensoes."
        );
    }
}

void validateWeight(double weight, const char* parameterName) {
    if (weight < 0.0 || weight > 1.0) {
        throw std::invalid_argument(
            std::string(parameterName) + " deve estar no intervalo [0, 1]."
        );
    }
}

unsigned char clampToByte(double value) {
    if (value < 0.0) {
        return 0;
    }

    if (value > 255.0) {
        return 255;
    }

    return static_cast<unsigned char>(std::round(value));
}

} // namespace

namespace pdi {

cv::Mat uniformDissolve(
    const cv::Mat& imageA,
    const cv::Mat& imageB,
    double alpha
) {
    validateImagePair(imageA, imageB);
    validateWeight(alpha, "alpha");

    cv::Mat result(imageA.rows, imageA.cols, CV_8UC1);

    for (int row = 0; row < imageA.rows; ++row) {
        const unsigned char* rowA = imageA.ptr<unsigned char>(row);
        const unsigned char* rowB = imageB.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < imageA.cols; ++col) {
            const double valueA = static_cast<double>(rowA[col]);
            const double valueB = static_cast<double>(rowB[col]);

            const double blended =
                (1.0 - alpha) * valueA +
                alpha * valueB;

            outputRow[col] = clampToByte(blended);
        }
    }

    return result;
}

cv::Mat nonUniformDissolve(
    const cv::Mat& imageA,
    const cv::Mat& imageB,
    const cv::Mat& weightMap
) {
    validateImagePair(imageA, imageB);

    if (weightMap.empty()) {
        throw std::invalid_argument("weightMap: matriz de pesos vazia.");
    }

    if (weightMap.type() != CV_64FC1) {
        throw std::invalid_argument(
            "weightMap deve possuir formato CV_64FC1."
        );
    }

    if (weightMap.rows != imageA.rows || weightMap.cols != imageA.cols) {
        throw std::invalid_argument(
            "weightMap deve possuir as mesmas dimensoes das imagens."
        );
    }

    cv::Mat result(imageA.rows, imageA.cols, CV_8UC1);

    for (int row = 0; row < imageA.rows; ++row) {
        const unsigned char* rowA = imageA.ptr<unsigned char>(row);
        const unsigned char* rowB = imageB.ptr<unsigned char>(row);
        const double* weightRow = weightMap.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < imageA.cols; ++col) {
            const double weight = weightRow[col];
            validateWeight(weight, "peso do dissolve nao uniforme");

            const double valueA = static_cast<double>(rowA[col]);
            const double valueB = static_cast<double>(rowB[col]);

            const double blended =
                (1.0 - weight) * valueA +
                weight * valueB;

            outputRow[col] = clampToByte(blended);
        }
    }

    return result;
}

} // namespace pdi
