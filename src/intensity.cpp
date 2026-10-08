#include "intensity.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void validateInput(const cv::Mat& image, const char* functionName) {
    if (image.empty()) {
        throw std::invalid_argument(
            std::string(functionName) + ": a imagem de entrada esta vazia."
        );
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": a imagem deve estar em escala de cinza com 8 bits por pixel."
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

cv::Mat negative(const cv::Mat& image) {
    validateInput(image, "negative");

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            outputRow[col] = static_cast<unsigned char>(255 - inputRow[col]);
        }
    }

    return result;
}

cv::Mat threshold(const cv::Mat& image, unsigned char thresholdValue) {
    validateInput(image, "threshold");

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            outputRow[col] = inputRow[col] >= thresholdValue ? 255 : 0;
        }
    }

    return result;
}

cv::Mat gammaTransform(const cv::Mat& image, double gamma, double c) {
    validateInput(image, "gammaTransform");

    if (gamma <= 0.0) {
        throw std::invalid_argument("gammaTransform: gamma deve ser maior que zero.");
    }

    if (c < 0.0) {
        throw std::invalid_argument("gammaTransform: c nao pode ser negativo.");
    }

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const double r = static_cast<double>(inputRow[col]) / 255.0;
            const double s = c * std::pow(r, gamma);

            outputRow[col] = clampToByte(s * 255.0);
        }
    }

    return result;
}

cv::Mat logTransform(const cv::Mat& image, double c) {
    validateInput(image, "logTransform");

    if (c < 0.0) {
        throw std::invalid_argument("logTransform: c nao pode ser negativo.");
    }

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    // Como r e normalizado para [0, 1], dividir por log(2) faz com que
    // log(1 + r) tambem ocupe [0, 1] quando c = 1.
    const double normalization = std::log(2.0);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const double r = static_cast<double>(inputRow[col]) / 255.0;
            const double s = c * (std::log(1.0 + r) / normalization);

            outputRow[col] = clampToByte(s * 255.0);
        }
    }

    return result;
}

cv::Mat contrastStretch(
    const cv::Mat& image,
    unsigned char rMin,
    unsigned char rMax
) {
    validateInput(image, "contrastStretch");

    if (rMin >= rMax) {
        throw std::invalid_argument(
            "contrastStretch: rMin deve ser menor que rMax."
        );
    }

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const unsigned char r = inputRow[col];

            if (r <= rMin) {
                outputRow[col] = 0;
            } else if (r >= rMax) {
                outputRow[col] = 255;
            } else {
                const double normalized =
                    static_cast<double>(r - rMin) /
                    static_cast<double>(rMax - rMin);

                outputRow[col] = clampToByte(normalized * 255.0);
            }
        }
    }

    return result;
}

} // namespace pdi
