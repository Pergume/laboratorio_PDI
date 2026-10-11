#include "histogram.hpp"

#include <algorithm>
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
            ": a imagem deve estar em escala de cinza com 8 bits (CV_8UC1)."
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

int firstUsedIntensity(const pdi::Histogram& histogram) {
    for (int intensity = 0; intensity < 256; ++intensity) {
        if (histogram[static_cast<std::size_t>(intensity)] > 0) {
            return intensity;
        }
    }

    return 0;
}

int lastUsedIntensity(const pdi::Histogram& histogram) {
    for (int intensity = 255; intensity >= 0; --intensity) {
        if (histogram[static_cast<std::size_t>(intensity)] > 0) {
            return intensity;
        }
    }

    return 0;
}

void validateAdaptiveContrastParameters(double c, int neighborhoodSize) {
    if (c < 0.0) {
        throw std::invalid_argument(
            "adaptiveContrast: a constante c deve ser maior ou igual a zero."
        );
    }

    if (neighborhoodSize <= 0 || neighborhoodSize % 2 == 0) {
        throw std::invalid_argument(
            "adaptiveContrast: o tamanho da vizinhanca deve ser um inteiro positivo impar."
        );
    }
}

double localMean(
    const cv::Mat& image,
    int centerRow,
    int centerCol,
    int radius
) {
    const int firstRow = std::max(0, centerRow - radius);
    const int lastRow = std::min(image.rows - 1, centerRow + radius);
    const int firstCol = std::max(0, centerCol - radius);
    const int lastCol = std::min(image.cols - 1, centerCol + radius);

    unsigned long long sum = 0;
    unsigned long long count = 0;

    for (int row = firstRow; row <= lastRow; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);

        for (int col = firstCol; col <= lastCol; ++col) {
            sum += inputRow[col];
            ++count;
        }
    }

    return static_cast<double>(sum) / static_cast<double>(count);
}

} // namespace

namespace pdi {

Histogram calculateHistogram(const cv::Mat& image) {
    validateInput(image, "calculateHistogram");

    Histogram histogram{};

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const unsigned char intensity = inputRow[col];
            ++histogram[static_cast<std::size_t>(intensity)];
        }
    }

    return histogram;
}

cv::Mat histogramExpansion(const cv::Mat& image) {
    validateInput(image, "histogramExpansion");

    const Histogram histogram = calculateHistogram(image);
    const int rMin = firstUsedIntensity(histogram);
    const int rMax = lastUsedIntensity(histogram);

    if (rMin == rMax) {
        return image.clone();
    }

    cv::Mat result(image.rows, image.cols, CV_8UC1);
    const double range = static_cast<double>(rMax - rMin);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const double r = static_cast<double>(inputRow[col]);
            const double s = ((r - rMin) / range) * 255.0;
            outputRow[col] = clampToByte(s);
        }
    }

    return result;
}

cv::Mat histogramEqualization(const cv::Mat& image) {
    validateInput(image, "histogramEqualization");

    const Histogram histogram = calculateHistogram(image);
    const unsigned long long totalPixels =
        static_cast<unsigned long long>(image.rows) *
        static_cast<unsigned long long>(image.cols);

    std::array<unsigned long long, 256> cumulative{};
    unsigned long long runningTotal = 0;

    for (std::size_t intensity = 0; intensity < histogram.size(); ++intensity) {
        runningTotal += histogram[intensity];
        cumulative[intensity] = runningTotal;
    }

    unsigned long long cdfMin = 0;
    for (std::size_t intensity = 0; intensity < cumulative.size(); ++intensity) {
        if (histogram[intensity] > 0) {
            cdfMin = cumulative[intensity];
            break;
        }
    }

    if (totalPixels == cdfMin) {
        return image.clone();
    }

    std::array<unsigned char, 256> mapping{};
    const double denominator =
        static_cast<double>(totalPixels - cdfMin);

    for (std::size_t intensity = 0; intensity < mapping.size(); ++intensity) {
        if (cumulative[intensity] < cdfMin) {
            mapping[intensity] = 0;
            continue;
        }

        const double normalized =
            static_cast<double>(cumulative[intensity] - cdfMin) /
            denominator;

        mapping[intensity] = clampToByte(normalized * 255.0);
    }

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            outputRow[col] = mapping[inputRow[col]];
        }
    }

    return result;
}

cv::Mat adaptiveContrast(
    const cv::Mat& image,
    double c,
    int neighborhoodSize
) {
    validateInput(image, "adaptiveContrast");
    validateAdaptiveContrastParameters(c, neighborhoodSize);

    cv::Mat result(image.rows, image.cols, CV_8UC1);
    const int radius = neighborhoodSize / 2;

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* inputRow = image.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const double mean = localMean(image, row, col, radius);
            const double r = static_cast<double>(inputRow[col]);

            // Realce de contraste local:
            // s = media_local + c * (r - media_local)
            const double s = mean + c * (r - mean);

            outputRow[col] = clampToByte(s);
        }
    }

    return result;
}

} 
