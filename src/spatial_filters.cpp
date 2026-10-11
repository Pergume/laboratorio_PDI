#include "spatial_filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validateInput(
    const cv::Mat& image,
    const char* functionName
) {
    if (image.empty()) {
        throw std::invalid_argument(
            std::string(functionName) + ": a imagem de entrada esta vazia."
        );
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": a imagem deve estar em escala de cinza com 8 bits."
        );
    }
}

void validateNeighborhoodSize(
    int neighborhoodSize,
    const char* functionName
) {
    if (neighborhoodSize <= 0 || neighborhoodSize % 2 == 0) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": o tamanho da vizinhanca deve ser positivo e impar."
        );
    }
}

void neighborhoodLimits(
    const cv::Mat& image,
    int centerRow,
    int centerCol,
    int radius,
    int& firstRow,
    int& lastRow,
    int& firstCol,
    int& lastCol
) {
    firstRow = std::max(0, centerRow - radius);
    lastRow = std::min(image.rows - 1, centerRow + radius);
    firstCol = std::max(0, centerCol - radius);
    lastCol = std::min(image.cols - 1, centerCol + radius);
}

} // namespace

namespace pdi {

cv::Mat meanFilter(
    const cv::Mat& image,
    int neighborhoodSize
) {
    validateInput(image, "meanFilter");
    validateNeighborhoodSize(neighborhoodSize, "meanFilter");

    const int radius = neighborhoodSize / 2;
    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            int firstRow = 0;
            int lastRow = 0;
            int firstCol = 0;
            int lastCol = 0;

            neighborhoodLimits(
                image,
                row,
                col,
                radius,
                firstRow,
                lastRow,
                firstCol,
                lastCol
            );

            unsigned long long sum = 0;
            unsigned long long count = 0;

            for (int neighborRow = firstRow;
                 neighborRow <= lastRow;
                 ++neighborRow) {
                const unsigned char* inputRow =
                    image.ptr<unsigned char>(neighborRow);

                for (int neighborCol = firstCol;
                     neighborCol <= lastCol;
                     ++neighborCol) {
                    sum += inputRow[neighborCol];
                    ++count;
                }
            }

            const double mean =
                static_cast<double>(sum) /
                static_cast<double>(count);

            outputRow[col] = static_cast<unsigned char>(std::round(mean));
        }
    }

    return result;
}

cv::Mat medianFilter(
    const cv::Mat& image,
    int neighborhoodSize
) {
    validateInput(image, "medianFilter");
    validateNeighborhoodSize(neighborhoodSize, "medianFilter");

    const int radius = neighborhoodSize / 2;
    cv::Mat result(image.rows, image.cols, CV_8UC1);

    std::vector<unsigned char> values;
    values.reserve(
        static_cast<std::size_t>(neighborhoodSize) *
        static_cast<std::size_t>(neighborhoodSize)
    );

    for (int row = 0; row < image.rows; ++row) {
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            int firstRow = 0;
            int lastRow = 0;
            int firstCol = 0;
            int lastCol = 0;

            neighborhoodLimits(
                image,
                row,
                col,
                radius,
                firstRow,
                lastRow,
                firstCol,
                lastCol
            );

            values.clear();

            for (int neighborRow = firstRow;
                 neighborRow <= lastRow;
                 ++neighborRow) {
                const unsigned char* inputRow =
                    image.ptr<unsigned char>(neighborRow);

                for (int neighborCol = firstCol;
                     neighborCol <= lastCol;
                     ++neighborCol) {
                    values.push_back(inputRow[neighborCol]);
                }
            }

            std::sort(values.begin(), values.end());

            const std::size_t middle = values.size() / 2;

            if (values.size() % 2 == 1) {
                outputRow[col] = values[middle];
            } else {
                const int lower = values[middle - 1];
                const int upper = values[middle];
                outputRow[col] = static_cast<unsigned char>(
                    (lower + upper) / 2
                );
            }
        }
    }

    return result;
}

} 
