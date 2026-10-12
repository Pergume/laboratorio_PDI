#include "convolution.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

void validateImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("convolve: a imagem de entrada esta vazia.");
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            "convolve: a imagem deve estar em escala de cinza com 8 bits."
        );
    }
}

void validateKernel(const cv::Mat& kernel) {
    if (kernel.empty()) {
        throw std::invalid_argument("convolve: a mascara esta vazia.");
    }

    if (kernel.type() != CV_64FC1) {
        throw std::invalid_argument(
            "convolve: a mascara deve possuir um canal do tipo double (CV_64FC1)."
        );
    }

    if (kernel.rows <= 0 || kernel.cols <= 0) {
        throw std::invalid_argument(
            "convolve: a mascara deve possuir dimensoes positivas."
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

int reflectIndex(int index, int size) {
    if (size <= 1) {
        return 0;
    }

    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index - 1;
        } else {
            index = 2 * size - index - 1;
        }
    }

    return index;
}

unsigned char pixelAt(
    const cv::Mat& image,
    int row,
    int col,
    pdi::BorderMode borderMode
) {
    const bool inside =
        row >= 0 && row < image.rows &&
        col >= 0 && col < image.cols;

    if (inside) {
        return image.ptr<unsigned char>(row)[col];
    }

    switch (borderMode) {
        case pdi::BorderMode::ZeroPadding:
            return 0;

        case pdi::BorderMode::Replicate: {
            const int validRow = std::clamp(row, 0, image.rows - 1);
            const int validCol = std::clamp(col, 0, image.cols - 1);
            return image.ptr<unsigned char>(validRow)[validCol];
        }

        case pdi::BorderMode::Reflect: {
            const int validRow = reflectIndex(row, image.rows);
            const int validCol = reflectIndex(col, image.cols);
            return image.ptr<unsigned char>(validRow)[validCol];
        }
    }

    throw std::invalid_argument("convolve: modo de borda invalido.");
}

} // namespace

namespace pdi {

cv::Mat convolveRaw(
    const cv::Mat& image,
    const cv::Mat& kernel,
    double offset,
    BorderMode borderMode
) {
    validateImage(image);
    validateKernel(kernel);

    cv::Mat result(image.rows, image.cols, CV_64FC1);

    // Para dimensoes pares, a ancora fica na posicao rows/2, cols/2.
    const int anchorRow = kernel.rows / 2;
    const int anchorCol = kernel.cols / 2;

    for (int row = 0; row < image.rows; ++row) {
        double* outputRow = result.ptr<double>(row);

        for (int col = 0; col < image.cols; ++col) {
            double sum = offset;

            for (int kernelRow = 0; kernelRow < kernel.rows; ++kernelRow) {
                for (int kernelCol = 0; kernelCol < kernel.cols; ++kernelCol) {
                    const int imageRow = row + kernelRow - anchorRow;
                    const int imageCol = col + kernelCol - anchorCol;

                    // Convolucao verdadeira: a mascara e invertida nos dois eixos.
                    const int flippedRow = kernel.rows - 1 - kernelRow;
                    const int flippedCol = kernel.cols - 1 - kernelCol;

                    const double coefficient =
                        kernel.ptr<double>(flippedRow)[flippedCol];

                    const double pixel = static_cast<double>(
                        pixelAt(image, imageRow, imageCol, borderMode)
                    );

                    sum += pixel * coefficient;
                }
            }

            outputRow[col] = sum;
        }
    }

    return result;
}

cv::Mat convolve(
    const cv::Mat& image,
    const cv::Mat& kernel,
    double offset,
    BorderMode borderMode
) {
    const cv::Mat raw = convolveRaw(image, kernel, offset, borderMode);
    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < raw.rows; ++row) {
        const double* inputRow = raw.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < raw.cols; ++col) {
            outputRow[col] = clampToByte(inputRow[col]);
        }
    }

    return result;
}

}
