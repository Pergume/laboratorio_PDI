#include "sharpening.hpp"

#include "convolution.hpp"
#include "spatial_filters.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void validateImage(
    const cv::Mat& image,
    const char* functionName
) {
    if (image.empty()) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": a imagem de entrada esta vazia."
        );
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            std::string(functionName) +
            ": a imagem deve estar no formato CV_8UC1."
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

cv::Mat createLaplacianKernel() {
    cv::Mat kernel(3, 3, CV_64FC1);

    kernel.ptr<double>(0)[0] =  0.0;
    kernel.ptr<double>(0)[1] = -1.0;
    kernel.ptr<double>(0)[2] =  0.0;

    kernel.ptr<double>(1)[0] = -1.0;
    kernel.ptr<double>(1)[1] =  4.0;
    kernel.ptr<double>(1)[2] = -1.0;

    kernel.ptr<double>(2)[0] =  0.0;
    kernel.ptr<double>(2)[1] = -1.0;
    kernel.ptr<double>(2)[2] =  0.0;

    return kernel;
}

cv::Mat absoluteResponseToByte(const cv::Mat& rawResponse) {
    cv::Mat result(rawResponse.rows, rawResponse.cols, CV_8UC1);

    for (int row = 0; row < rawResponse.rows; ++row) {
        const double* inputRow = rawResponse.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < rawResponse.cols; ++col) {
            outputRow[col] = clampToByte(std::abs(inputRow[col]));
        }
    }

    return result;
}

} // namespace

namespace pdi {

LaplacianSharpenResult laplacianSharpen(const cv::Mat& image) {
    validateImage(image, "laplacianSharpen");

    const cv::Mat kernel = createLaplacianKernel();

    // Mantemos a resposta em double para preservar valores negativos e
    // superiores a 255 antes de combinar com a imagem original.
    const cv::Mat laplacianRaw = convolveRaw(
        image,
        kernel,
        0.0,
        BorderMode::Replicate
    );

    LaplacianSharpenResult result;
    result.laplacianResponse = absoluteResponseToByte(laplacianRaw);
    result.sharpenedImage = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* imageRow = image.ptr<unsigned char>(row);
        const double* laplacianRow = laplacianRaw.ptr<double>(row);
        unsigned char* outputRow = result.sharpenedImage.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            // A máscara usada tem centro positivo (4). Por isso, o
            // aguçamento é feito somando a resposta à imagem original.
            const double sharpenedValue =
                static_cast<double>(imageRow[col]) + laplacianRow[col];

            outputRow[col] = clampToByte(sharpenedValue);
        }
    }

    return result;
}

cv::Mat highBoost(
    const cv::Mat& image,
    double k,
    int neighborhoodSize
) {
    validateImage(image, "highBoost");

    if (k < 0.0) {
        throw std::invalid_argument(
            "highBoost: k deve ser maior ou igual a zero."
        );
    }

    // Reutilizamos o filtro da média manual já implementado para construir
    // a versão suavizada da imagem.
    const cv::Mat blurred = meanFilter(image, neighborhoodSize);

    cv::Mat result(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        const unsigned char* imageRow = image.ptr<unsigned char>(row);
        const unsigned char* blurredRow = blurred.ptr<unsigned char>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            const double original = static_cast<double>(imageRow[col]);
            const double smooth = static_cast<double>(blurredRow[col]);

            // Máscara de altas frequências: detalhes removidos pela suavização.
            const double highFrequencyMask = original - smooth;

            // High Boost: reforça a máscara e a soma novamente à imagem.
            const double boostedValue =
                original + k * highFrequencyMask;

            outputRow[col] = clampToByte(boostedValue);
        }
    }

    return result;
}

}
