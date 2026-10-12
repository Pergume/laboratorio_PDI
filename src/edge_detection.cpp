#include "edge_detection.hpp"

#include "convolution.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

void validateRawPair(
    const cv::Mat& gxRaw,
    const cv::Mat& gyRaw
) {
    if (gxRaw.empty() || gyRaw.empty()) {
        throw std::invalid_argument(
            "As respostas Gx e Gy nao podem estar vazias."
        );
    }

    if (gxRaw.type() != CV_64FC1 || gyRaw.type() != CV_64FC1) {
        throw std::invalid_argument(
            "Gx e Gy devem estar no formato CV_64FC1."
        );
    }

    if (gxRaw.rows != gyRaw.rows || gxRaw.cols != gyRaw.cols) {
        throw std::invalid_argument(
            "Gx e Gy devem possuir as mesmas dimensoes."
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

cv::Mat absoluteResponseToByte(const cv::Mat& rawResponse) {
    if (rawResponse.empty() || rawResponse.type() != CV_64FC1) {
        throw std::invalid_argument(
            "A resposta deve estar no formato CV_64FC1."
        );
    }

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

cv::Mat positiveResponseToByte(const cv::Mat& rawResponse) {
    if (rawResponse.empty() || rawResponse.type() != CV_64FC1) {
        throw std::invalid_argument(
            "A resposta deve estar no formato CV_64FC1."
        );
    }

    cv::Mat result(rawResponse.rows, rawResponse.cols, CV_8UC1);

    for (int row = 0; row < rawResponse.rows; ++row) {
        const double* inputRow = rawResponse.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < rawResponse.cols; ++col) {
            outputRow[col] = clampToByte(inputRow[col]);
        }
    }

    return result;
}

cv::Mat gradientMagnitude(
    const cv::Mat& gxRaw,
    const cv::Mat& gyRaw
) {
    validateRawPair(gxRaw, gyRaw);

    cv::Mat result(gxRaw.rows, gxRaw.cols, CV_8UC1);

    for (int row = 0; row < gxRaw.rows; ++row) {
        const double* gxRow = gxRaw.ptr<double>(row);
        const double* gyRow = gyRaw.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < gxRaw.cols; ++col) {
            const double gx = gxRow[col];
            const double gy = gyRow[col];

            const double magnitude = std::sqrt(gx * gx + gy * gy);
            outputRow[col] = clampToByte(magnitude);
        }
    }

    return result;
}

cv::Mat createKernel3x3(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22
) {
    cv::Mat kernel(3, 3, CV_64FC1);

    kernel.ptr<double>(0)[0] = a00;
    kernel.ptr<double>(0)[1] = a01;
    kernel.ptr<double>(0)[2] = a02;
    kernel.ptr<double>(1)[0] = a10;
    kernel.ptr<double>(1)[1] = a11;
    kernel.ptr<double>(1)[2] = a12;
    kernel.ptr<double>(2)[0] = a20;
    kernel.ptr<double>(2)[1] = a21;
    kernel.ptr<double>(2)[2] = a22;

    return kernel;
}

cv::Mat createRobertsGxKernel() {
    cv::Mat kernel(2, 2, CV_64FC1);

    kernel.ptr<double>(0)[0] =  1.0;
    kernel.ptr<double>(0)[1] =  0.0;
    kernel.ptr<double>(1)[0] =  0.0;
    kernel.ptr<double>(1)[1] = -1.0;

    return kernel;
}

cv::Mat createRobertsGyKernel() {
    cv::Mat kernel(2, 2, CV_64FC1);

    kernel.ptr<double>(0)[0] =  0.0;
    kernel.ptr<double>(0)[1] =  1.0;
    kernel.ptr<double>(1)[0] = -1.0;
    kernel.ptr<double>(1)[1] =  0.0;

    return kernel;
}

cv::Mat createSobelGxKernel() {
    return createKernel3x3(
        -1.0,  0.0,  1.0,
        -2.0,  0.0,  2.0,
        -1.0,  0.0,  1.0
    );
}

cv::Mat createSobelGyKernel() {
    return createKernel3x3(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );
}

std::array<cv::Mat, 8> createKirschKernels() {
    // Oito orientacoes compass. Como convolveRaw executa convolucao verdadeira,
    // a orientacao espacial efetiva e invertida em 180 graus. O conjunto completo
    // permanece equivalente e a resposta final por maximo nao e alterada.
    return {
        createKernel3x3(
             5.0,  5.0,  5.0,
            -3.0,  0.0, -3.0,
            -3.0, -3.0, -3.0
        ),
        createKernel3x3(
             5.0,  5.0, -3.0,
             5.0,  0.0, -3.0,
            -3.0, -3.0, -3.0
        ),
        createKernel3x3(
             5.0, -3.0, -3.0,
             5.0,  0.0, -3.0,
             5.0, -3.0, -3.0
        ),
        createKernel3x3(
            -3.0, -3.0, -3.0,
             5.0,  0.0, -3.0,
             5.0,  5.0, -3.0
        ),
        createKernel3x3(
            -3.0, -3.0, -3.0,
            -3.0,  0.0, -3.0,
             5.0,  5.0,  5.0
        ),
        createKernel3x3(
            -3.0, -3.0, -3.0,
            -3.0,  0.0,  5.0,
            -3.0,  5.0,  5.0
        ),
        createKernel3x3(
            -3.0, -3.0,  5.0,
            -3.0,  0.0,  5.0,
            -3.0, -3.0,  5.0
        ),
        createKernel3x3(
            -3.0,  5.0,  5.0,
            -3.0,  0.0,  5.0,
            -3.0, -3.0, -3.0
        )
    };
}

cv::Mat createLaplacianKernel() {
    // Laplaciano de 4-vizinhos com centro positivo.
    return createKernel3x3(
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0
    );
}

pdi::GradientResult calculateGradient(
    const cv::Mat& image,
    const cv::Mat& gxKernel,
    const cv::Mat& gyKernel
) {
    const cv::Mat gxRaw = pdi::convolveRaw(
        image,
        gxKernel,
        0.0,
        pdi::BorderMode::Replicate
    );

    const cv::Mat gyRaw = pdi::convolveRaw(
        image,
        gyKernel,
        0.0,
        pdi::BorderMode::Replicate
    );

    pdi::GradientResult result;
    result.gx = absoluteResponseToByte(gxRaw);
    result.gy = absoluteResponseToByte(gyRaw);
    result.magnitude = gradientMagnitude(gxRaw, gyRaw);

    return result;
}

} // namespace

namespace pdi {

GradientResult roberts(const cv::Mat& image) {
    const cv::Mat gxKernel = createRobertsGxKernel();
    const cv::Mat gyKernel = createRobertsGyKernel();

    return calculateGradient(image, gxKernel, gyKernel);
}

GradientResult sobel(const cv::Mat& image) {
    const cv::Mat gxKernel = createSobelGxKernel();
    const cv::Mat gyKernel = createSobelGyKernel();

    return calculateGradient(image, gxKernel, gyKernel);
}

KirschResult kirsch(const cv::Mat& image) {
    const std::array<cv::Mat, 8> kernels = createKirschKernels();
    std::array<cv::Mat, 8> rawResponses;

    KirschResult result;

    for (int direction = 0; direction < 8; ++direction) {
        rawResponses[direction] = convolveRaw(
            image,
            kernels[direction],
            0.0,
            BorderMode::Replicate
        );

        // Para cada mascara mostramos apenas respostas positivas. A mascara
        // oposta cobre a mesma transicao no sentido contrario.
        result.responses[direction] =
            positiveResponseToByte(rawResponses[direction]);
    }

    result.finalResponse = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row < image.rows; ++row) {
        unsigned char* outputRow = result.finalResponse.ptr<unsigned char>(row);

        for (int col = 0; col < image.cols; ++col) {
            double maximumResponse = rawResponses[0].ptr<double>(row)[col];

            for (int direction = 1; direction < 8; ++direction) {
                const double response =
                    rawResponses[direction].ptr<double>(row)[col];

                if (response > maximumResponse) {
                    maximumResponse = response;
                }
            }

            outputRow[col] = clampToByte(maximumResponse);
        }
    }

    return result;
}

cv::Mat laplacian(const cv::Mat& image) {
    const cv::Mat kernel = createLaplacianKernel();

    const cv::Mat rawResponse = convolveRaw(
        image,
        kernel,
        0.0,
        BorderMode::Replicate
    );

    
    return absoluteResponseToByte(rawResponse);
}

}
