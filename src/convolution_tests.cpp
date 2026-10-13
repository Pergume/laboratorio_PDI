#include "convolution_tests.hpp"

#include "convolution.hpp"
#include "image_io.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

cv::Mat createSharpeningKernel(int c, int d) {
    if (c <= 0 || d <= 0) {
        throw std::invalid_argument(
            "createSharpeningKernel: c e d devem ser inteiros positivos."
        );
    }

    const double negativeC = -static_cast<double>(c);
    const double center = 8.0 * static_cast<double>(c) +
                          static_cast<double>(d);

    return createKernel3x3(
        negativeC, negativeC, negativeC,
        negativeC, center,    negativeC,
        negativeC, negativeC, negativeC
    );
}

cv::Mat createReliefKernelH1() {
    return createKernel3x3(
        0.0, 0.0,  0.0,
        0.0, 1.0,  0.0,
        0.0, 0.0, -1.0
    );
}

cv::Mat createReliefKernelH2() {
    return createKernel3x3(
         0.0,  0.0, 2.0,
         0.0, -1.0, 0.0,
        -1.0,  0.0, 0.0
    );
}

cv::Mat createEdgeKernelH3() {
    return createKernel3x3(
        -1.0, -1.0, 0.0,
         1.0,  0.0, 1.0,
         0.0,  1.0, 1.0
    );
}

cv::Mat createEdgeKernelH4() {
    return createKernel3x3(
         0.0, -1.0,  0.0,
        -1.0, -4.0, -1.0,
         0.0, -1.0,  0.0
    );
}

void validateImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument(
            "runMandatoryConvolutionTest: a imagem de entrada esta vazia."
        );
    }

    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            "runMandatoryConvolutionTest: a imagem deve ser CV_8UC1."
        );
    }
}

unsigned char clampMagnitudeToByte(double value) {
    const double absolute = std::abs(value);
    if (absolute > 255.0) {
        return 255;
    }
    return static_cast<unsigned char>(std::round(absolute));
}

cv::Mat rawAbsoluteToByte(const cv::Mat& raw) {
    cv::Mat result(raw.rows, raw.cols, CV_8UC1);

    for (int row = 0; row < raw.rows; ++row) {
        const double* inputRow = raw.ptr<double>(row);
        unsigned char* outputRow = result.ptr<unsigned char>(row);

        for (int col = 0; col < raw.cols; ++col) {
            outputRow[col] = clampMagnitudeToByte(inputRow[col]);
        }
    }

    return result;
}

} // namespace

namespace pdi {

cv::Mat runMandatoryConvolutionTest(
    const cv::Mat& image,
    MandatoryConvolutionTest test
) {
    validateImage(image);

    switch (test) {
        case MandatoryConvolutionTest::SharpenC1D1:
            return convolve(
                image,
                createSharpeningKernel(1, 1),
                0.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::SharpenC2D1:
            return convolve(
                image,
                createSharpeningKernel(2, 1),
                0.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::SharpenC1D2:
            return convolve(
                image,
                createSharpeningKernel(1, 2),
                0.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::ReliefH1:
            return convolve(
                image,
                createReliefKernelH1(),
                128.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::ReliefH2:
            return convolve(
                image,
                createReliefKernelH2(),
                128.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::EdgeH3:
            return convolve(
                image,
                createEdgeKernelH3(),
                0.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::EdgeH4Clipped:
            return convolve(
                image,
                createEdgeKernelH4(),
                0.0,
                BorderMode::Replicate
            );

        case MandatoryConvolutionTest::EdgeH4Absolute: {
            const cv::Mat raw = convolveRaw(
                image,
                createEdgeKernelH4(),
                0.0,
                BorderMode::Replicate
            );
            return rawAbsoluteToByte(raw);
        }
    }

    throw std::invalid_argument("Teste obrigatorio desconhecido.");
}

std::string mandatoryConvolutionTestName(MandatoryConvolutionTest test) {
    switch (test) {
        case MandatoryConvolutionTest::SharpenC1D1:
            return "Agucamento c=1 d=1";
        case MandatoryConvolutionTest::SharpenC2D1:
            return "Agucamento c=2 d=1";
        case MandatoryConvolutionTest::SharpenC1D2:
            return "Agucamento c=1 d=2";
        case MandatoryConvolutionTest::ReliefH1:
            return "Relevo h1";
        case MandatoryConvolutionTest::ReliefH2:
            return "Relevo h2";
        case MandatoryConvolutionTest::EdgeH3:
            return "Bordas h3";
        case MandatoryConvolutionTest::EdgeH4Clipped:
            return "Bordas h4 - saturacao";
        case MandatoryConvolutionTest::EdgeH4Absolute:
            return "Bordas h4 - modulo da resposta";
    }
    return "Teste";
}

void runMandatoryConvolutionTests(
    const cv::Mat& image,
    const std::string& outputDirectory
) {
    validateImage(image);

    saveImage(outputDirectory + "33_teste_convolucao_original.png", image);

    saveImage(
        outputDirectory + "34_teste_agucamento_c1_d1.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::SharpenC1D1)
    );
    saveImage(
        outputDirectory + "35_teste_agucamento_c2_d1.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::SharpenC2D1)
    );
    saveImage(
        outputDirectory + "36_teste_agucamento_c1_d2.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::SharpenC1D2)
    );
    saveImage(
        outputDirectory + "37_teste_relevo_h1.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::ReliefH1)
    );
    saveImage(
        outputDirectory + "38_teste_relevo_h2.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::ReliefH2)
    );
    saveImage(
        outputDirectory + "39_teste_bordas_h3.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::EdgeH3)
    );
    saveImage(
        outputDirectory + "40_teste_bordas_h4.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::EdgeH4Clipped)
    );
    saveImage(
        outputDirectory + "41_teste_bordas_h4_modulo.png",
        runMandatoryConvolutionTest(image, MandatoryConvolutionTest::EdgeH4Absolute)
    );

    std::cout << "Testes obrigatorios de convolucao salvos em "
              << outputDirectory << "\n";
    std::cout << "  40: h4 com saturacao direta (pode ficar preto).\n";
    std::cout << "  41: h4 com |resposta bruta| para visualizacao.\n";
}

} // namespace pdi
