#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace pdi {

enum class MandatoryConvolutionTest {
    SharpenC1D1,
    SharpenC2D1,
    SharpenC1D2,
    ReliefH1,
    ReliefH2,
    EdgeH3,
    EdgeH4Clipped,
    EdgeH4Absolute
};

// Executa um teste e retorna a imagem resultante sem salva-la.
cv::Mat runMandatoryConvolutionTest(
    const cv::Mat& image,
    MandatoryConvolutionTest test
);

std::string mandatoryConvolutionTestName(MandatoryConvolutionTest test);

/**
 * @brief Executa os testes obrigatorios de convolucao definidos na AV1.
 *
 * Usa a mesma imagem de entrada nos tres grupos de testes:
 * 1) agucamento com diferentes valores de c e d;
 * 2) duas mascaras de relevo;
 * 3) duas mascaras de deteccao de bordas.
 *
 * Os resultados sao salvos no diretorio informado.
 */
void runMandatoryConvolutionTests(
    const cv::Mat& image,
    const std::string& outputDirectory
);

} // namespace pdi
