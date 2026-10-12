#include "convolution.hpp"
#include "edge_detection.hpp"
#include "histogram.hpp"
#include "image_io.hpp"
#include "intensity.hpp"
#include "spatial_filters.hpp"
#include "sharpening.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {
constexpr const char* INPUT_PATH = "assets/input/input.tif";
constexpr const char* OUTPUT_DIRECTORY = "assets/output/";

constexpr unsigned char THRESHOLD_VALUE = 128;
constexpr double GAMMA_VALUE = 0.5;
constexpr double GAMMA_C = 1.0;
constexpr double LOG_C = 1.0;
constexpr unsigned char CONTRAST_MIN = 60;
constexpr unsigned char CONTRAST_MAX = 190;

constexpr double ADAPTIVE_CONTRAST_C = 1.5;
constexpr int ADAPTIVE_NEIGHBORHOOD_SIZE = 3;

constexpr int MEAN_NEIGHBORHOOD_SIZE = 3;
constexpr int MEDIAN_NEIGHBORHOOD_SIZE = 3;

constexpr double CONVOLUTION_OFFSET = 0.0;

constexpr double HIGH_BOOST_K = 1.5;
constexpr int HIGH_BOOST_NEIGHBORHOOD_SIZE = 3;
}

int main() {
    try {
        const cv::Mat image = pdi::loadGrayscale8Bit(INPUT_PATH);

        std::cout << "Imagem carregada com sucesso.\n";
        std::cout << "Resolucao: " << image.cols << " x " << image.rows << "\n";
        std::cout << "Formato: escala de cinza, 8 bits.\n\n";

        // Transformacoes de intensidade.
        const cv::Mat negativeImage = pdi::negative(image);
        const cv::Mat thresholdImage = pdi::threshold(image, THRESHOLD_VALUE);
        const cv::Mat gammaImage = pdi::gammaTransform(image, GAMMA_VALUE, GAMMA_C);
        const cv::Mat logImage = pdi::logTransform(image, LOG_C);
        const cv::Mat contrastImage =
            pdi::contrastStretch(image, CONTRAST_MIN, CONTRAST_MAX);

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "01_negativo.png",
            negativeImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "02_limiarizacao.png",
            thresholdImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "03_gamma.png",
            gammaImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "04_logaritmo.png",
            logImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "05_alargamento_contraste.png",
            contrastImage
        );

        // Histograma.
        const pdi::Histogram originalHistogram = pdi::calculateHistogram(image);
        const cv::Mat expandedImage = pdi::histogramExpansion(image);
        const cv::Mat equalizedImage = pdi::histogramEqualization(image);

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "06_expansao_histograma.png",
            expandedImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "07_equalizacao_histograma.png",
            equalizedImage
        );

        // Contraste adaptativo.
        const cv::Mat adaptiveContrastImage = pdi::adaptiveContrast(
            image,
            ADAPTIVE_CONTRAST_C,
            ADAPTIVE_NEIGHBORHOOD_SIZE
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "08_contraste_adaptativo.png",
            adaptiveContrastImage
        );

        // Filtragem espacial.
        const cv::Mat meanImage = pdi::meanFilter(
            image,
            MEAN_NEIGHBORHOOD_SIZE
        );
        const cv::Mat medianImage = pdi::medianFilter(
            image,
            MEDIAN_NEIGHBORHOOD_SIZE
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "09_filtro_media.png",
            meanImage
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "10_filtro_mediana.png",
            medianImage
        );

        // Convolucao generica.
        cv::Mat convolutionKernel(3, 3, CV_64FC1);
        convolutionKernel.ptr<double>(0)[0] = -1.0;
        convolutionKernel.ptr<double>(0)[1] = -1.0;
        convolutionKernel.ptr<double>(0)[2] = -1.0;
        convolutionKernel.ptr<double>(1)[0] = -1.0;
        convolutionKernel.ptr<double>(1)[1] =  9.0;
        convolutionKernel.ptr<double>(1)[2] = -1.0;
        convolutionKernel.ptr<double>(2)[0] = -1.0;
        convolutionKernel.ptr<double>(2)[1] = -1.0;
        convolutionKernel.ptr<double>(2)[2] = -1.0;

        const cv::Mat convolutionImage = pdi::convolve(
            image,
            convolutionKernel,
            CONVOLUTION_OFFSET,
            pdi::BorderMode::Replicate
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "11_convolucao_generica.png",
            convolutionImage
        );

        // Deteccao de bordas: Roberts e Sobel.
        const pdi::GradientResult robertsResult = pdi::roberts(image);
        const pdi::GradientResult sobelResult = pdi::sobel(image);

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "12_roberts_gx.png",
            robertsResult.gx
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "13_roberts_gy.png",
            robertsResult.gy
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "14_roberts_magnitude.png",
            robertsResult.magnitude
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "15_sobel_gx.png",
            sobelResult.gx
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "16_sobel_gy.png",
            sobelResult.gy
        );
        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "17_sobel_magnitude.png",
            sobelResult.magnitude
        );

        // Deteccao de bordas: Kirsch e Laplaciano.
        const pdi::KirschResult kirschResult = pdi::kirsch(image);
        const cv::Mat laplacianImage = pdi::laplacian(image);

        for (int direction = 0; direction < 8; ++direction) {
            const std::string filename =
                std::string(OUTPUT_DIRECTORY) +
                "18_kirsch_mascara_" +
                std::to_string(direction + 1) +
                ".png";

            pdi::saveImage(filename, kirschResult.responses[direction]);
        }

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "26_kirsch_resposta_final.png",
            kirschResult.finalResponse
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "27_laplaciano.png",
            laplacianImage
        );

        // Agucamento: Laplaciano e High Boost.
        const pdi::LaplacianSharpenResult laplacianSharpenResult =
            pdi::laplacianSharpen(image);

        const cv::Mat highBoostImage = pdi::highBoost(
            image,
            HIGH_BOOST_K,
            HIGH_BOOST_NEIGHBORHOOD_SIZE
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "28_laplaciano_resposta_agucamento.png",
            laplacianSharpenResult.laplacianResponse
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "29_agucamento_laplaciano.png",
            laplacianSharpenResult.sharpenedImage
        );

        pdi::saveImage(
            std::string(OUTPUT_DIRECTORY) + "30_high_boost.png",
            highBoostImage
        );

        unsigned long long histogramPixelCount = 0;
        for (const unsigned long long count : originalHistogram) {
            histogramPixelCount += count;
        }

        std::cout << "Transformacoes de intensidade concluidas.\n";
        std::cout << "  01_negativo.png\n";
        std::cout << "  02_limiarizacao.png (T = "
                  << static_cast<int>(THRESHOLD_VALUE) << ")\n";
        std::cout << "  03_gamma.png (gamma = " << GAMMA_VALUE
                  << ", c = " << GAMMA_C << ")\n";
        std::cout << "  04_logaritmo.png (c = " << LOG_C << ")\n";
        std::cout << "  05_alargamento_contraste.png (rMin = "
                  << static_cast<int>(CONTRAST_MIN)
                  << ", rMax = " << static_cast<int>(CONTRAST_MAX) << ")\n\n";

        std::cout << "Operacoes de histograma concluidas.\n";
        std::cout << "  Pixels contabilizados no histograma: "
                  << histogramPixelCount << "\n";
        std::cout << "  06_expansao_histograma.png\n";
        std::cout << "  07_equalizacao_histograma.png\n\n";

        std::cout << "Contraste adaptativo concluido.\n";
        std::cout << "  08_contraste_adaptativo.png (c = "
                  << ADAPTIVE_CONTRAST_C
                  << ", vizinhanca = "
                  << ADAPTIVE_NEIGHBORHOOD_SIZE << "x"
                  << ADAPTIVE_NEIGHBORHOOD_SIZE << ")\n\n";

        std::cout << "Filtragem espacial concluida.\n";
        std::cout << "  09_filtro_media.png (vizinhanca = "
                  << MEAN_NEIGHBORHOOD_SIZE << "x"
                  << MEAN_NEIGHBORHOOD_SIZE << ")\n";
        std::cout << "  10_filtro_mediana.png (vizinhanca = "
                  << MEDIAN_NEIGHBORHOOD_SIZE << "x"
                  << MEDIAN_NEIGHBORHOOD_SIZE << ")\n\n";

        std::cout << "Convolucao generica concluida.\n";
        std::cout << "  11_convolucao_generica.png\n";
        std::cout << "  Mascara: 3x3 de agucamento\n";
        std::cout << "  Offset: " << CONVOLUTION_OFFSET << "\n";
        std::cout << "  Borda: Replicate\n\n";

        std::cout << "Deteccao de bordas concluida.\n";
        std::cout << "  Roberts: Gx, Gy e magnitude -> arquivos 12 a 14\n";
        std::cout << "  Sobel:   Gx, Gy e magnitude -> arquivos 15 a 17\n";
        std::cout << "  Borda utilizada: Replicate\n\n";

        std::cout << "Kirsch e Laplaciano concluidos.\n";
        std::cout << "  Kirsch: respostas das 8 mascaras -> arquivos 18 a 25\n";
        std::cout << "  Kirsch: resposta final por maximo -> arquivo 26\n";
        std::cout << "  Laplaciano 4-vizinhos -> arquivo 27\n";
        std::cout << "  Borda utilizada: Replicate\n\n";

        std::cout << "Agucamento concluido.\n";
        std::cout << "  28_laplaciano_resposta_agucamento.png\n";
        std::cout << "  29_agucamento_laplaciano.png\n";
        std::cout << "  30_high_boost.png (k = " << HIGH_BOOST_K
                  << ", vizinhanca = "
                  << HIGH_BOOST_NEIGHBORHOOD_SIZE << "x"
                  << HIGH_BOOST_NEIGHBORHOOD_SIZE << ")\n\n";

        std::cout << "Resultados salvos em: " << OUTPUT_DIRECTORY << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
