#include "histogram.hpp"
#include "image_io.hpp"
#include "intensity.hpp"
#include "spatial_filters.hpp"

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

        std::cout << "Resultados salvos em: " << OUTPUT_DIRECTORY << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
