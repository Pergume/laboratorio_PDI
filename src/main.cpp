#include "image_io.hpp"
#include "intensity.hpp"

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
}

int main() {
    try {
        const cv::Mat image = pdi::loadGrayscale8Bit(INPUT_PATH);

        std::cout << "Imagem carregada com sucesso.\n";
        std::cout << "Resolucao: " << image.cols << " x " << image.rows << "\n";
        std::cout << "Formato: escala de cinza, 8 bits.\n\n";

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

        std::cout << "Transformacoes de intensidade concluidas.\n";
        std::cout << "Resultados salvos em: " << OUTPUT_DIRECTORY << "\n";
        std::cout << "  01_negativo.png\n";
        std::cout << "  02_limiarizacao.png (T = "
                  << static_cast<int>(THRESHOLD_VALUE) << ")\n";
        std::cout << "  03_gamma.png (gamma = " << GAMMA_VALUE
                  << ", c = " << GAMMA_C << ")\n";
        std::cout << "  04_logaritmo.png (c = " << LOG_C << ")\n";
        std::cout << "  05_alargamento_contraste.png (rMin = "
                  << static_cast<int>(CONTRAST_MIN)
                  << ", rMax = " << static_cast<int>(CONTRAST_MAX) << ")\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}