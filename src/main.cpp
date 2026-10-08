#include "image_io.hpp"

#include <exception>
#include <iostream>

namespace {
constexpr const char* INPUT_PATH = "assets/input/lena_color.tif";
constexpr const char* OUTPUT_PATH = "assets/output/copia_entrada.png";
}

int main() {
    try {
        const cv::Mat image = pdi::loadGrayscale8Bit(INPUT_PATH);

        std::cout << "Imagem carregada com sucesso.\n";
        std::cout << "Resolucao: " << image.cols << " x " << image.rows << "\n";
        std::cout << "Tipo esperado confirmado: escala de cinza, 8 bits.\n";

        pdi::saveImage(OUTPUT_PATH, image);

        std::cout << "Copia salva em: " << OUTPUT_PATH << "\n";
        std::cout << "Estrutura base do laboratorio funcionando.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
