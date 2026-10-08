#include "image_io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <stdexcept>

namespace pdi {

bool isGrayscale8Bit(const cv::Mat& image) noexcept {
    return !image.empty() && image.type() == CV_8UC1;
}

cv::Mat loadGrayscale8Bit(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        throw std::runtime_error("Nao foi possivel carregar a imagem: " + path);
    }

    if (!isGrayscale8Bit(image)) {
        throw std::runtime_error(
            "A imagem carregada nao esta no formato esperado: escala de cinza, 8 bits."
        );
    }

    return image;
}

void saveImage(const std::string& path, const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Nao e possivel salvar uma imagem vazia.");
    }

    const std::filesystem::path outputPath(path);
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Nao foi possivel salvar a imagem: " + path);
    }
}

} // namespace pdi
