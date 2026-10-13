#include "frontend.hpp"

#include "convolution.hpp"
#include "convolution_tests.hpp"
#include "dissolve.hpp"
#include "edge_detection.hpp"
#include "histogram.hpp"
#include "image_io.hpp"
#include "intensity.hpp"
#include "sharpening.hpp"
#include "spatial_filters.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* WINDOW_NAME = "Laboratorio PDI";
constexpr int CANVAS_WIDTH = 1500;
constexpr int CANVAS_HEIGHT = 800;
constexpr int PANEL_WIDTH = 380;
constexpr int BUTTON_HEIGHT = 30;
constexpr int BUTTON_GAP = 5;
constexpr const char* INTERFACE_OUTPUT = "assets/output/interface/";

const cv::Scalar BG(24, 27, 32);
const cv::Scalar PANEL_BG(34, 38, 45);
const cv::Scalar CARD_BG(45, 50, 59);
const cv::Scalar BUTTON_BG(61, 68, 80);
const cv::Scalar BUTTON_ACTIVE(89, 111, 153);
const cv::Scalar TEXT(238, 241, 245);
const cv::Scalar MUTED(170, 178, 190);
const cv::Scalar ACCENT(120, 170, 255);
const cv::Scalar BORDER(78, 85, 98);
const cv::Scalar ERROR_COLOR(100, 140, 255);

enum class Category {
    Intensity,
    Histogram,
    Spatial,
    Edges,
    Sharpening,
    Dissolve,
    Tests
};

enum class Operation {
    None,
    Negative,
    Threshold,
    Gamma,
    Log,
    ContrastStretch,
    HistogramExpansion,
    HistogramEqualization,
    AdaptiveContrast,
    Mean,
    Median,
    GenericConvolution,
    RobertsGx,
    RobertsGy,
    RobertsMagnitude,
    SobelGx,
    SobelGy,
    SobelMagnitude,
    Kirsch1,
    Kirsch2,
    Kirsch3,
    Kirsch4,
    Kirsch5,
    Kirsch6,
    Kirsch7,
    Kirsch8,
    KirschFinal,
    Laplacian,
    LaplacianResponse,
    LaplacianSharpen,
    HighBoost,
    UniformDissolve,
    NonUniformDissolve,
    TestSharpen11,
    TestSharpen21,
    TestSharpen12,
    TestReliefH1,
    TestReliefH2,
    TestEdgeH3,
    TestEdgeH4Clipped,
    TestEdgeH4Absolute
};

enum class ButtonAction {
    CycleImageA,
    CycleImageB,
    SelectCategory,
    RunOperation,
    ResetParameters,
    Exit
};

struct ImageOption {
    std::string label;
    std::string path;
};

struct Button {
    cv::Rect rect;
    std::string label;
    ButtonAction action;
    Category category{Category::Intensity};
    Operation operation{Operation::None};
};

struct AppState {
    std::vector<ImageOption> images{
        {"input.tif", "assets/input/input.tif"},
        {"barbara.tif", "assets/input/barbara.tif"},
        {"textura.jpg", "assets/input/textura.jpg"},
        {"nequinha.jpeg", "assets/input/nequinha.jpeg"}
    };

    int imageAIndex{0};
    int imageBIndex{1};
    Category category{Category::Intensity};
    Operation operation{Operation::None};

    cv::Mat imageA;
    cv::Mat imageB;
    cv::Mat result;

    std::string resultTitle{"Selecione uma operacao"};
    std::string status{"Interface pronta. Ajuste os sliders e clique em uma operacao."};
    bool statusIsError{false};
    bool running{true};

    std::vector<Button> buttons;
};

std::string categoryLabel(Category category) {
    switch (category) {
        case Category::Intensity: return "Intensidade";
        case Category::Histogram: return "Histograma";
        case Category::Spatial: return "Filtros";
        case Category::Edges: return "Bordas";
        case Category::Sharpening: return "Agucamento";
        case Category::Dissolve: return "Dissolve";
        case Category::Tests: return "Testes AV1";
    }
    return "Categoria";
}

std::string operationLabel(Operation op) {
    switch (op) {
        case Operation::Negative: return "Negativo";
        case Operation::Threshold: return "Limiarizacao";
        case Operation::Gamma: return "Gamma";
        case Operation::Log: return "Logaritmo";
        case Operation::ContrastStretch: return "Alargamento contraste";
        case Operation::HistogramExpansion: return "Expansao histograma";
        case Operation::HistogramEqualization: return "Equalizacao histograma";
        case Operation::AdaptiveContrast: return "Contraste adaptativo";
        case Operation::Mean: return "Filtro da media";
        case Operation::Median: return "Filtro da mediana";
        case Operation::GenericConvolution: return "Convolucao 3x3";
        case Operation::RobertsGx: return "Roberts |Gx|";
        case Operation::RobertsGy: return "Roberts |Gy|";
        case Operation::RobertsMagnitude: return "Roberts magnitude";
        case Operation::SobelGx: return "Sobel |Gx|";
        case Operation::SobelGy: return "Sobel |Gy|";
        case Operation::SobelMagnitude: return "Sobel magnitude";
        case Operation::Kirsch1: return "Kirsch mascara 1";
        case Operation::Kirsch2: return "Kirsch mascara 2";
        case Operation::Kirsch3: return "Kirsch mascara 3";
        case Operation::Kirsch4: return "Kirsch mascara 4";
        case Operation::Kirsch5: return "Kirsch mascara 5";
        case Operation::Kirsch6: return "Kirsch mascara 6";
        case Operation::Kirsch7: return "Kirsch mascara 7";
        case Operation::Kirsch8: return "Kirsch mascara 8";
        case Operation::KirschFinal: return "Kirsch resposta final";
        case Operation::Laplacian: return "Laplaciano";
        case Operation::LaplacianResponse: return "Resposta Laplaciana";
        case Operation::LaplacianSharpen: return "Agucamento Laplaciano";
        case Operation::HighBoost: return "High Boost";
        case Operation::UniformDissolve: return "Dissolve uniforme";
        case Operation::NonUniformDissolve: return "Dissolve nao uniforme";
        case Operation::TestSharpen11: return "Teste c=1 d=1";
        case Operation::TestSharpen21: return "Teste c=2 d=1";
        case Operation::TestSharpen12: return "Teste c=1 d=2";
        case Operation::TestReliefH1: return "Teste relevo h1";
        case Operation::TestReliefH2: return "Teste relevo h2";
        case Operation::TestEdgeH3: return "Teste bordas h3";
        case Operation::TestEdgeH4Clipped: return "Teste h4 saturado";
        case Operation::TestEdgeH4Absolute: return "Teste h4 |resposta|";
        case Operation::None: return "Nenhuma";
    }
    return "Operacao";
}

std::string operationStem(Operation op) {
    switch (op) {
        case Operation::Negative: return "negativo";
        case Operation::Threshold: return "limiarizacao";
        case Operation::Gamma: return "gamma";
        case Operation::Log: return "logaritmo";
        case Operation::ContrastStretch: return "alargamento_contraste";
        case Operation::HistogramExpansion: return "expansao_histograma";
        case Operation::HistogramEqualization: return "equalizacao_histograma";
        case Operation::AdaptiveContrast: return "contraste_adaptativo";
        case Operation::Mean: return "media";
        case Operation::Median: return "mediana";
        case Operation::GenericConvolution: return "convolucao_generica";
        case Operation::RobertsGx: return "roberts_gx";
        case Operation::RobertsGy: return "roberts_gy";
        case Operation::RobertsMagnitude: return "roberts_magnitude";
        case Operation::SobelGx: return "sobel_gx";
        case Operation::SobelGy: return "sobel_gy";
        case Operation::SobelMagnitude: return "sobel_magnitude";
        case Operation::Kirsch1: return "kirsch_1";
        case Operation::Kirsch2: return "kirsch_2";
        case Operation::Kirsch3: return "kirsch_3";
        case Operation::Kirsch4: return "kirsch_4";
        case Operation::Kirsch5: return "kirsch_5";
        case Operation::Kirsch6: return "kirsch_6";
        case Operation::Kirsch7: return "kirsch_7";
        case Operation::Kirsch8: return "kirsch_8";
        case Operation::KirschFinal: return "kirsch_final";
        case Operation::Laplacian: return "laplaciano";
        case Operation::LaplacianResponse: return "laplaciano_resposta";
        case Operation::LaplacianSharpen: return "agucamento_laplaciano";
        case Operation::HighBoost: return "high_boost";
        case Operation::UniformDissolve: return "dissolve_uniforme";
        case Operation::NonUniformDissolve: return "dissolve_nao_uniforme";
        case Operation::TestSharpen11: return "teste_c1_d1";
        case Operation::TestSharpen21: return "teste_c2_d1";
        case Operation::TestSharpen12: return "teste_c1_d2";
        case Operation::TestReliefH1: return "teste_relevo_h1";
        case Operation::TestReliefH2: return "teste_relevo_h2";
        case Operation::TestEdgeH3: return "teste_bordas_h3";
        case Operation::TestEdgeH4Clipped: return "teste_bordas_h4_saturado";
        case Operation::TestEdgeH4Absolute: return "teste_bordas_h4_modulo";
        case Operation::None: return "resultado";
    }
    return "resultado";
}

std::vector<Operation> operationsFor(Category category) {
    switch (category) {
        case Category::Intensity:
            return {
                Operation::Negative,
                Operation::Threshold,
                Operation::Gamma,
                Operation::Log,
                Operation::ContrastStretch
            };
        case Category::Histogram:
            return {
                Operation::HistogramExpansion,
                Operation::HistogramEqualization,
                Operation::AdaptiveContrast
            };
        case Category::Spatial:
            return {
                Operation::Mean,
                Operation::Median,
                Operation::GenericConvolution
            };
        case Category::Edges:
            return {
                Operation::RobertsGx,
                Operation::RobertsGy,
                Operation::RobertsMagnitude,
                Operation::SobelGx,
                Operation::SobelGy,
                Operation::SobelMagnitude,
                Operation::Kirsch1,
                Operation::Kirsch2,
                Operation::Kirsch3,
                Operation::Kirsch4,
                Operation::Kirsch5,
                Operation::Kirsch6,
                Operation::Kirsch7,
                Operation::Kirsch8,
                Operation::KirschFinal,
                Operation::Laplacian
            };
        case Category::Sharpening:
            return {
                Operation::LaplacianResponse,
                Operation::LaplacianSharpen,
                Operation::HighBoost
            };
        case Category::Dissolve:
            return {
                Operation::UniformDissolve,
                Operation::NonUniformDissolve
            };
        case Category::Tests:
            return {
                Operation::TestSharpen11,
                Operation::TestSharpen21,
                Operation::TestSharpen12,
                Operation::TestReliefH1,
                Operation::TestReliefH2,
                Operation::TestEdgeH3,
                Operation::TestEdgeH4Clipped,
                Operation::TestEdgeH4Absolute
            };
    }
    return {};
}

void drawText(
    cv::Mat& canvas,
    const std::string& text,
    cv::Point origin,
    double scale = 0.5,
    cv::Scalar color = TEXT,
    int thickness = 1
) {
    cv::putText(
        canvas,
        text,
        origin,
        cv::FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv::LINE_AA
    );
}

void drawWrappedText(
    cv::Mat& canvas,
    const std::string& text,
    cv::Point origin,
    int maxChars,
    cv::Scalar color
) {
    std::istringstream words(text);
    std::string word;
    std::string line;
    int y = origin.y;

    while (words >> word) {
        if (!line.empty() && static_cast<int>(line.size() + word.size() + 1) > maxChars) {
            drawText(canvas, line, cv::Point(origin.x, y), 0.48, color);
            y += 22;
            line.clear();
        }
        if (!line.empty()) {
            line += " ";
        }
        line += word;
    }

    if (!line.empty()) {
        drawText(canvas, line, cv::Point(origin.x, y), 0.48, color);
    }
}

void addButton(
    AppState& state,
    cv::Mat& canvas,
    const cv::Rect& rect,
    const std::string& label,
    ButtonAction action,
    bool active = false,
    Category category = Category::Intensity,
    Operation operation = Operation::None
) {
    cv::rectangle(canvas, rect, active ? BUTTON_ACTIVE : BUTTON_BG, cv::FILLED);
    cv::rectangle(canvas, rect, BORDER, 1);

    int baseline = 0;
    const cv::Size size = cv::getTextSize(
        label,
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
        &baseline
    );

    const int x = rect.x + std::max(8, (rect.width - size.width) / 2);
    const int y = rect.y + (rect.height + size.height) / 2 - 2;
    drawText(canvas, label, cv::Point(x, y), 0.45, TEXT);

    state.buttons.push_back({rect, label, action, category, operation});
}

cv::Mat previewImage(const cv::Mat& gray, const cv::Size& box) {
    if (gray.empty()) {
        return cv::Mat(box, CV_8UC3, CARD_BG);
    }

    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

    const double scale = std::min(
        static_cast<double>(box.width) / static_cast<double>(bgr.cols),
        static_cast<double>(box.height) / static_cast<double>(bgr.rows)
    );

    const int width = std::max(1, static_cast<int>(bgr.cols * scale));
    const int height = std::max(1, static_cast<int>(bgr.rows * scale));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);

    cv::Mat result(box, CV_8UC3, CARD_BG);
    const int x = (box.width - width) / 2;
    const int y = (box.height - height) / 2;
    resized.copyTo(result(cv::Rect(x, y, width, height)));
    return result;
}

std::string selectedStem(const AppState& state) {
    const std::filesystem::path path(state.images[state.imageAIndex].path);
    return path.stem().string();
}

int neighborhoodFromTrackbar() {
    static const std::array<int, 5> values{1, 3, 5, 7, 9};
    int index = cv::getTrackbarPos("Vizinhanca", WINDOW_NAME);
    index = std::clamp(index, 0, static_cast<int>(values.size()) - 1);
    return values[static_cast<std::size_t>(index)];
}

void reloadImages(AppState& state) {
    state.imageA = pdi::loadGrayscale8Bit(state.images[state.imageAIndex].path);
    state.imageB = pdi::loadGrayscale8Bit(state.images[state.imageBIndex].path);
    state.result.release();
    state.resultTitle = "Selecione uma operacao";
    state.operation = Operation::None;
    state.status = "Imagem selecionada: " + state.images[state.imageAIndex].label;
    state.statusIsError = false;
}

cv::Mat createDefaultConvolutionKernel() {
    cv::Mat kernel(3, 3, CV_64FC1);
    const double values[3][3] = {
        {-1.0, -1.0, -1.0},
        {-1.0,  9.0, -1.0},
        {-1.0, -1.0, -1.0}
    };

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            kernel.ptr<double>(row)[col] = values[row][col];
        }
    }
    return kernel;
}

cv::Mat horizontalWeightMap(const cv::Mat& image) {
    cv::Mat weights(image.rows, image.cols, CV_64FC1);

    for (int row = 0; row < image.rows; ++row) {
        double* output = weights.ptr<double>(row);
        for (int col = 0; col < image.cols; ++col) {
            output[col] = image.cols <= 1
                ? 0.0
                : static_cast<double>(col) / static_cast<double>(image.cols - 1);
        }
    }
    return weights;
}

pdi::MandatoryConvolutionTest testFromOperation(Operation op) {
    switch (op) {
        case Operation::TestSharpen11:
            return pdi::MandatoryConvolutionTest::SharpenC1D1;
        case Operation::TestSharpen21:
            return pdi::MandatoryConvolutionTest::SharpenC2D1;
        case Operation::TestSharpen12:
            return pdi::MandatoryConvolutionTest::SharpenC1D2;
        case Operation::TestReliefH1:
            return pdi::MandatoryConvolutionTest::ReliefH1;
        case Operation::TestReliefH2:
            return pdi::MandatoryConvolutionTest::ReliefH2;
        case Operation::TestEdgeH3:
            return pdi::MandatoryConvolutionTest::EdgeH3;
        case Operation::TestEdgeH4Clipped:
            return pdi::MandatoryConvolutionTest::EdgeH4Clipped;
        case Operation::TestEdgeH4Absolute:
            return pdi::MandatoryConvolutionTest::EdgeH4Absolute;
        default:
            throw std::invalid_argument("Operacao nao corresponde a teste obrigatorio.");
    }
}

cv::Mat executeOperation(AppState& state, Operation op, std::string& details) {
    const int thresholdValue = cv::getTrackbarPos("Limiar", WINDOW_NAME);
    const double gamma = std::max(1, cv::getTrackbarPos("Gamma x100", WINDOW_NAME)) / 100.0;
    const int contrastMin = cv::getTrackbarPos("Contraste min", WINDOW_NAME);
    const int contrastMax = cv::getTrackbarPos("Contraste max", WINDOW_NAME);
    const double adaptiveC = cv::getTrackbarPos("Contraste C x10", WINDOW_NAME) / 10.0;
    const int neighborhood = neighborhoodFromTrackbar();
    const double highBoostK = cv::getTrackbarPos("HighBoost K x10", WINDOW_NAME) / 10.0;
    const double alpha = cv::getTrackbarPos("Dissolve alpha %", WINDOW_NAME) / 100.0;

    switch (op) {
        case Operation::Negative:
            details = "s = 255 - r";
            return pdi::negative(state.imageA);
        case Operation::Threshold:
            details = "T = " + std::to_string(thresholdValue);
            return pdi::threshold(state.imageA, static_cast<unsigned char>(thresholdValue));
        case Operation::Gamma:
            details = "gamma = " + std::to_string(gamma) + ", c = 1";
            return pdi::gammaTransform(state.imageA, gamma, 1.0);
        case Operation::Log:
            details = "Transformacao logaritmica, c = 1";
            return pdi::logTransform(state.imageA, 1.0);
        case Operation::ContrastStretch:
            if (contrastMin >= contrastMax) {
                throw std::invalid_argument("Contraste min deve ser menor que contraste max.");
            }
            details = "rMin = " + std::to_string(contrastMin) +
                      ", rMax = " + std::to_string(contrastMax);
            return pdi::contrastStretch(
                state.imageA,
                static_cast<unsigned char>(contrastMin),
                static_cast<unsigned char>(contrastMax)
            );
        case Operation::HistogramExpansion:
            details = "Expansao automatica usando min/max do histograma";
            return pdi::histogramExpansion(state.imageA);
        case Operation::HistogramEqualization:
            details = "Equalizacao manual pela CDF";
            return pdi::histogramEqualization(state.imageA);
        case Operation::AdaptiveContrast:
            details = "c = " + std::to_string(adaptiveC) +
                      ", vizinhanca = " + std::to_string(neighborhood) + "x" +
                      std::to_string(neighborhood);
            return pdi::adaptiveContrast(state.imageA, adaptiveC, neighborhood);
        case Operation::Mean:
            details = "Vizinhanca = " + std::to_string(neighborhood) + "x" +
                      std::to_string(neighborhood);
            return pdi::meanFilter(state.imageA, neighborhood);
        case Operation::Median:
            details = "Vizinhanca = " + std::to_string(neighborhood) + "x" +
                      std::to_string(neighborhood);
            return pdi::medianFilter(state.imageA, neighborhood);
        case Operation::GenericConvolution:
            details = "Mascara [-1 -1 -1; -1 9 -1; -1 -1 -1], borda Replicate";
            return pdi::convolve(
                state.imageA,
                createDefaultConvolutionKernel(),
                0.0,
                pdi::BorderMode::Replicate
            );
        case Operation::RobertsGx:
        case Operation::RobertsGy:
        case Operation::RobertsMagnitude: {
            const pdi::GradientResult result = pdi::roberts(state.imageA);
            details = "Roberts 2x2, borda Replicate";
            if (op == Operation::RobertsGx) return result.gx;
            if (op == Operation::RobertsGy) return result.gy;
            return result.magnitude;
        }
        case Operation::SobelGx:
        case Operation::SobelGy:
        case Operation::SobelMagnitude: {
            const pdi::GradientResult result = pdi::sobel(state.imageA);
            details = "Sobel 3x3, borda Replicate";
            if (op == Operation::SobelGx) return result.gx;
            if (op == Operation::SobelGy) return result.gy;
            return result.magnitude;
        }
        case Operation::Kirsch1:
        case Operation::Kirsch2:
        case Operation::Kirsch3:
        case Operation::Kirsch4:
        case Operation::Kirsch5:
        case Operation::Kirsch6:
        case Operation::Kirsch7:
        case Operation::Kirsch8:
        case Operation::KirschFinal: {
            const pdi::KirschResult result = pdi::kirsch(state.imageA);
            details = "Kirsch: 8 mascaras direcionais, resposta final por maximo";
            if (op == Operation::KirschFinal) {
                return result.finalResponse;
            }
            const int index = static_cast<int>(op) - static_cast<int>(Operation::Kirsch1);
            return result.responses[index];
        }
        case Operation::Laplacian:
            details = "Laplaciano 4-vizinhos, modulo para visualizacao";
            return pdi::laplacian(state.imageA);
        case Operation::LaplacianResponse:
        case Operation::LaplacianSharpen: {
            const pdi::LaplacianSharpenResult result = pdi::laplacianSharpen(state.imageA);
            details = "Mascara Laplaciana com centro positivo";
            return op == Operation::LaplacianResponse
                ? result.laplacianResponse
                : result.sharpenedImage;
        }
        case Operation::HighBoost:
            details = "k = " + std::to_string(highBoostK) +
                      ", vizinhanca = " + std::to_string(neighborhood) + "x" +
                      std::to_string(neighborhood);
            return pdi::highBoost(state.imageA, highBoostK, neighborhood);
        case Operation::UniformDissolve:
            details = "alpha = " + std::to_string(alpha) +
                      "; A=" + state.images[state.imageAIndex].label +
                      ", B=" + state.images[state.imageBIndex].label;
            return pdi::uniformDissolve(state.imageA, state.imageB, alpha);
        case Operation::NonUniformDissolve:
            details = "Peso horizontal 0..1; A=" + state.images[state.imageAIndex].label +
                      ", B=" + state.images[state.imageBIndex].label;
            return pdi::nonUniformDissolve(
                state.imageA,
                state.imageB,
                horizontalWeightMap(state.imageA)
            );
        case Operation::TestSharpen11:
        case Operation::TestSharpen21:
        case Operation::TestSharpen12:
        case Operation::TestReliefH1:
        case Operation::TestReliefH2:
        case Operation::TestEdgeH3:
        case Operation::TestEdgeH4Clipped:
        case Operation::TestEdgeH4Absolute: {
            const pdi::MandatoryConvolutionTest test = testFromOperation(op);
            details = pdi::mandatoryConvolutionTestName(test);
            if (test == pdi::MandatoryConvolutionTest::EdgeH4Clipped) {
                details += ": valores negativos saturam em 0; resultado pode ficar preto";
            }
            if (test == pdi::MandatoryConvolutionTest::EdgeH4Absolute) {
                details += ": visualizacao auxiliar de |resposta bruta|";
            }
            return pdi::runMandatoryConvolutionTest(state.imageA, test);
        }
        case Operation::None:
            return state.imageA.clone();
    }

    throw std::runtime_error("Operacao nao implementada na interface.");
}

void saveCurrentResult(AppState& state) {
    if (state.result.empty() || state.operation == Operation::None) {
        return;
    }

    const std::string path =
        std::string(INTERFACE_OUTPUT) +
        operationStem(state.operation) + "_" +
        selectedStem(state) + ".png";

    pdi::saveImage(path, state.result);
    state.status = "Executado e salvo em: " + path;
    state.statusIsError = false;
}

void runOperation(AppState& state, Operation operation) {
    try {
        std::string details;
        state.result = executeOperation(state, operation, details);
        state.operation = operation;
        state.resultTitle = operationLabel(operation);
        saveCurrentResult(state);
        if (!details.empty()) {
            state.status += " | " + details;
        }
    } catch (const std::exception& error) {
        state.status = std::string("Erro: ") + error.what();
        state.statusIsError = true;
    }
}

struct TrackbarParameter {
    const char* name;
    int maximum;
    int defaultValue;
};

constexpr std::array<TrackbarParameter, 8> TRACKBAR_PARAMETERS{{
    {"Limiar", 255, 128},
    {"Gamma x100", 300, 50},
    {"Contraste min", 255, 60},
    {"Contraste max", 255, 190},
    {"Contraste C x10", 50, 15},
    {"Vizinhanca", 4, 1}, // 0..4 corresponde a 1, 3, 5, 7, 9.
    {"HighBoost K x10", 50, 15},
    {"Dissolve alpha %", 100, 50}
}};

void resetParameters() {
    for (const auto& parameter : TRACKBAR_PARAMETERS) {
        cv::setTrackbarPos(parameter.name, WINDOW_NAME, parameter.defaultValue);
    }
}

void render(AppState& state) {
    cv::Mat canvas(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC3, BG);
    state.buttons.clear();

    cv::rectangle(canvas, cv::Rect(0, 0, PANEL_WIDTH, CANVAS_HEIGHT), PANEL_BG, cv::FILLED);

    drawText(canvas, "LABORATORIO PDI", cv::Point(20, 30), 0.7, TEXT, 2);
    drawText(canvas, "Processamento de Imagens - AV1", cv::Point(20, 53), 0.43, MUTED);

    addButton(
        state,
        canvas,
        cv::Rect(20, 70, 340, 30),
        "Imagem A: " + state.images[state.imageAIndex].label + "  >",
        ButtonAction::CycleImageA
    );
    addButton(
        state,
        canvas,
        cv::Rect(20, 105, 340, 30),
        "Imagem B: " + state.images[state.imageBIndex].label + "  >",
        ButtonAction::CycleImageB
    );

    const std::array<Category, 7> categories{
        Category::Intensity,
        Category::Histogram,
        Category::Spatial,
        Category::Edges,
        Category::Sharpening,
        Category::Dissolve,
        Category::Tests
    };

    const int categoryY = 150;
    const int categoryWidth = 165;
    for (int index = 0; index < static_cast<int>(categories.size()); ++index) {
        const int col = index % 2;
        const int row = index / 2;
        const cv::Rect rect(
            20 + col * (categoryWidth + 10),
            categoryY + row * 35,
            categoryWidth,
            30
        );
        addButton(
            state,
            canvas,
            rect,
            categoryLabel(categories[index]),
            ButtonAction::SelectCategory,
            categories[index] == state.category,
            categories[index]
        );
    }

    drawText(
        canvas,
        "Operacoes - " + categoryLabel(state.category),
        cv::Point(20, 300),
        0.5,
        ACCENT,
        1
    );

    const std::vector<Operation> operations = operationsFor(state.category);
    const bool twoColumns = operations.size() > 10;
    const int operationWidth = twoColumns ? 165 : 340;
    for (int index = 0; index < static_cast<int>(operations.size()); ++index) {
        const int col = twoColumns ? index % 2 : 0;
        const int row = twoColumns ? index / 2 : index;
        const int x = 20 + col * (operationWidth + 10);
        const int y = 315 + row * (BUTTON_HEIGHT + BUTTON_GAP);

        addButton(
            state,
            canvas,
            cv::Rect(x, y, operationWidth, BUTTON_HEIGHT),
            operationLabel(operations[index]),
            ButtonAction::RunOperation,
            operations[index] == state.operation,
            state.category,
            operations[index]
        );
    }

    addButton(
        state,
        canvas,
        cv::Rect(20, 755, 340, 28),
        "Sair (Esc)",
        ButtonAction::Exit
    );

    const cv::Rect originalRect(410, 105, 500, 480);
    const cv::Rect resultRect(950, 105, 500, 480);

    cv::rectangle(canvas, originalRect, CARD_BG, cv::FILLED);
    cv::rectangle(canvas, originalRect, BORDER, 1);
    cv::rectangle(canvas, resultRect, CARD_BG, cv::FILLED);
    cv::rectangle(canvas, resultRect, BORDER, 1);

    drawText(
        canvas,
        "Original - " + state.images[state.imageAIndex].label,
        cv::Point(originalRect.x, 85),
        0.55,
        TEXT,
        1
    );
    drawText(
        canvas,
        "Resultado - " + state.resultTitle,
        cv::Point(resultRect.x, 85),
        0.55,
        TEXT,
        1
    );

    const cv::Mat originalPreview = previewImage(
        state.imageA,
        cv::Size(originalRect.width - 20, originalRect.height - 20)
    );
    originalPreview.copyTo(
        canvas(cv::Rect(
            originalRect.x + 10,
            originalRect.y + 10,
            originalPreview.cols,
            originalPreview.rows
        ))
    );

    const cv::Mat shownResult = state.result.empty() ? state.imageA : state.result;
    const cv::Mat resultPreview = previewImage(
        shownResult,
        cv::Size(resultRect.width - 20, resultRect.height - 20)
    );
    resultPreview.copyTo(
        canvas(cv::Rect(
            resultRect.x + 10,
            resultRect.y + 10,
            resultPreview.cols,
            resultPreview.rows
        ))
    );

    addButton(
        state,
        canvas,
        cv::Rect(1120, 605, 330, 30),
        "Restaurar padroes",
        ButtonAction::ResetParameters
    );

    drawText(canvas, "Parametros (sliders no topo da janela)", cv::Point(410, 625), 0.55, ACCENT, 1);
    drawText(
        canvas,
        "Vizinhanca usa: 1, 3, 5, 7 ou 9. Resultados sao salvos automaticamente.",
        cv::Point(410, 650),
        0.48,
        MUTED
    );

    drawText(canvas, "Status:", cv::Point(410, 690), 0.52, TEXT, 1);
    drawWrappedText(
        canvas,
        state.status,
        cv::Point(410, 715),
        105,
        state.statusIsError ? ERROR_COLOR : MUTED
    );

    cv::imshow(WINDOW_NAME, canvas);
}

void handleMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) {
        return;
    }

    AppState& state = *static_cast<AppState*>(userdata);

    for (const Button& button : state.buttons) {
        if (!button.rect.contains(cv::Point(x, y))) {
            continue;
        }

        try {
            switch (button.action) {
                case ButtonAction::CycleImageA:
                    state.imageAIndex = (state.imageAIndex + 1) % static_cast<int>(state.images.size());
                    reloadImages(state);
                    break;
                case ButtonAction::CycleImageB:
                    state.imageBIndex = (state.imageBIndex + 1) % static_cast<int>(state.images.size());
                    state.imageB = pdi::loadGrayscale8Bit(state.images[state.imageBIndex].path);
                    state.status = "Imagem B: " + state.images[state.imageBIndex].label;
                    state.statusIsError = false;
                    break;
                case ButtonAction::SelectCategory:
                    state.category = button.category;
                    state.operation = Operation::None;
                    break;
                case ButtonAction::RunOperation:
                    runOperation(state, button.operation);
                    break;
                case ButtonAction::ResetParameters:
                    resetParameters();
                    state.status = "Parametros restaurados. Clique em uma operacao para aplicar.";
                    state.statusIsError = false;
                    break;
                case ButtonAction::Exit:
                    state.running = false;
                    break;
            }
        } catch (const std::exception& error) {
            state.status = std::string("Erro: ") + error.what();
            state.statusIsError = true;
        }

        render(state);
        return;
    }
}

void createTrackbars() {
    for (const auto& parameter : TRACKBAR_PARAMETERS) {
        cv::createTrackbar(parameter.name, WINDOW_NAME, nullptr, parameter.maximum);
    }
    resetParameters();
}

} // namespace

namespace pdi {

int runFrontend() {
    AppState state;

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    createTrackbars();
    cv::setMouseCallback(WINDOW_NAME, handleMouse, &state);

    try {
        reloadImages(state);
    } catch (const std::exception& error) {
        state.status = std::string("Erro ao carregar imagens: ") + error.what();
        state.statusIsError = true;
    }

    render(state);

    while (state.running) {
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') {
            state.running = false;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace pdi
