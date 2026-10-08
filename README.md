# Laboratório de Processamento de Imagens

Projeto-base em C++ para a AV1 de Processamento de Imagens.

## Objetivo desta versão

Esta primeira versão implementa somente a infraestrutura do laboratório:

- organização dos módulos;
- leitura de imagem com OpenCV;
- conversão/carregamento em escala de cinza;
- validação de imagem de 8 bits por pixel;
- criação automática da pasta de saída;
- gravação de uma cópia da imagem de entrada.

Os algoritmos avaliados ainda não estão implementados. Seus módulos e assinaturas já foram separados para que possamos desenvolvê-los e testá-los um por um.

## Regra importante do projeto

O OpenCV será utilizado para entrada, saída e representação da imagem (`cv::Mat`). As operações avaliadas na disciplina serão implementadas manualmente, sem usar funções prontas como `cv::Sobel`, `cv::Laplacian`, `cv::medianBlur`, `cv::filter2D` ou `cv::equalizeHist`.

## Estrutura

```text
LaboratorioPDI/
├── CMakeLists.txt
├── README.md
├── assets/
│   ├── input/
│   │   └── input.png
│   └── output/
├── include/
│   ├── image_io.hpp
│   ├── intensity.hpp
│   ├── histogram.hpp
│   ├── convolution.hpp
│   ├── spatial_filters.hpp
│   ├── edge_detection.hpp
│   ├── sharpening.hpp
│   └── dissolve.hpp
└── src/
    ├── main.cpp
    ├── image_io.cpp
    ├── intensity.cpp
    ├── histogram.cpp
    ├── convolution.cpp
    ├── spatial_filters.cpp
    ├── edge_detection.cpp
    ├── sharpening.cpp
    └── dissolve.cpp
```

## Módulos planejados

### image_io
Entrada, saída e validação do formato da imagem.

### intensity
Transformações pontuais:

- negativo;
- limiarização;
- gamma;
- logaritmo;
- alargamento de contraste.

### histogram

- cálculo do histograma;
- expansão;
- equalização;
- contraste adaptativo.

### convolution
Convolução genérica com máscara `n x m`, offset e estratégia explícita de borda.

### spatial_filters

- média;
- mediana.

### edge_detection

- Roberts;
- Sobel;
- Kirsch;
- Laplaciano;
- resultados intermediários como `Gx`, `Gy` e magnitude.

### sharpening

- aguçamento Laplaciano;
- high boost.

### dissolve

- dissolve uniforme;
- dissolve não uniforme.

## Teste da estrutura base

1. Coloque uma imagem em:

```text
assets/input/input.png
```

2. Configure o projeto:

```bash
cmake -S . -B build
```

3. Compile:

```bash
cmake --build build
```

4. Execute a partir da raiz do projeto:

```bash
./build/laboratorio_pdi
```

5. A saída inicial será criada em:

```text
assets/output/copia_entrada.png
```

Nesta fase a imagem de saída será igual à imagem de entrada, pois ainda não há nenhum algoritmo de processamento aplicado.
