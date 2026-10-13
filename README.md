# Laboratório de Processamento de Imagens

Projeto em C++ para a AV1 de Processamento de Imagens, com algoritmos implementados manualmente e interface gráfica para selecionar imagens, ajustar parâmetros e comparar os resultados.

O OpenCV é utilizado para representar imagens (`cv::Mat`), ler e gravar arquivos e construir a interface, incluindo janelas, sliders e previews. As operações avaliadas são implementadas nos módulos do projeto, sem funções prontas como `cv::Sobel`, `cv::Laplacian`, `cv::medianBlur`, `cv::filter2D` ou `cv::equalizeHist`.

## Requisitos e execução

- Compilador com suporte a C++17.
- CMake 3.16 ou superior.
- OpenCV 4 com os componentes `core`, `imgcodecs`, `highgui` e `imgproc`.
- Ambiente gráfico para abrir as janelas da interface.

Configure, compile e execute **a partir da raiz do projeto**:

```bash
cmake -S . -B build
cmake --build build
./build/laboratorio_PDI
```

Os caminhos das imagens e das saídas são relativos à raiz. O nome do executável diferencia maiúsculas e minúsculas: `laboratorio_PDI`.

## Uso da interface

1. Use o botão **Imagem A** para alternar a imagem de entrada.
2. Selecione uma categoria e uma operação.
3. Ajuste os sliders na janela **Parametros - Laboratorio PDI** e clique novamente na operação para aplicar os novos valores.
4. Compare a entrada com o resultado na janela principal. Cada execução salva o resultado automaticamente.

O botão **Imagem B** seleciona a segunda entrada para o dissolve. As duas imagens precisam ter as mesmas dimensões; imagens incompatíveis geram uma mensagem de erro. **Restaurar padroes** repõe os parâmetros iniciais, que serão aplicados quando uma operação for executada novamente. Para fechar, use **Sair**, `Esc` ou `Q`.

As imagens disponíveis nos seletores são:

- `assets/input/input.tif`
- `assets/input/barbara.tif`
- `assets/input/textura.jpg`
- `assets/input/nequinha.jpeg`
- `assets/input/mell.jpeg`

O carregamento converte as entradas para escala de cinza de 8 bits. A lista de imagens da interface é definida em `src/frontend.cpp`.

## Operações implementadas

| Categoria | Operações disponíveis na interface |
| --- | --- |
| Intensidade | Negativo, limiarização, gamma, logaritmo e alargamento de contraste |
| Histograma | Expansão, equalização pela CDF e contraste adaptativo |
| Filtros | Média, mediana e convolução 3×3 com máscara de aguçamento |
| Bordas | Roberts e Sobel com componentes `\|Gx\|`, `\|Gy\|` e magnitude; oito máscaras Kirsch e resposta final; Laplaciano |
| Aguçamento | Resposta Laplaciana, aguçamento Laplaciano e High Boost |
| Dissolve | Uniforme com parâmetro alpha e não uniforme com mapa de pesos horizontal de 0 a 1 |
| Testes AV1 | Aguçamento com `c=1,d=1`, `c=2,d=1` e `c=1,d=2`; relevo h1 e h2; bordas h3; h4 saturado e módulo da resposta h4 |

O módulo de convolução também oferece uma função genérica com máscara, offset e tratamento explícito de bordas. A interface usa a máscara `[-1 -1 -1; -1 9 -1; -1 -1 -1]`, com replicação nas bordas.

No teste **h4 saturado**, respostas negativas são limitadas a zero e podem produzir uma imagem preta. A opção **h4 |resposta|** permite visualizar o módulo da resposta bruta.

### Mediana sobre imagem com ruído

Ao executar a mediana, a interface adiciona ruído sal e pimenta à imagem A: cada pixel tem probabilidade de 5% de ser substituído por preto e 5% por branco. O painel esquerdo mostra essa imagem com ruído, e o direito mostra a mediana aplicada à mesma entrada.

A amostra de ruído permanece a mesma ao repetir a operação com diferentes tamanhos de vizinhança. A imagem original é preservada para as demais operações.

### Contraste adaptativo com estatísticas locais

A interface mostra quatro painéis: imagem original, resultado, mapa de média local e mapa de desvio padrão local. As estatísticas são calculadas sobre a vizinhança selecionada, recortada nos limites da imagem, usando a variância populacional `E[r²] - E[r]²`.

O realce utiliza:

```text
s = m + c × (desvio_global / max(desvio_local, 1)) × (r − m)
```

`r` é a intensidade de entrada, `m` é a média local e `c` é o parâmetro de contraste. O desvio global é calculado sobre toda a imagem de entrada. O piso de 1 evita divisão por zero e estabiliza regiões com pouca variação; vizinhanças constantes preservam a intensidade. A saída é arredondada e limitada a [0,255].

Os cálculos das estatísticas usam precisão `double`. Os mapas exibidos e salvos são convertidos para 8 bits na escala 0..255, sem normalização individual.

## Parâmetros

| Slider | Faixa e interpretação | Padrão |
| --- | --- | --- |
| Limiar | 0..255 | 128 |
| Gamma x100 | 0..300; dividido por 100, com mínimo efetivo de 0,01 | 0,50 |
| Contraste min | 0..255; deve ser menor que Contraste max | 60 |
| Contraste max | 0..255 | 190 |
| Contraste C x10 | 0..50; dividido por 10, multiplica o ganho adaptativo | 1,5 |
| Vizinhanca | Índices 0..4 correspondem a 1×1, 3×3, 5×5, 7×7 e 9×9 | 3×3 |
| HighBoost K x10 | 0..50; dividido por 10 | 1,5 |
| Dissolve alpha % | 0..100; dividido por 100 | 50% |

Os sliders afetam somente as operações correspondentes. Alterar um slider não recalcula o resultado automaticamente.

## Arquivos de saída

Os resultados da interface são gravados em `assets/output/interface/`. A pasta é criada automaticamente, e o nome combina a operação com o nome da imagem A, por exemplo:

```text
negativo_input.png
sobel_magnitude_barbara.png
mediana_mell.png
contraste_adaptativo_input.png
```

A mediana também salva `ruido_<imagem>.png`. O contraste adaptativo salva `media_local_<imagem>.png` e `desvio_local_<imagem>.png`.

Repetir uma operação sobre a mesma imagem sobrescreve o arquivo correspondente; os parâmetros não fazem parte do nome. O status da interface informa o caminho salvo, os detalhes da operação ou eventuais erros.

## Organização do código

```text
laboratorio_PDI/
├── CMakeLists.txt
├── README.md
├── FRONTEND.md
├── assets/
│   ├── input/
│   └── output/
│       └── interface/
├── include/                 # Interfaces dos módulos (.hpp)
├── src/
│   ├── main.cpp             # Inicialização da interface
│   ├── frontend.cpp         # Janelas, controles, previews e execução
│   ├── image_io.cpp         # Leitura, validação e gravação
│   ├── intensity.cpp        # Transformações de intensidade
│   ├── histogram.cpp        # Histograma e contraste adaptativo
│   ├── convolution.cpp      # Convolução genérica
│   ├── convolution_tests.cpp # Máscaras e casos obrigatórios da AV1
│   ├── spatial_filters.cpp  # Filtros de média e mediana
│   ├── edge_detection.cpp   # Roberts, Sobel, Kirsch e Laplaciano
│   ├── sharpening.cpp       # Aguçamento Laplaciano e High Boost
│   └── dissolve.cpp         # Combinação de imagens
└── build/                   # Arquivos de compilação e executável
```

Cada módulo de `src/` possui seu cabeçalho correspondente em `include/`, com exceção de `main.cpp`. Consulte também [FRONTEND.md](FRONTEND.md) para detalhes da interface.
