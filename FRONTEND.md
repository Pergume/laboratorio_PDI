# Frontend do Laboratorio PDI

A interface foi implementada em C++ usando apenas os modulos `highgui` e `imgproc` do OpenCV para **janela, desenho, preview e sliders**. Os algoritmos avaliados continuam nos arquivos originais e nao foram substituidos por funcoes prontas do OpenCV.

## Execucao

Na raiz do projeto:

```bash
cmake -S . -B build
cmake --build build
./build/laboratorio_PDI
```

Execute sempre a partir da raiz para que os caminhos `assets/input/...` sejam encontrados.

## Imagens disponiveis

O botao `Imagem A` alterna entre:

- `input.tif`
- `barbara.tif`
- `textura.jpg`

O botao `Imagem B` seleciona a segunda imagem usada somente nas operacoes de dissolve.

`input.tif` e `barbara.tif` possuem 512x512 e podem ser usadas diretamente no dissolve. Se A e B tiverem dimensoes diferentes, a propria implementacao do dissolve informa o erro.

## Categorias

### Intensidade
- Negativo
- Limiarizacao
- Gamma
- Logaritmo
- Alargamento de contraste

### Histograma
- Expansao
- Equalizacao
- Contraste adaptativo

### Filtros
- Media
- Mediana
- Convolucao 3x3 de aguçamento

### Bordas
- Roberts: |Gx|, |Gy| e magnitude
- Sobel: |Gx|, |Gy| e magnitude
- Kirsch: oito respostas direcionais e resposta final
- Laplaciano

### Agucamento
- Resposta Laplaciana
- Agucamento Laplaciano
- High Boost

### Dissolve
- Uniforme
- Nao uniforme com mapa horizontal 0..1

### Testes AV1
- Aguçamento c=1,d=1
- Aguçamento c=2,d=1
- Aguçamento c=1,d=2
- Relevo h1
- Relevo h2
- Bordas h3
- Bordas h4 com saturacao direta
- Bordas h4 com modulo da resposta bruta

A versao `h4 saturado` preserva exatamente o comportamento do teste anterior: como a mascara produz respostas negativas, a conversao para [0,255] pode gerar uma imagem preta. A versao `h4 |resposta|` e uma visualizacao auxiliar para enxergar as bordas sem alterar a mascara fornecida pelo professor.

## Sliders

Os oito sliders ficam na janela separada `Parametros - Laboratorio PDI`. A janela principal pode ser redimensionada para caber na tela. Os controles e so afetam as operacoes correspondentes:

- `Limiar`: 0..255
- `Gamma x100`: ex. 50 = gamma 0.50, 200 = gamma 2.00
- `Contraste min` e `Contraste max`
- `Contraste C x10`: ex. 15 = c 1.5
- `Vizinhanca`: 0..4 corresponde a 1x1, 3x3, 5x5, 7x7 e 9x9
- `HighBoost K x10`: ex. 15 = k 1.5
- `Dissolve alpha %`: 0..100

O botao `Restaurar padroes` retorna todos os sliders aos valores iniciais: limiar 128, gamma 0.50, contraste min 60 e max 190, contraste C 1.5, vizinhanca 3x3, High Boost K 1.5 e dissolve 50%. Clique novamente em uma operacao para aplicar os valores restaurados.

Depois de mudar um slider, clique novamente na operacao para executa-la com o novo valor.

## Saidas

Cada operacao executada pelo frontend e salva automaticamente em:

```text
assets/output/interface/
```

O nome inclui a operacao e a imagem A selecionada, por exemplo:

```text
sobel_magnitude_input.png
media_barbara.png
teste_bordas_h4_modulo_input.png
```

## Estrutura adicionada

```text
include/frontend.hpp
src/frontend.cpp
FRONTEND.md
```

O `main.cpp` agora apenas inicializa `pdi::runFrontend()`. Toda a logica dos algoritmos continua nos modulos anteriores.

## Comparacao da mediana e estatisticas locais

Ao executar a mediana, o painel esquerdo mostra a entrada com ruido sal e pimenta (probabilidade de 10% por pixel, metade preto e metade branco), e o direito mostra a filtragem dessa mesma entrada. A amostra de ruido permanece a mesma ao repetir a operacao e mudar a vizinhanca. A imagem com ruido tambem e salva como `ruido_<imagem>.png`.

O contraste adaptativo mostra quatro paineis: original, resultado, media local e desvio padrao local. As estatisticas sao calculadas manualmente sobre a vizinhanca, recortada nas bordas, com variancia populacional `E[r²] - E[r]²`. O realce usa `s = m + c * (desvio_global / max(desvio_local, 1)) * (r - m)`, com arredondamento e saturacao em [0,255]. O piso de 1 estabiliza regioes com pouca variacao; regioes constantes permanecem constantes. O desvio global usa todos os pixels da imagem de entrada.

Os mapas usam a escala de intensidade 0..255, sem normalizacao individual, e sao salvos como `media_local_<imagem>.png` e `desvio_local_<imagem>.png`. Os calculos preservam precisao double; somente os previews e os PNGs sao arredondados para 8 bits. O controle C multiplica agora o ganho relativo entre os desvios global e local.
