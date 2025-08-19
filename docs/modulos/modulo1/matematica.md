---
sidebar_position: 4
title: "Revisão Matemática com NumPy"
description: "Álgebra linear aplicada, operações matriciais e visualização com Matplotlib"
tags: [matemática, numpy, álgebra-linear, matplotlib]
---

# Revisão Matemática com NumPy (3h)

## Álgebra Linear Aplicada (1h20min)

A Álgebra Linear é a linguagem fundamental do Deep Learning. Compreender como os dados são representados e manipulados é crucial para construir e otimizar redes neurais.

### Vetores e matrizes como representação de dados (20min)

No Machine Learning, os dados são naturalmente representados como entidades matemáticas que podem ser processadas eficientemente.

- **Escalares**: São as entidades matemáticas mais simples, representando um único número. No contexto de arrays NumPy, um escalar é um array de 0 dimensões (0-D array). Por exemplo, a temperatura de um pixel ou uma única coordenada podem ser representadas como escalares.

```python
import numpy as np
scalar = np.array(25.5)
print(f"Escalar: {scalar}, Dimensões: {scalar.ndim}")
# Saída: Escalar: 25.5, Dimensões: 0
```

- **Vetores**: Constituem arrays ordenados de números únicos, encapsulando uma quantidade que possui tanto magnitude quanto direção. Em NumPy, são arrays de 1 dimensão (1-D arrays). No Deep Learning, vetores são frequentemente empregados para representar entradas de características, como coordenadas GPS (latitude, longitude), ou as saídas de camadas individuais de uma rede.

```python
vector = np.array([40.7, -74.0])
print(f"Vetor: {vector}, Dimensões: {vector.ndim}")
# Saída: Vetor: [ 40.7 -74. ], Dimensões: 1
```

- **Matrizes**: São arranjos retangulares de números, organizados em linhas e colunas. No NumPy, correspondem a arrays de 2 dimensões (2-D arrays). As matrizes são amplamente utilizadas para armazenar os pesos das redes neurais e para realizar transformações lineares sobre os dados. Uma imagem em escala de cinza, por exemplo, pode ser representada como uma matriz (altura x largura).

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matriz:\n{matrix}, Dimensões: {matrix.ndim}, Forma: {matrix.shape}")
# Saída:
# Matriz:
# [[1 2 3]
#  [4 5 6]], Dimensões: 2, Forma: (2, 3)
```

- **Tensores**: Representam a generalização mais abrangente para entidades matemáticas que encapsulam escalares, vetores e matrizes, estendendo o conceito para um número arbitrário de dimensões (N-D arrays). Tensores são a estrutura de dados fundamental em todas as modernas frameworks de Deep Learning, incluindo NumPy, PyTorch e TensorFlow.

A forma como os dados são representados em tensores é crucial para o Deep Learning:

- **Imagens Coloridas (RGB)**: São tipicamente representadas como um tensor 3D (altura x largura x 3 canais, para vermelho, verde e azul).

```python
# Exemplo de imagem RGB (256x256 pixels com 3 canais de cor)
image_rgb = np.zeros((256, 256, 3))
print(f"Tensor de Imagem RGB: Forma: {image_rgb.shape}, Dimensões: {image_rgb.ndim}")
# Saída: Tensor de Imagem RGB: Forma: (256, 256, 3), Dimensões: 3
```

- **Lotes (Batches) de Imagens**: Para otimizar a eficiência do treinamento, os dados são processados em pequenos subconjuntos, ou "lotes". Um lote de imagens coloridas seria, portanto, um tensor 4D (tamanho_do_lote x altura x largura x canais).

```python
# Exemplo de um lote de 32 imagens RGB (cada uma 256x256 pixels)
batch_images = np.zeros((32, 256, 256, 3))
print(f"Tensor de Lote de Imagens: Forma: {batch_images.shape}, Dimensões: {batch_images.ndim}")
# Saída: Tensor de Lote de Imagens: Forma: (32, 256, 256, 3), Dimensões: 4
```

- **Séries Temporais Multiespectrais (Sensoriamento Remoto)**: Dados de satélite coletados ao longo do tempo, que incluem múltiplas bandas espectrais, podem ser representados como tensores de 5 dimensões (tamanho_do_lote x tempo x altura x largura x canais espectrais).

```python
# Exemplo de lote de séries temporais multiespectrais
# (16 amostras, 10 timestamps, 128x128 pixels, 7 bandas espectrais)
time_series_data = np.zeros((16, 10, 128, 128, 7))
print(f"Tensor de Série Temporal Multiespectral: Forma: {time_series_data.shape}, Dimensões: {time_series_data.ndim}")
# Saída: Tensor de Série Temporal Multiespectral: Forma: (16, 10, 128, 128, 7), Dimensões: 5
```

Compreender a conexão entre a representação matemática e a implementação computacional é fundamental para o Deep Learning, pois permite o manuseio compacto e eficiente de grandes volumes de dados em hardware moderno, como GPUs.

### Operações matriciais essenciais (25min)

NumPy (Numerical Python) é uma biblioteca essencial em Python para computação numérica, fornecendo estruturas de dados de array multidimensionais altamente eficientes, conhecidas como ndarray, e uma vasta coleção de funções otimizadas para operar sobre elas.

- **Multiplicação de Matrizes (Dot Product)**: Esta é a multiplicação matricial padrão da álgebra linear, onde as linhas da primeira matriz são multiplicadas pelas colunas da segunda. Esta operação é fundamental para a propagação de sinais através das camadas de uma rede neural, onde as entradas são combinadas com os pesos da camada para produzir as saídas. Para numpy.array(), a função np.dot() ou o operador @ (disponível a partir do Python 3.5) são utilizados para essa finalidade.

```python
import numpy as np

# Matriz A (2x3)
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Matriz B (3x2)
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Multiplicação de matrizes usando @
C = A @ B
print(f"Multiplicação de Matrizes (A @ B):\n{C}")
# Saída:
# Multiplicação de Matrizes (A @ B):
# [[ 58  64]
#  [139 154]]

# Multiplicação de matrizes usando np.dot()
D = np.dot(A, B)
print(f"Multiplicação de Matrizes (np.dot(A, B)):\n{D}")
# Saída:
# Multiplicação de Matrizes (np.dot(A, B)):
# [[ 58  64]
#  [139 154]]
```

- **Produto Elemento a Elemento (Hadamard Product)**: Também conhecido como produto entrywise ou Schur product, esta operação envolve a multiplicação dos elementos correspondentes de duas matrizes ou tensores que possuem as mesmas dimensões. Em NumPy, o operador * (para ndarray) ou a função np.multiply() realizam o produto Hadamard. Este tipo de multiplicação é frequentemente empregado em funções de ativação, operações de masking (como no dropout) ou para escalonar elementos individualmente.

```python
# Matriz A (2x2)
A_elem = np.array([[1, 2],
                   [3, 4]])

# Matriz B (2x2)
B_elem = np.array([[5, 6],
                   [7, 8]])

# Produto elemento a elemento usando *
C_elem = A_elem * B_elem
print(f"Produto Elemento a Elemento (A_elem * B_elem):\n{C_elem}")
# Saída:
# Produto Elemento a Elemento (A_elem * B_elem):
# [[ 5 12]
#  [21 32]]

# Produto elemento a elemento usando np.multiply()
D_elem = np.multiply(A_elem, B_elem)
print(f"Produto Elemento a Elemento (np.multiply(A_elem, B_elem)):\n{D_elem}")
# Saída:
# Produto Elemento a Elemento (np.multiply(A_elem, B_elem)):
# [[ 5 12]
#  [21 32]]
```

- **Transposição**: A transposição de uma matriz ou tensor é uma operação que envolve a troca de suas linhas por colunas, ou vice-versa, efetivamente refletindo-a ao longo de sua diagonal principal. No NumPy, essa operação é realizada de forma simples e eficiente utilizando o atributo .T ou a função transpose(). Esta operação é fundamental em várias etapas do Deep Learning, como no cálculo de gradientes durante a retropropagação (backpropagation) ou no reajuste da forma dos dados para garantir a compatibilidade com outras operações matriciais.

```python
# Matriz original (2x3)
original_matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])

# Transposição usando .T
transposed_matrix_T = original_matrix.T
print(f"Matriz Original:\n{original_matrix}")
print(f"Matriz Transposta (.T):\n{transposed_matrix_T}")
# Saída:
# Matriz Original:
# [[1 2 3]
#  [4 5 6]]
# Matriz Transposta (.T):
# [[1 4]
#  [2 5]
#  [3 6]]

# Transposição usando transpose()
transposed_matrix_func = np.transpose(original_matrix)
print(f"Matriz Transposta (np.transpose()):\n{transposed_matrix_func}")
# Saída:
# Matriz Transposta (np.transpose()):
# [[1 4]
#  [2 5]
#  [3 6]]
```

- **Soma, Subtração e Escalonamento**: Essas operações se comportam de maneira intuitiva em tensores multidimensionais, aplicando-se elemento a elemento.

```python
# Matriz A
A_ops = np.array([[1, 2],
                  [3, 4]])

# Matriz B
B_ops = np.array([[5, 6],
                  [7, 8]])

# Soma
soma = A_ops + B_ops
print(f"Soma (A_ops + B_ops):\n{soma}")
# Saída:
# Soma (A_ops + B_ops):
# [[ 6  8]
#  [10 12]]

# Subtração
subtracao = A_ops - B_ops
print(f"Subtração (A_ops - B_ops):\n{subtracao}")
# Saída:
# Subtração (A_ops - B_ops):
# [[-4 -4]
#  [-4 -4]]

# Escalonamento (multiplicação por um escalar)
escalonamento = A_ops * 2
print(f"Escalonamento (A_ops * 2):\n{escalonamento}")
# Saída:
# Escalonamento (A_ops * 2):
# [[2 4]
#  [6 8]]
```

### Broadcasting e vetorização

A eficiência computacional é um fator crítico no Deep Learning, e o NumPy oferece mecanismos poderosos para otimizá-la:

- **Vetorização**: Refere-se à prática de expressar tarefas de processamento de dados como expressões concisas de array, em vez de loops explícitos em Python. Operações vetorizadas são significativamente mais rápidas porque são executadas em código C otimizado subjacente, o que é essencial para lidar com grandes volumes de dados em Deep Learning.

- **Broadcasting**: É um conjunto de regras que permite ao NumPy realizar operações aritméticas binárias em arrays com diferentes formas (dimensões) sem a necessidade de criar cópias explícitas do array menor. O array menor é "esticado" ou "duplicado" virtualmente para corresponder à forma do array maior, permitindo operações elemento a elemento.

As regras de Broadcasting são as seguintes:

1. Se os arrays diferem no número de dimensões, a forma do array com menos dimensões é preenchida com 1s no lado esquerdo.
2. Se as formas dos arrays não correspondem em qualquer dimensão, o array com dimensão 1 nessa posição é "esticado" para corresponder à outra forma.
3. Se nenhuma das regras anteriores se aplica e as dimensões ainda não correspondem, a operação de broadcasting falha.

Exemplos práticos de broadcasting incluem:

- **Adicionar um viés (bias) a uma matriz de ativações**: Um vetor de viés pode ser adicionado a cada linha de uma matriz de características, onde o vetor é transmitido (broadcast) ao longo das linhas.

```python
# Matriz de ativações (3x4)
activations = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])

# Vetor de bias (1x4)
bias = np.array([0.1, 0.2, 0.3, 0.4])

# Adicionar bias usando broadcasting
output = activations + bias
print(f"Ativações com Bias (Broadcasting):\n{output}")
# Saída:
# Ativações com Bias (Broadcasting):
# [[ 1.1  2.2  3.3  4.4]
#  [ 5.1  6.2  7.3  8.4]
#  [ 9.1 10.2 11.3 12.4]]
```

- **Normalizar dados por canal**: Subtrair a média de cada coluna de um conjunto de dados, onde o vetor de médias é transmitido (broadcast) sobre as linhas do conjunto de dados.

```python
# Dados de exemplo (10 observações, 3 características/canais)
data = np.random.rand(10, 3)

# Calcular a média de cada característica (ao longo do eixo 0)
mean_per_feature = data.mean(axis=0)
print(f"Média por característica: {mean_per_feature}")

# Normalizar os dados subtraindo a média usando broadcasting
normalized_data = data - mean_per_feature
print(f"Dados Normalizados (primeiras 3 linhas):\n{normalized_data[:3]}")
# Saída (exemplo, valores variam):
# Média por característica: [0.5123... 0.4876... 0.5012...]
# Dados Normalizados (primeiras 3 linhas):
# [[-0.21... -0.18... -0.00...]
#  [ 0.15...  0.01...  0.39...]
#  [-0.40... -0.46... -0.49...]]
```

- **Aplicar transformações elemento-wise**: Multiplicar uma imagem RGB (HxWx3) por um array 1D de 3 valores para escalar cada canal de cor independentemente.

```python
# Imagem RGB simulada (2x2 pixels, 3 canais)
image = np.array([[[255, 0, 0], [0, 255, 0]],
                  [[0, 0, 255], [128, 128, 128]]], dtype=np.float32)

# Fatores de escala para cada canal (R, G, B)
scale_factors = np.array([0.5, 0.8, 1.2])

# Escalar canais usando broadcasting
scaled_image = image * scale_factors
print(f"Imagem Original:\n{image}")
print(f"Fatores de Escala: {scale_factors}")
print(f"Imagem Escalonada (Broadcasting):\n{scaled_image}")
# Saída:
# Imagem Original:
# [[[255.   0.   0.]
#   [  0. 255.   0.]]
#  [[  0.   0. 255.]
#   [128. 128. 128.]]]
# Fatores de Escala: [0.5 0.8 1.2]
# Imagem Escalonada (Broadcasting):
# [[[127.5   0.    0. ]
#   [  0.  204.    0. ]]
#  [[  0.    0.  306. ]
#   [ 64.  102.4 153.6]]]
```

## Plotagem de Gráficos e Imagens com Matplotlib no Google Colab

A visualização de dados é uma etapa fundamental em qualquer processo de análise, permitindo a compreensão de padrões, tendências e anomalias. No campo da ciência de dados e visão computacional, ferramentas como Matplotlib e OpenCV, em conjunto com NumPy, são indispensáveis para transformar dados brutos em representações visuais significativas e para manipular imagens de forma programática.

### Configuração do Ambiente e Fundamentos do Matplotlib

Para iniciar a jornada de visualização de dados no Google Colab, é crucial configurar o ambiente corretamente e compreender a estrutura fundamental do Matplotlib.

#### Configuração Essencial para Google Colab

A primeira linha de código em qualquer notebook Colab que envolva plotagem com Matplotlib deve ser `%matplotlib inline`. Esta "magic command" instrui o ambiente a renderizar os gráficos diretamente no notebook, logo abaixo da célula de código que os gera, em vez de abri-los em janelas separadas.

```python
# Configuração para exibir gráficos inline no Google Colab
%matplotlib inline

# Importações essenciais
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Para carregar imagens
import cv2  # Para processamento de imagens (se necessário)

print("Ambiente configurado e bibliotecas importadas com sucesso!")
```

#### A Anatomia de um Plot Matplotlib: Figure e Axes

O Matplotlib adota uma estrutura hierárquica para a construção de gráficos, que é fundamental para um controle preciso sobre a visualização. No topo dessa hierarquia está a **Figure**, que pode ser concebida como a "tela" ou "janela" principal onde o gráfico será desenhado. Dentro de uma Figure, um ou mais objetos **Axes** são definidos.

A maneira mais recomendada para iniciar a criação de um gráfico é utilizando `plt.subplots()`. Esta função retorna uma tupla contendo a Figure e um ou mais objetos Axes, simplificando a configuração inicial.

```python
fig, ax = plt.subplots(figsize=(8, 6))  # Cria uma Figure e um único Axes
ax.set_title("Título do Gráfico")
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
plt.show()  # Exibe o gráfico
```

### Plotagem de Gráficos Fundamentais com Matplotlib

#### Gráficos de Linha (plt.plot)

```python
# Dados de exemplo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Criação do plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando múltiplas linhas com personalização
ax.plot(x, y1, color='blue', linestyle='-', linewidth=2, label='Seno')
ax.plot(x, y2, 'ro--', markersize=5, label='Cosseno')  # String de formato

# Adicionando título, rótulos e legenda
ax.set_title('Gráfico de Seno e Cosseno', fontsize=16)
ax.set_xlabel('Ângulo (radianos)', fontsize=12)
ax.set_ylabel('Magnitude', fontsize=12)
ax.legend(loc='upper right', fontsize=10)  # Posição da legenda
ax.grid(True, linestyle=':', alpha=0.7)  # Adiciona grade

plt.show()
```

#### Gráficos de Dispersão (plt.scatter)

```python
# Dados de exemplo
np.random.seed(42)  # Para reprodutibilidade
num_pontos = 100
x_data = np.random.rand(num_pontos) * 10
y_data = 2 * x_data + np.random.randn(num_pontos) * 5
tamanhos = np.random.rand(num_pontos) * 500  # Tamanho dos pontos
cores = np.random.rand(num_pontos)  # Valores para mapear a cor

# Criação do plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando o gráfico de dispersão com personalização
scatter = ax.scatter(x_data, y_data, s=tamanhos, c=cores, cmap='viridis', alpha=0.7,
                    edgecolors='w', linewidth=0.5, label='Pontos de Dados')

# Adicionando título e rótulos
ax.set_title('Gráfico de Dispersão Personalizado', fontsize=16)
ax.set_xlabel('Variável X', fontsize=12)
ax.set_ylabel('Variável Y', fontsize=12)

# Adicionando colorbar para o mapeamento de cores
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Valor de Cor', fontsize=12)

ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)
plt.show()
```

### Visualização e Processamento de Imagens

#### Exibindo Imagens (plt.imshow)

```python
# Para carregar uma imagem de uma URL no Colab:
import requests
from io import BytesIO

# URL de uma imagem de exemplo
image_url = "https://matplotlib.org/_static/stinkbug.png"
response = requests.get(image_url)
img_pil = Image.open(BytesIO(response.content))
img_np = np.array(img_pil)

print(f"Shape da imagem original: {img_np.shape}")

# Exibindo imagem RGB/RGBA
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_np)
ax.set_title('Imagem Original (RGB/RGBA)', fontsize=16)
ax.axis('off')  # Remove os eixos para visualização de imagem
plt.show()

# Exibindo imagem em escala de cinza (canal de luminância)
lum_img = img_np[:, :, 0]  # Pega o primeiro canal (Red) como luminância
print(f"Shape da imagem em escala de cinza: {lum_img.shape}")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(lum_img, cmap='gray', origin='upper', interpolation='bilinear')
fig.colorbar(im, ax=ax, label='Intensidade do Pixel')
ax.set_title('Imagem em Escala de Cinza com Colormap', fontsize=16)
ax.axis('off')
plt.show()
```