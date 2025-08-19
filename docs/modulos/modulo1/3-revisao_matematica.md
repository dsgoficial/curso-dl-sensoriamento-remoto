---
prev_page: "/modulos/modulo1/2-setup/"
next_page: "/modulos/modulo1/4-fundamentos_proc_imagens/"
---

## Revisão Matemática com NumPy (3h)

### Álgebra Linear Aplicada (1h20min)

A Álgebra Linear é a linguagem fundamental do Deep Learning. Compreender como os dados são representados e manipulados é crucial para construir e otimizar redes neurais.

#### Vetores e matrizes como representação de dados (20min)

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

#### Operações matriciais essenciais (25min)

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

#### Broadcasting e vetorização

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

### numpy.reshape

A função numpy.reshape é uma ferramenta poderosa para alterar a estrutura dimensional de um array NumPy sem, no entanto, modificar os dados subjacentes. Sua principal finalidade é reorganizar a interpretação dos elementos do array em uma nova forma, mantendo o número total de elementos constante.¹ Esta funcionalidade é frequentemente utilizada para transformar arrays de uma dimensão para outra, como converter um vetor unidimensional em uma matriz bidimensional, ou vice-versa, ou ainda ajustar as dimensões de uma matriz para uma nova configuração.


A função numpy.reshape aceita os seguintes parâmetros principais:

- **a**: O array de entrada que se deseja remodelar. Este pode ser qualquer objeto que possa ser interpretado como um array.¹

- **shape**: Um inteiro ou uma tupla de inteiros que especifica a nova forma desejada para o array. É imperativo que o produto das dimensões na nova forma seja compatível com o número total de elementos no array original. Se o shape for um único inteiro, o resultado será um array 1-D com esse comprimento.¹

  Uma característica notável do parâmetro shape é a possibilidade de utilizar -1 para uma das dimensões. Quando -1 é fornecido, o NumPy infere automaticamente o tamanho dessa dimensão com base no comprimento total do array e nas outras dimensões especificadas.¹ Por exemplo, se um array tem 6 elementos e se deseja remodelá-lo para (3, -1), o NumPy inferirá que a segunda dimensão deve ser 2, resultando em uma forma (3, 2). Essa capacidade de inferência de dimensão simplifica significativamente o código, tornando-o mais robusto e menos propenso a erros de cálculo manual de dimensões, o que é particularmente valioso em pipelines de dados complexos ou ao lidar com dados de tamanho variável.

- **order**: Este é um parâmetro opcional que define a ordem na qual os elementos do array de entrada são lidos e, subsequentemente, colocados no array remodelado. As opções incluem 'C', 'F' e 'A'.¹
  - **'C'** (padrão): Indica uma ordem de índice C-like (linha principal), onde o índice do último eixo muda mais rapidamente, e o do primeiro eixo muda mais lentamente.
  - **'F'**: Indica uma ordem de índice Fortran-like (coluna principal), onde o índice do primeiro eixo muda mais rapidamente, e o do último eixo muda mais lentamente.
  - **'A'**: Significa que os elementos serão lidos/escritos em ordem Fortran-like se o array for contíguo em Fortran na memória, e C-like caso contrário.

É crucial notar que as opções 'C' e 'F' referem-se estritamente à ordem de indexação ou iteração lógica dos elementos, e não necessariamente ao layout físico dos dados na memória subjacente do array.¹

### Valor de Retorno

A função numpy.reshape retorna um novo objeto ndarray com a forma especificada. Este objeto será uma view (visão) do array original sempre que for possível. Se a nova forma não puder ser obtida através de uma view (por exemplo, devido a uma mudança na ordem de leitura/escrita que exige uma reorganização física dos dados), uma cópia do array será feita. É importante ressaltar que não há garantia quanto ao layout de memória (C-contíguo ou Fortran-contíguo) do array retornado.¹

### Exemplos Práticos

A seguir, são apresentados exemplos que ilustram o uso de numpy.reshape em diferentes cenários:

#### Exemplo 1: Array 1D para 2D

```python
import numpy as np

arr_1d = np.arange(6)
print("Array 1D original:\n", arr_1d)

# Remodelando para 2x3
arr_2d = np.reshape(arr_1d, (2, 3))
print("\nArray 2D (2x3):\n", arr_2d)
```

**Saída:**
```
Array 1D original:
[0 1 2 3 4 5]

Array 2D (2x3):
[[0 1 2]
 [3 4 5]]
```

#### Exemplo 2: Array 2D para 1D

```python
arr_2d_orig = np.array([[0, 1, 2], [3, 4, 5]])
print("Array 2D original:\n", arr_2d_orig)

# Remodelando para 1D usando -1
arr_flat = np.reshape(arr_2d_orig, -1)  # ou (6,)
print("\nArray 1D (achatado):\n", arr_flat)
```

**Saída:**
```
Array 2D original:
[[0 1 2]
 [3 4 5]]

Array 1D (achatado):
[0 1 2 3 4 5]
```

#### Exemplo 3: Uso de order='F'

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print("Array original:\n", a)

# Remodelando com ordem 'C' (padrão)
reshaped_c = np.reshape(a, (3, 2), order='C')
print("\nRemodelado (3x2, order='C'):\n", reshaped_c)

# Remodelando com ordem 'F'
reshaped_f = np.reshape(a, (3, 2), order='F')
print("\nRemodelado (3x2, order='F'):\n", reshaped_f)
```

**Saída:**
```
Array original:
[[1 2 3]
 [4 5 6]]

Remodelado (3x2, order='C'):
[[1 2]
 [3 4]
 [5 6]]

Remodelado (3x2, order='F'):
[[1 4]
 [2 5]
 [3 6]]
```

Neste exemplo, a diferença entre as ordens 'C' e 'F' é evidente. A ordem 'C' (padrão) lê os elementos linha por linha (1, 2, 3, 4, 5, 6) e os preenche na nova forma seguindo a mesma lógica. Em contraste, a ordem 'F' lê os elementos coluna por coluna (1, 4, 2, 5, 3, 6) e os insere na nova forma de acordo com essa sequência. A compreensão de como o parâmetro order influencia a reorganização lógica dos dados é crucial para prever o resultado da operação.¹

### Casos de Uso do reshape

numpy.reshape é amplamente utilizado em diversas aplicações:

- **Preparação de Dados**: É comum transformar dados para que se adequem aos requisitos de entrada de algoritmos de Machine Learning, como converter imagens 2D em vetores 1D achatados ou adicionar uma dimensão de canal para modelos de redes neurais.

- **Processamento de Imagem**: Reorganizar pixels de uma imagem para diferentes representações ou para aplicar filtros e transformações.

- **Análise de Séries Temporais**: Ajustar a forma de sequências de dados para que se encaixem em modelos específicos que esperam uma determinada estrutura de entrada.

### Considerações Importantes

#### View vs. Copy

A função numpy.reshape é otimizada para eficiência, e, por isso, tenta retornar uma view do array original sempre que possível. Isso implica que, se o array remodelado for uma view, quaisquer modificações realizadas nele afetarão diretamente o array original, uma vez que ambos compartilham os mesmos dados subjacentes. Uma cópia do array é criada apenas quando a nova forma não pode ser obtida através de uma view, o que geralmente ocorre quando uma mudança na ordem de leitura/escrita (order='F' para um array C-contíguo, por exemplo) exige uma reorganização física dos dados na memória. A ausência de garantia sobre o layout de memória do array retornado significa que o desenvolvedor deve estar ciente dessa dualidade entre view e copy para evitar efeitos colaterais inesperados. Para garantir uma independência total entre o array original e o remodelado, é aconselhável chamar explicitamente o método .copy() após a operação de reshape, embora isso acarrete um custo adicional de desempenho e memória.¹

#### Compatibilidade de Forma

O número total de elementos no array deve ser idêntico antes e depois da operação de reshape. Se as formas especificadas forem incompatíveis com o número total de elementos, um ValueError será levantado.

#### O parâmetro -1

A flexibilidade oferecida pelo uso de -1 no parâmetro shape permite que uma das dimensões seja calculada automaticamente pelo NumPy. Essa funcionalidade é uma conveniência poderosa, garantindo que o número total de elementos seja mantido e simplificando a codificação, especialmente em cenários onde o tamanho exato de uma dimensão pode não ser conhecido de antemão ou pode variar dinamicamente.

## numpy.stack: Empilhando Arrays ao Longo de um Novo Eixo

A função numpy.stack() é empregada para unir uma sequência de arrays (ou objetos array-like) ao longo de um novo eixo. Sua funcionalidade distingue-se fundamentalmente da numpy.concatenate(), que une arrays ao longo de um eixo existente. Enquanto concatenate() mantém a dimensionalidade do array resultante, stack() aumenta a dimensionalidade dos arrays de entrada em um. Por exemplo, stack() pode transformar arrays 1D em um array 2D, ou arrays 2D em um array 3D.²

Esta distinção é vital para estruturar dados corretamente, especialmente em áreas como Machine Learning, onde a forma dos tensores é crítica. Um erro na escolha entre stack e concatenate pode levar a incompatibilidades de forma e falhas em algoritmos.

### Parâmetros de numpy.stack

A sintaxe básica para numpy.stack() é `numpy.stack((a1, a2,...), axis=0)`.² Os parâmetros são:

- **tup**: Uma sequência (como uma tupla ou lista) de arrays a serem empilhados. Um requisito crucial é que todos os arrays nesta sequência devem ter a mesma forma exata.² Se as formas diferirem, a função levantará um erro.

- **axis**: Um inteiro que especifica o índice do novo eixo no array resultante ao longo do qual os arrays de entrada serão empilhados. O valor padrão para axis é 0.²

### Valor de Retorno

A função retorna um único ndarray que é o resultado do empilhamento dos arrays fornecidos.²

### Exemplos Práticos

Os exemplos a seguir demonstram o comportamento de numpy.stack():

#### Exemplo 1: Empilhando Arrays 1D (axis=0 - padrão)

```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
c = np.stack((a, b))  # axis=0 por padrão
print("Arrays 1D empilhados (axis=0):\n", c)
print("Forma do array resultante:", c.shape)
```

**Saída:**
```
Arrays 1D empilhados (axis=0):
[[1 2]
 [3 4]]
Forma do array resultante: (2, 2)
```

#### Exemplo 2: Empilhando Arrays 1D (axis=1)

```python
a = np.array([1, 2])
b = np.array([3, 4])
c = np.stack((a, b), axis=1)
print("\nArrays 1D empilhados (axis=1):\n", c)
print("Forma do array resultante:", c.shape)
```

**Saída:**
```
Arrays 1D empilhados (axis=1):
[[1 3]
 [2 4]]
Forma do array resultante: (2, 2)
```

#### Exemplo 3: Empilhando Arrays 2D (axis=0)

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.stack((a, b))
print("\nArrays 2D empilhados (axis=0):\n", c)
print("Forma do array resultante:", c.shape)
```

**Saída:**
```
Arrays 2D empilhados (axis=0):
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
Forma do array resultante: (2, 2, 2)
```

### Casos de Uso

A função stack() é particularmente útil em cenários onde é necessário combinar arrays adicionando uma nova dimensão, em vez de simplesmente concatená-los ao longo de uma dimensão existente²:

- **Criação de Lotes de Dados para Machine Learning**: É frequentemente utilizada para empilhar amostras individuais (por exemplo, imagens 2D ou sequências de tempo 1D) para formar um lote 3D ou 2D, onde o novo eixo representa a dimensão do lote para processamento em redes neurais.

- **Combinação de Features**: Permite agrupar diferentes tipos de features (como canais de cor RGB de imagens ou leituras de múltiplos sensores) para uma única observação multidimensional.

- **Organização de Resultados Experimentais**: Facilita a comparação e análise de resultados ao empilhar arrays de resultados provenientes de múltiplos experimentos.

### Considerações Importantes

#### Requisito de Mesma Forma

O ponto mais crítico ao usar numpy.stack() é que todos os arrays de entrada (a1, a2,...) devem ter a mesma forma exata. Qualquer diferença nas suas dimensões resultará em um erro.²

#### Criação de Novo Eixo

Ao contrário de numpy.concatenate(), que une arrays ao longo de um eixo já existente, numpy.stack() sempre cria um novo eixo no array resultante. Isso significa que a dimensionalidade do array de saída será sempre uma unidade maior do que a dimensionalidade dos arrays de entrada.² Por exemplo, empilhar dois arrays 1D de forma (N,) resultará em um array 2D de forma (2, N), enquanto concatená-los resultaria em um array 1D de forma (2N,). Esta distinção é fundamental para estruturar dados corretamente em aplicações que dependem de formas de tensor específicas.

#### Relação com vstack e hstack

numpy.vstack() e numpy.hstack() são funções especializadas que oferecem sintaxe mais legível para operações comuns de união vertical e horizontal. Embora numpy.stack() seja uma função mais geral para unir arrays ao longo de um novo eixo, vstack() e hstack() podem ser vistas como casos específicos ou conveniências sintáticas. vstack() empilha arrays verticalmente, o que é equivalente a stack(axis=0) para arrays 1D (após uma remodelação implícita para 2D) e 2D. hstack() empilha arrays horizontalmente; para arrays 1D, é similar a concatenate ao longo do primeiro eixo, mas para arrays de maior dimensão, hstack concatena ao longo do segundo eixo, enquanto stack(axis=1) insere um novo eixo na posição 1.² A compreensão de sua relação com stack e concatenate ajuda a escolher a ferramenta mais apropriada e a prever o comportamento de dimensionalidade.

A tabela a seguir resume as principais diferenças entre as funções de empilhamento e concatenação do NumPy:

| Função | Ação Principal | Altera Dimensionalidade? | Requisito de Forma dos Arrays de Entrada | Comportamento com Arrays 1D | Eixo Padrão/Principal |
|--------|----------------|--------------------------|------------------------------------------|----------------------------|---------------------|
| numpy.stack() | Empilha ao longo de um novo eixo | Sim (aumenta em 1) | Devem ter a mesma forma exata | Transforma para 2D (e.g., (N,) -> (1, N) antes de empilhar) | axis=0 (novo eixo) |
| numpy.vstack() | Empilha verticalmente (linha por linha) | Sim (para 1D), Não (para >1D) | Mesma forma em todos os eixos, exceto o primeiro | Remodelados implicitamente para (1, N) antes de empilhar | Eixo 0 (concatenação) |
| numpy.hstack() | Empilha horizontalmente (coluna por coluna) | Não | Mesma forma em todos os eixos, exceto o segundo | Concatena diretamente ao longo do primeiro eixo | Eixo 1 (concatenação) |
| numpy.concatenate() | Une ao longo de um eixo existente | Não | Mesma forma em todos os eixos, exceto o eixo de concatenação | Concatena diretamente ao longo do eixo especificado | Requer axis explícito (ou 0 por padrão) |

## numpy.vstack: Empilhamento Vertical de Arrays

A função numpy.vstack() é uma ferramenta especializada no NumPy, projetada para empilhar arrays verticalmente, ou seja, linha por linha, para formar um único array coeso.⁴ Ela aceita uma sequência de arrays e os une ao longo do primeiro eixo (eixo 0). Uma característica importante de vstack() é seu tratamento de arrays unidimensionais: arrays com forma (N,) são implicitamente remodelados para (1, N) (ou seja, uma única linha) antes de serem empilhados.⁴ Essa remodelação automática é uma funcionalidade-chave que a torna extremamente conveniente para transformar coleções de vetores 1D em uma matriz 2D, onde cada vetor se torna uma linha.

### Parâmetros de numpy.vstack

A função numpy.vstack() possui um único parâmetro principal:

- **tup**: Uma sequência (como uma tupla ou lista) de arrays a serem empilhados. Para que a operação seja bem-sucedida, os arrays de entrada devem ter a mesma forma em todos os eixos, exceto no primeiro eixo (o eixo de concatenação). Para arrays 1D, isso significa que devem ter o mesmo comprimento.⁴ Esta restrição é fundamental para garantir que o array resultante seja uma estrutura retangular válida. Se os arrays tiverem formas diferentes ao longo do primeiro eixo, numpy.vstack() levantará um ValueError.⁴

### Valor de Retorno

numpy.vstack() retorna um novo ndarray que é o resultado do empilhamento vertical dos arrays fornecidos.⁴ É importante notar que a função não modifica os arrays originais; ela sempre produz um novo array.

### Exemplos Práticos

Os exemplos a seguir ilustram o uso de numpy.vstack() em diferentes cenários:

#### Exemplo 1: Empilhamento Vertical de Arrays 1D

```python
import numpy as np

x = np.array([3, 5, 7])
y = np.array([5, 7, 9])
result = np.vstack((x, y))
print("Arrays 1D empilhados verticalmente:\n", result)
print("Forma do array resultante:", result.shape)
```

**Saída:**
```
Arrays 1D empilhados verticalmente:
[[3 5 7]
 [5 7 9]]
Forma do array resultante: (2, 3)
```

Neste exemplo, os arrays 1D x e y são tratados como arrays (1, 3) antes do empilhamento, resultando em um array 2D de forma (2, 3).

#### Exemplo 2: Empilhamento Vertical de Arrays 2D

```python
x = np.array([[1], [2], [3]])
y = np.array([[4], [5], [6]])
result = np.vstack((x, y))
print("\nArrays 2D empilhados verticalmente:\n", result)
print("Forma do array resultante:", result.shape)
```

**Saída:**
```
Arrays 2D empilhados verticalmente:
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
Forma do array resultante: (6, 1)
```

#### Exemplo 3: Combinando Múltiplos Arrays 2D

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])
result = np.vstack((a, b, c))
print("\nMúltiplos arrays 2D empilhados verticalmente:\n", result)
print("Forma do array resultante:", result.shape)
```

**Saída:**
```
Múltiplos arrays 2D empilhados verticalmente:
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]
Forma do array resultante: (6, 2)
```

### Casos de Uso

numpy.vstack() é útil em diversas situações práticas:

- **Combinar Múltiplos Arrays 1D em um Array 2D**: Frequentemente utilizado para transformar uma coleção de vetores de características (features) em uma matriz de dados, onde cada vetor se torna uma linha.

- **Mesclar Múltiplos Arrays 2D Linha por Linha**: Ideal para combinar conjuntos de dados que possuem as mesmas colunas, mas diferentes registros (linhas), como adicionar novos pontos de dados a um dataset existente.

- **Anexar Linhas a um Array 2D Existente**: Permite adicionar novas observações ou registros a um conjunto de dados bidimensional de forma eficiente.⁴

### Considerações Importantes

#### Forma Consistente (Exceto Eixo 0)

É fundamental que todos os arrays de entrada tenham a mesma forma em todos os eixos, exceto no primeiro (o eixo de concatenação). Para arrays 2D, isso significa que o número de colunas deve ser idêntico. Para arrays 1D, o comprimento deve ser o mesmo. Essa restrição é crucial para garantir a integridade estrutural do array empilhado e evitar ValueErrors.⁴

#### Remodelação Implícita de Arrays 1D

A funcionalidade de vstack de remodelar arrays 1D (N,) para (1, N) antes do empilhamento vertical é uma conveniência notável. Ela simplifica a preparação de dados, permitindo a combinação direta de dados unidimensionais em uma estrutura bidimensional sem a necessidade de etapas de remodelação explícitas. No entanto, é importante que o usuário compreenda que a dimensionalidade dos arrays 1D está sendo aumentada para 2D antes do empilhamento.⁴

#### Não Modifica Arrays Originais

Como muitas funções NumPy, vstack() retorna um novo array e não modifica os arrays de entrada originais.⁴

## numpy.transpose: Permutando Eixos de Arrays

A função numpy.transpose() (ou o método equivalente ndarray.transpose(), ou o atributo conveniente .T) é uma operação fundamental para permutar as dimensões, ou eixos, de um array NumPy. Ela permite reorganizar a estrutura lógica dos dados sem necessariamente copiar os dados subjacentes, tornando-a uma operação eficiente em termos de memória. Para um array 2D, transpose realiza a transposição de matriz padrão, trocando linhas por colunas. Para arrays de N dimensões, ela possibilita uma reordenação arbitrária dos eixos, o que é crucial em diversas aplicações de processamento de dados e álgebra linear.

### Parâmetros de numpy.transpose

- **a**: O array de entrada que se deseja transpor.
- **axes**: Este é um parâmetro opcional que define a nova ordem dos eixos. Sua flexibilidade é o cerne da capacidade de transpose de manipular arrays multidimensionais.

### Comportamento para Arrays 1D e 2D

O comportamento de numpy.transpose() varia ligeiramente dependendo da dimensionalidade do array de entrada:

#### Array 1D

Para um array 1D, numpy.transpose() retorna uma view inalterada do array original.⁵ Isso ocorre porque um vetor 1D possui apenas um eixo, e a operação de transposição, que se refere à permutação de eixos, não tem efeito sobre sua forma. Essa é uma fonte comum de confusão para iniciantes, pois a expectativa de obter um "vetor coluna" de um vetor linha 1D não é atendida diretamente por transpose. Para converter um array 1D em um vetor coluna 2D, é necessário adicionar explicitamente uma dimensão extra, por exemplo, usando `np.atleast_2d(a).T` ou `a[:, np.newaxis]`.

#### Array 2D

Para um array 2D, transpose executa a transposição de matriz padrão, efetivamente trocando as linhas pelas colunas.⁵

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print("Array 2D original:\n", a)
transposed_a = np.transpose(a)
print("\nArray 2D transposto:\n", transposed_a)
```

**Saída:**
```
Array 2D original:
[[1 2]
 [3 4]]

Array 2D transposto:
[[1 3]
 [2 4]]
```

### Compreendendo o Parâmetro axes em numpy.transpose

O parâmetro axes em numpy.transpose (ou ndarray.transpose(*axes)) oferece um controle preciso sobre como os eixos de um array são reordenados. Ele pode assumir algumas formas distintas, cada uma resultando em um efeito específico nas dimensões do array.⁶

#### 1. axes=None ou Sem Argumento

Quando o parâmetro axes é None ou nenhum argumento é fornecido, a ordem dos eixos é invertida. Este é o comportamento padrão da função. Se o array original possui uma forma (d0, d1,..., dn-1), o array transposto resultante terá a forma (dn-1,..., d1, d0).⁵

##### Exemplos:

**Array 2D:**
```python
a = np.array([[1, 2], [3, 4]])
print("Array 2D original:\n", a)
transposed_a = a.transpose()  # Equivalente a a.transpose(None)
print("\nArray 2D transposto (axes=None):\n", transposed_a)
```

**Saída:**
```
Array 2D original:
[[1 2]
 [3 4]]

Array 2D transposto (axes=None):
[[1 3]
 [2 4]]
```

Neste exemplo 2D, os eixos originais são 0 e 1. Invertê-los resulta na ordem (1, 0), que efetivamente troca linhas e colunas, produzindo a transposta da matriz.

**Array 1D:**
```python
a = np.array([1, 2, 3, 4])
print("Array 1D original:\n", a)
print(f"Shape array 1D original:{a.shape}")
transposed_a = a.transpose()
print("\nArray 1D transposto (axes=None):\n", transposed_a)
print(f"Shape array 1D transposto:{transposed_a.shape}")
```

**Saída:**
```
Array 1D original:
[1 2 3 4]

Array 1D transposto (axes=None):
[1 2 3 4]
```

Para um array 1D, inverter a ordem dos eixos não altera nada, pois há apenas um eixo. A forma permanece a mesma (4,).

**Array 3D:**
```python
print(f"Array 1D original:{np.arange(24)}")
print(f"Shape array 1D original:{np.arange(24).shape}")

a = np.arange(24).reshape(2, 3, 4)
print(f"Array 3D original (shape {a.shape}):\n{a}")
transposed_a = a.transpose()
print("\nArray 3D transposto (axes=None, shape {}):\n".format(transposed_a.shape), transposed_a)
```

**Saída:**
```
Array 1D original:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Shape array 1D original:(24,)
Array 3D original (shape (2, 3, 4)):
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

Array 3D transposto (axes=None, shape (4, 3, 2)):
[[[ 0 12]
  [ 4 16]
  [ 8 20]]

 [[ 1 13]
  [ 5 17]
  [ 9 21]]

 [[ 2 14]
  [ 6 18]
  [10 22]]

 [[ 3 15]
  [ 7 19]
  [11 23]]]
```

Os eixos originais são 0, 1, 2. Invertê-los significa que a nova ordem dos eixos será 2, 1, 0. Consequentemente, a forma original (2, 3, 4) torna-se (4, 3, 2) no array transposto.⁶

#### 2. axes=tuple of ints

Esta é a maneira mais flexível e poderosa de especificar a nova ordem dos eixos. A lógica de mapeamento é a seguinte: se um inteiro i está na j-ésima posição da tupla axes, isso significa que o i-ésimo eixo original do array se tornará o j-ésimo eixo do array transposto. A tupla fornecida deve ser uma permutação de 0, 1,..., N-1, onde N é o número de dimensões do array. Compreender essa lógica de mapeamento é fundamental para manipular corretamente arrays multidimensionais complexos, como dados de imagem com canais ou dados de vídeo com múltiplas dimensões, pois um entendimento incorreto pode levar a formas de array inesperadas e erros lógicos em algoritmos.

##### Exemplos:

**Array 2D:**
```python
a = np.array([[1, 2], [3, 4]])
print("Array 2D original:\n", a)
transposed_a = a.transpose((1, 0))
print("\nArray 2D transposto (axes=(1, 0)):\n", transposed_a)
```

**Saída:**
```
Array 2D original:
[[1 2]
 [3 4]]

Array 2D transposto (axes=(1, 0)):
[[1 3]
 [2 4]]
```

Aqui, (1, 0) significa que o eixo original no índice 1 (colunas) se torna o novo eixo no índice 0, e o eixo original no índice 0 (linhas) se torna o novo eixo no índice 1. Isso efetivamente troca linhas e colunas.

**Array 3D:** Usando o mesmo array 3D a com forma (2, 3, 4):
- Eixo 0: Representa os "blocos" ou "camadas" (tamanho 2)
- Eixo 1: Representa as "linhas dentro de cada bloco" (tamanho 3)
- Eixo 2: Representa as "colunas dentro de cada linha" (tamanho 4)

**Exemplo a.transpose((0, 2, 1)):**
```python
a = np.arange(24).reshape(2, 3, 4)
transposed_a = a.transpose((0, 2, 1))
print("\nArray 3D transposto (axes=(0, 2, 1), shape {}):\n".format(transposed_a.shape), transposed_a)
```

**Saída:**
```
Array 3D transposto (axes=(0, 2, 1), shape (2, 4, 3)):
[[[ 0  4  8]
  [ 1  5  9]
  [ 2  6 10]
  [ 3  7 11]]

 [[12 16 20]
  [13 17 21]
  [14 18 22]
  [15 19 23]]]
```

Aqui, (0, 2, 1) significa: o eixo original 0 permanece como o novo eixo 0; o eixo original 2 se torna o novo eixo 1; e o eixo original 1 se torna o novo eixo 2. A nova forma será (tamanho_eixo0_original, tamanho_eixo2_original, tamanho_eixo1_original), ou seja, (2, 4, 3). Isso efetivamente troca a segunda e a terceira dimensões.

**Exemplo a.transpose((2, 0, 1)):**
```python
transposed_a = a.transpose((2, 0, 1))
print("\nArray 3D transposto (axes=(2, 0, 1), shape {}):\n".format(transposed_a.shape), transposed_a)
```

**Saída:**
```
Array 3D transposto (axes=(2, 0, 1), shape (4, 2, 3)):
[[[ 0  4  8]
  [12 16 20]]

 [[ 1  5  9]
  [13 17 21]]

 [[ 2  6 10]
  [14 18 22]]

 [[ 3  7 11]
  [15 19 23]]]
```

Aqui, (2, 0, 1) significa: o eixo original 2 se torna o novo eixo 0; o eixo original 0 se torna o novo eixo 1; e o eixo original 1 se torna o novo eixo 2. A nova forma será (tamanho_eixo2_original, tamanho_eixo0_original, tamanho_eixo1_original), ou seja, (4, 2, 3).⁶

#### 3. axes=n ints

Esta forma é uma alternativa de conveniência à forma de tupla. Em vez de encapsular os inteiros em uma tupla, eles são passados diretamente como argumentos separados. O comportamento é idêntico ao da forma tuple of ints.

##### Exemplo (Array 2D):
```python
a = np.array([[1, 2], [3, 4]])
transposed_a = a.transpose(1, 0)
print("\nArray 2D transposto (axes=1, 0):\n", transposed_a)
```

**Saída:**
```
Array 2D transposto (axes=1, 0):
[[1 3]
 [2 4]]
```

Este código produz o mesmo resultado que `a.transpose((1, 0))`.

A tabela a seguir resume as diferentes formas do parâmetro axes e seus efeitos na ordem dos eixos:

| Formato do axes | Explicação | Efeito na Ordem dos Eixos (Exemplo 3D) | Forma Original -> Transposta |
|-----------------|------------|----------------------------------------|----------------------------|
| None ou Sem Argumento | Inverte a ordem dos eixos | (0, 1, 2) -> (2, 1, 0) | (d0, d1, d2) -> (d2, d1, d0) |
| tuple of ints | Mapeia o i-ésimo eixo original para o j-ésimo novo eixo, onde i está na j-ésima posição da tupla. Deve ser uma permutação de 0...N-1. | (0, 2, 1) significa: orig_0->new_0, orig_2->new_1, orig_1->new_2 | (d0, d1, d2) -> (d0, d2, d1) |
| n ints | Alternativa de conveniência para tuple of ints, passando os inteiros diretamente. | Idêntico a tuple of ints | Idêntico a tuple of ints |

### Valor de Retorno

numpy.transpose retorna um ndarray que é o array de entrada com seus eixos permutados. É importante destacar que uma view do array original é retornada sempre que possível. Isso significa que a operação é geralmente muito eficiente em termos de memória, pois não há cópia de dados; o array transposto simplesmente compartilha os mesmos dados subjacentes com o array original. Consequentemente, qualquer modificação realizada no array transposto também afetará o array original. Se uma cópia independente for necessária, é fundamental usar .copy() explicitamente após a transposição para garantir que os arrays não compartilhem a mesma memória.

### Casos de Uso

numpy.transpose é uma função versátil com diversas aplicações:

- **Álgebra Linear**: A transposição de matrizes é uma operação fundamental em cálculos de álgebra linear, como multiplicação de matrizes e resolução de sistemas lineares.

- **Reordenação de Dimensões em Machine Learning e Processamento de Imagem**: É crucial para ajustar a ordem das dimensões de tensores para se adequar aos requisitos de entrada de modelos de aprendizado de máquina (por exemplo, converter o formato de imagem de (altura, largura, canais) para (canais, altura, largura)).

- **Preparação de Dados**: Reorganizar conjuntos de dados para facilitar operações subsequentes, visualizações ou para compatibilidade com bibliotecas específicas.

### Considerações Importantes

#### View vs. Copy

A natureza de numpy.transpose de retornar uma view sempre que possível é uma otimização de desempenho significativa, pois evita cópias de dados, o que é benéfico para arrays grandes. No entanto, o usuário deve estar ciente de que alterações no array transposto afetarão o original, exigindo cautela ou o uso explícito de .copy() quando a independência é necessária.

#### Comportamento de Array 1D

É crucial lembrar que transpose em um array 1D não altera sua forma. Para obter um vetor coluna 2D de um array 1D, são necessárias operações como `a[:, np.newaxis]` ou `np.atleast_2d(a).T`.

#### Invertendo Transposição

A transposição de tensores pode ser invertida usando a combinação `transpose(a, argsort(axes))`, o que é útil para desfazer uma permutação específica e retornar à ordem original dos eixos.

#### Funções Relacionadas

O NumPy oferece outras funções relacionadas que podem ser úteis para manipulação de eixos:

- **ndarray.T**: Um atributo conveniente diretamente disponível em objetos ndarray que é equivalente a chamar ndarray.transpose() sem argumentos, revertendo a ordem dos eixos.

- **numpy.moveaxis**: Permite mover um ou mais eixos de um array para novas posições, o que pode ser mais intuitivo para certas reordenações complexas.

- **numpy.swapaxes**: Troca dois eixos específicos de um array.

## Plotagem de Gráficos e Imagens com Matplotlib no Google Colab

A visualização de dados é uma etapa fundamental em qualquer processo de análise, permitindo a compreensão de padrões, tendências e anomalias. No campo da ciência de dados e visão computacional, ferramentas como Matplotlib e OpenCV, em conjunto com NumPy, são indispensáveis para transformar dados brutos em representações visuais significativas e para manipular imagens de forma programática. Este relatório serve como um complemento aprofundado ao Módulo 1, focando nas capacidades do Matplotlib para plotagem de gráficos e exibição de imagens, integrando-o com as poderosas funcionalidades de processamento de imagem do OpenCV e as operações de array do NumPy. Todos os exemplos de código são projetados para serem executáveis no ambiente Google Colab, garantindo praticidade e acessibilidade para o aprendizado e a experimentação.

### Configuração do Ambiente e Fundamentos do Matplotlib

Para iniciar a jornada de visualização de dados no Google Colab, é crucial configurar o ambiente corretamente e compreender a estrutura fundamental do Matplotlib.

#### Configuração Essencial para Google Colab

A primeira linha de código em qualquer notebook Colab que envolva plotagem com Matplotlib deve ser `%matplotlib inline`. Esta "magic command" instrui o ambiente a renderizar os gráficos diretamente no notebook, logo abaixo da célula de código que os gera, em vez de abri-los em janelas separadas. Além disso, a importação das bibliotecas necessárias é um passo inicial indispensável.

`matplotlib.pyplot` é a interface de plotagem principal, `numpy` é essencial para manipulação numérica de dados e imagens, `PIL.Image` (Pillow) é comumente usada para carregar imagens, e `cv2` (OpenCV) é a biblioteca padrão para operações de visão computacional.

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

O Matplotlib adota uma estrutura hierárquica para a construção de gráficos, que é fundamental para um controle preciso sobre a visualização. No topo dessa hierarquia está a **Figure**, que pode ser concebida como a "tela" ou "janela" principal onde o gráfico será desenhado. Dentro de uma Figure, um ou mais objetos **Axes** são definidos. Um Axes representa a área de plotagem individual, contendo os elementos visuais do gráfico, como linhas, pontos, barras, eixos X e Y, rótulos e um título.

É importante notar a distinção entre **Axes** (a área de plotagem completa) e **Axis** (os eixos individuais, como o eixo X ou Y, que controlam ticks e rótulos).

A maneira mais recomendada para iniciar a criação de um gráfico é utilizando `plt.subplots()`. Esta função retorna uma tupla contendo a Figure e um ou mais objetos Axes, simplificando a configuração inicial.

```python
fig, ax = plt.subplots(figsize=(8, 6))  # Cria uma Figure e um único Axes
ax.set_title("Título do Gráfico")
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
plt.show()  # Exibe o gráfico
```

#### Abordagem Orientada a Objetos para Maior Controle e Previsibilidade

O Matplotlib oferece duas interfaces principais para a criação de gráficos: a interface baseada em estado (matplotlib.pyplot) e a interface orientada a objetos (OO). A interface pyplot, que se assemelha ao MATLAB, opera em um estado global implícito, onde funções como `plt.plot()` ou `plt.title()` atuam sobre a Figure e o Axes "atuais". Embora conveniente para scripts rápidos, essa abordagem pode levar a comportamentos imprevisíveis em ambientes como o Google Colab, onde as células são executadas sequencialmente e o estado global pode ser modificado de forma não óbvia entre as execuções de células.

Em contraste, a interface orientada a objetos oferece um controle explícito sobre os elementos do gráfico. Ao usar `fig, ax = plt.subplots()` e, subsequentemente, chamar métodos diretamente nos objetos ax (como `ax.plot()`, `ax.set_title()`, `ax.set_xlabel()`), o desenvolvedor garante que as modificações se apliquem ao Axes específico que se pretende. Esta prática resulta em um código mais robusto, legível e previsível, especialmente benéfico em notebooks interativos como o Colab, onde a clareza sobre qual objeto está sendo manipulado é crucial para evitar resultados inesperados.

### Plotagem de Gráficos Fundamentais com Matplotlib

Matplotlib oferece uma vasta gama de tipos de gráficos para diversas necessidades de visualização de dados. Os gráficos de linha, dispersão, barras e histogramas são pilares para a análise exploratória e apresentação de dados.

#### Gráficos de Linha (plt.plot)

A função `plt.plot()` é a ferramenta fundamental para a criação de gráficos de linha, que são ideais para exibir tendências ao longo do tempo ou em relação a uma variável contínua. Ela aceita dados para os eixos X e Y, permitindo uma representação flexível. No uso mais básico, pode-se fornecer apenas os valores Y, e o Matplotlib gerará automaticamente os valores X como uma sequência de inteiros (0, 1, 2...). No entanto, é comum fornecer explicitamente tanto os valores X quanto Y para maior controle.

A personalização é um ponto forte do `plt.plot()`. É possível estilizar as linhas usando um terceiro argumento opcional, uma "string de formato", que combina informações de cor, estilo de linha e marcador (por exemplo, 'r--' para uma linha tracejada vermelha). Alternativamente, argumentos de palavra-chave como `color`, `linestyle` (ls), `linewidth` (lw), `marker`, `markersize` e `label` oferecem um controle mais granular sobre cada aspecto visual da linha.

##### Formatos Comuns para plt.plot

| Formato | Descrição da Cor | Descrição do Estilo de Linha | Descrição do Marcador | Exemplo |
|---------|------------------|------------------------------|----------------------|---------|
| b- | Azul | Linha Sólida | Nenhum | `plt.plot(x, y, 'b-')` |
| r-- | Vermelho | Linha Tracejada | Nenhum | `plt.plot(x, y, 'r--')` |
| g^ | Verde | Nenhum | Triângulos | `plt.plot(x, y, 'g^')` |
| ks: | Preto | Linha Pontilhada | Quadrados | `plt.plot(x, y, 'ks:')` |
| c-. | Ciano | Linha Traço-Ponto | Nenhum | `plt.plot(x, y, 'c-.')` |

A adição de elementos textuais é crucial para a interpretabilidade do gráfico. O título do gráfico é definido com `ax.set_title()`, os rótulos dos eixos X e Y com `ax.set_xlabel()` e `ax.set_ylabel()`, respectivamente. Para identificar múltiplas linhas em um gráfico, a função `ax.legend()` exibe uma legenda, utilizando os rótulos fornecidos no parâmetro label de cada chamada `ax.plot()`. Finalmente, `ax.grid(True)` adiciona uma grade ao fundo do gráfico, auxiliando na leitura dos valores.

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

Os gráficos de dispersão, criados com `plt.scatter()`, são ferramentas excelentes para visualizar a relação entre duas variáveis numéricas, onde cada ponto no gráfico representa uma única observação. Diferentemente de `plt.plot`, que conecta pontos com linhas, `plt.scatter` plota pontos individuais, o que é ideal para identificar agrupamentos, tendências ou valores atípicos em um conjunto de dados.

Uma das grandes vantagens de `plt.scatter` é a sua capacidade de mapear dimensões adicionais de dados para as propriedades visuais dos pontos. Os parâmetros `s` (size) e `c` (color) permitem que uma terceira e quarta variável sejam representadas, respectivamente, pelo tamanho e cor dos marcadores. Isso permite a visualização efetiva de até quatro dimensões de dados (X, Y, Tamanho, Cor) em um único gráfico 2D. Esta capacidade é fundamental na análise exploratória de dados, pois possibilita a identificação de clusters, outliers ou padrões complexos que não seriam evidentes ao observar apenas as duas dimensões principais. A escolha de um `cmap` (colormap) apropriado é crucial aqui, pois a percepção humana de gradientes de cor pode influenciar significativamente a interpretação dos dados.

O parâmetro `data` é outra funcionalidade útil, permitindo que se passe um dicionário ou um objeto similar a um DataFrame e referencie as colunas por nome, o que aumenta a legibilidade do código.

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

#### Gráficos de Barras (plt.bar)

Os gráficos de barras, gerados com `plt.bar()`, são ideais para a representação de dados categóricos, onde o comprimento de cada barra é diretamente proporcional ao valor que ela representa. Os parâmetros mais básicos são `x` para as categorias (rótulos no eixo X) e `height` para os valores correspondentes (altura das barras).

A personalização dos gráficos de barras é bastante flexível. O parâmetro `width` permite controlar a largura das barras (o padrão é 0.8). A cor das barras pode ser definida usando o parâmetro `color`, que aceita uma única cor, uma lista de cores (aplicadas recursivamente a cada barra) ou um dicionário que mapeia nomes de colunas para cores. Uma funcionalidade importante é o parâmetro `bottom`, que possibilita a criação de gráficos de barras empilhadas. Neste caso, a base de uma barra é definida pelo topo da barra anterior, permitindo visualizar a composição de diferentes componentes dentro de cada categoria. O parâmetro `align` controla como as barras são alinhadas em relação aos rótulos do eixo X, com opções como 'center' ou 'edge'. Para gráficos de barras horizontais, a função `plt.barh()` é utilizada, operando de forma similar a `plt.bar()` mas exibindo as barras na horizontal.

Embora os gráficos de barras sejam intuitivos para comparações diretas entre categorias, a capacidade de empilhar barras com o parâmetro `bottom` expande sua utilidade para a visualização de "partes de um todo" ou a composição de diferentes componentes dentro de cada categoria. Isso transforma um gráfico de comparação simples em uma ferramenta para entender a estrutura interna de cada grupo. A escolha entre barras verticais e horizontais também não é meramente estética; barras horizontais são frequentemente mais eficazes para exibir rótulos de categorias longos, pois oferecem mais espaço para o texto.

```python
# Dados de exemplo
categorias = ['A', 'B', 'C', 'D']
valores1 = np.array([20, 35, 30, 25])
valores2 = np.array([15, 20, 10, 30])

# Gráfico de Barras Simples
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categorias, valores1, color='skyblue', width=0.6, label='Série 1')
ax.set_title('Vendas por Categoria', fontsize=16)
ax.set_xlabel('Categoria', fontsize=12)
ax.set_ylabel('Quantidade Vendida', fontsize=12)
ax.legend()
plt.show()

# Gráfico de Barras Empilhadas
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categorias, valores1, color='lightcoral', label='Série 1')
ax.bar(categorias, valores2, bottom=valores1, color='lightgreen', label='Série 2')  # Empilha a Série 2
ax.set_title('Vendas Empilhadas por Categoria', fontsize=16)
ax.set_xlabel('Categoria', fontsize=12)
ax.set_ylabel('Quantidade Total', fontsize=12)
ax.legend()
plt.show()

# Gráfico de Barras Horizontal
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(categorias, valores1, color='gold', label='Série 1')
ax.set_title('Vendas Horizontais por Categoria', fontsize=16)
ax.set_xlabel('Quantidade Vendida', fontsize=12)
ax.set_ylabel('Categoria', fontsize=12)
ax.legend()
plt.show()
```

#### Histogramas (plt.hist)

Histogramas são ferramentas estatísticas cruciais para a visualização da distribuição de dados numéricos. Eles funcionam dividindo o intervalo de valores dos dados em uma série de "bins" (intervalos não sobrepostos) e, em seguida, contando a frequência ou a densidade de valores que caem em cada bin, representando essas contagens como barras.

Para a análise da distribuição de dados unidimensionais (1D), a função `plt.hist()` é a escolha apropriada. Para conjuntos de dados bidimensionais (2D), onde se busca visualizar a frequência conjunta de duas variáveis, `plt.hist2d()` é a função utilizada, que gera um mapa de calor representando a densidade de pontos nas regiões do plano.

Os parâmetros chave para `plt.hist()` incluem:

- **x**: Os dados de entrada para os quais o histograma será construído.
- **bins**: Pode ser um inteiro que especifica o número de bins, uma sequência de valores que define as bordas dos bins, ou a string 'auto' para que o Matplotlib determine automaticamente o número ideal de bins.
- **density**: Se definido como True, o histograma é normalizado para formar uma estimativa de densidade de probabilidade, onde a área total sob o histograma soma 1. Isso é fundamental para comparar a forma de distribuições com diferentes números de pontos de dados, pois a área total sob cada histograma se torna 1, permitindo uma comparação justa das formas das distribuições. Sem `density=True`, um conjunto de dados maior sempre pareceria ter barras mais altas, obscurecendo a verdadeira forma da distribuição.
- **histtype**: Define o estilo visual do histograma, com opções como 'bar' (barras tradicionais), 'barstacked' (barras empilhadas para múltiplos datasets), 'step' (apenas contornos) ou 'stepfilled' (contornos preenchidos).
- **range**: Um tupla que especifica o limite inferior e superior do intervalo dos bins. Valores fora deste intervalo são ignorados.

A personalização visual é possível através do acesso aos "patches" (as barras individuais) que `plt.hist` retorna. Isso permite modificar cores individualmente, por exemplo, colorindo as barras com base em sua altura. Adicionalmente, `PercentFormatter` pode ser empregado para formatar o eixo Y como porcentagens quando `density=True`, facilitando a interpretação das proporções. Para histogramas 2D, a escolha do `cmap` (colormap) e a inclusão de uma colorbar são cruciais para interpretar a "densidade" de pontos em regiões bidimensionais, revelando correlações ou agrupamentos nos dados.

```python
# Dados de exemplo (distribuição normal)
np.random.seed(19680801)
data1 = np.random.normal(loc=0, scale=1, size=10000)
data2 = np.random.normal(loc=5, scale=0.8, size=10000)

# Histograma 1D Simples
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data1, bins=30, color='teal', alpha=0.7, label='Distribuição 1')
ax.set_title('Histograma de Dados 1D', fontsize=16)
ax.set_xlabel('Valor', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
ax.legend()
plt.show()

# Histograma 1D com Densidade e Múltiplos Datasets
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data1, bins=30, density=True, alpha=0.6, color='blue', label='Distribuição 1 (Densidade)')
ax.hist(data2, bins=30, density=True, alpha=0.6, color='red', label='Distribuição 2 (Densidade)')
ax.set_title('Histograma de Densidade para Múltiplos Datasets', fontsize=16)
ax.set_xlabel('Valor', fontsize=12)
ax.set_ylabel('Densidade de Probabilidade', fontsize=12)
ax.legend()
plt.show()

# Histograma 2D
fig, ax = plt.subplots(figsize=(8, 7))
hist_2d = ax.hist2d(data1, data2, bins=40, cmap='Blues')
fig.colorbar(hist_2d[3], ax=ax, label='Contagem de Pontos')  # hist_2d[3] é o objeto mappable
ax.set_title('Histograma 2D', fontsize=16)
ax.set_xlabel('Variável X', fontsize=12)
ax.set_ylabel('Variável Y', fontsize=12)
plt.show()
```

### Visualização e Processamento de Imagens com Matplotlib e OpenCV

A manipulação e visualização de imagens são tarefas centrais em visão computacional e análise de dados visuais. Matplotlib, com sua função imshow, e OpenCV, com suas capacidades de processamento, formam uma dupla poderosa para essas finalidades.

#### Exibindo Imagens (plt.imshow)

A função `plt.imshow()` é a principal ferramenta do Matplotlib para exibir dados como uma imagem em uma grade 2D regular. Ela é versátil, capaz de exibir tanto dados RGB(A) (imagens coloridas com ou sem transparência) quanto dados escalares 2D (imagens em escala de cinza ou mapas de calor), que são mapeados para cores usando um colormap.

O carregamento de dados de imagem é geralmente feito convertendo-os para arrays NumPy. A biblioteca Pillow (PIL.Image) é comumente utilizada para abrir arquivos de imagem, e `np.asarray()` para converter o objeto PIL.Image.Image em um array NumPy. Para exemplos no Google Colab, imagens podem ser carregadas diretamente de URLs usando `requests` e `BytesIO` ou através do upload de arquivos.

É fundamental compreender os formatos dos arrays NumPy que `imshow()` espera para representar diferentes tipos de imagens:

- **(M, N)**: Representa uma imagem em escala de cinza ou dados escalares 2D. Os valores são mapeados para cores usando um colormap (cmap) e normalização.
- **(M, N, 3)**: Representa uma imagem RGB (Vermelho, Verde, Azul). Os valores dos pixels podem ser floats entre 0 e 1 ou inteiros entre 0 e 255.
- **(M, N, 4)**: Representa uma imagem RGBA (Vermelho, Verde, Azul, Alpha), incluindo um canal de transparência. Os valores também podem ser floats (0-1) ou inteiros (0-255).

As duas primeiras dimensões, M e N, definem, respectivamente, o número de linhas (altura) e colunas (largura) da imagem.

##### Formatos de Array para plt.imshow

| Shape do Array X | Descrição | Exemplo de Dados | Casos de Uso |
|------------------|-----------|------------------|--------------|
| (M, N) | Imagem em escala de cinza ou dados escalares 2D. Os valores são mapeados para cores via cmap. | `np.array([[1, 2], [3, 4]])` | Mapas de calor, dados de sensor, imagens médicas, imagens monocromáticas. |
| (M, N, 3) | Imagem RGB (Vermelho, Verde, Azul). Canais de cor explícitos. | `np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]])` | Fotografias coloridas, saídas de modelos de visão computacional. |
| (M, N, 4) | Imagem RGBA (Vermelho, Verde, Azul, Alpha). Inclui canal de transparência. | `np.array([[[255, 0, 0, 255], [0, 255, 0, 128]], [[0, 0, 255, 255], [255, 255, 0, 64]]])` | Imagens com transparência (ex: PNGs), sobreposição de camadas. |

Os parâmetros essenciais para imshow permitem um controle significativo sobre a exibição da imagem:

- **X**: Os dados da imagem, como um array NumPy ou uma imagem PIL.
- **cmap**: O colormap, usado para mapear dados escalares para cores (por exemplo, 'gray' para escala de cinza, 'viridis' para um gradiente de cores perceptualmente uniforme, 'hot' para um mapa de calor). Este parâmetro é ignorado para imagens RGB(A).
- **origin**: Define se o índice `[0, 0]` do array corresponde ao canto superior esquerdo ('upper', padrão para imagens e matrizes) ou inferior esquerdo ('lower', padrão para gráficos cartesianos). Isso afeta a orientação vertical da imagem.
- **aspect**: Controla a proporção do Axes. 'equal' garante que os pixels sejam quadrados, enquanto 'auto' ajusta o aspecto para preencher o Axes, o que pode resultar em pixels não quadrados.
- **interpolation**: O método de interpolação usado para redimensionar a imagem ao exibi-la (por exemplo, 'nearest' para pixels nítidos, 'bilinear' ou 'bicubic' para suavização).
- **vmin, vmax**: Para dados escalares, esses parâmetros definem o intervalo de dados que o colormap cobrirá, sendo úteis para ajustar o contraste da imagem.

Para interpretar o mapeamento de cores em imagens de dados escalares, `plt.colorbar()` é uma adição crucial, fornecendo uma referência visual da escala de valores.

##### Parâmetros Chave para plt.imshow

| Parâmetro | Descrição | Valores Comuns/Efeito | Implicação |
|-----------|-----------|----------------------|------------|
| X | Dados da imagem. | Array NumPy (M,N), (M,N,3), (M,N,4). | Entrada fundamental. |
| cmap | Colormap para dados escalares. | 'gray', 'viridis', 'hot', 'jet'. | Define a paleta de cores para imagens em escala de cinza ou dados numéricos. |
| origin | Posição do índice `[0, 0]`. | 'upper' (padrão para imagens), 'lower' (padrão para gráficos). | Afeta a orientação vertical da imagem. |
| aspect | Proporção do Axes. | 'equal' (pixels quadrados), 'auto' (preenche o Axes). | Garante que os pixels sejam exibidos corretamente ou que a imagem preencha a área. |
| interpolation | Método de interpolação. | 'nearest', 'bilinear', 'bicubic'. | Suaviza ou pixeliza a imagem ao redimensionar. |
| vmin, vmax | Limites de dados para o colormap. | Números floats. | Ajusta o contraste e a faixa de valores visíveis para dados escalares. |

Embora imshow seja comumente associado à exibição de fotografias, sua capacidade de mapear dados escalares (arrays com shape (M, N)) para cores através de cmap, vmin e vmax a transforma em uma ferramenta poderosa para visualizar qualquer matriz numérica como um mapa de calor ou imagem pseudocolorizada. Isso é extremamente útil em campos como ciência de dados para visualizar matrizes de correlação, mapas de ativação de redes neurais ou dados de densidade. Os parâmetros origin e aspect são cruciais para garantir que a interpretação visual (seja como uma matriz matemática ou uma imagem fotográfica) esteja correta.

```python
# Para carregar uma imagem de uma URL no Colab:
import requests
from io import BytesIO

# URL de uma imagem de exemplo (substitua por sua própria imagem se desejar)
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

# Ajustando contraste com vmin/vmax
fig, ax = plt.subplots(figsize=(6, 6))
im_contrast = ax.imshow(lum_img, cmap='gray', vmin=50, vmax=200)
fig.colorbar(im_contrast, ax=ax, label='Intensidade do Pixel (Contraste Ajustado)')
ax.set_title('Imagem em Escala de Cinza (Contraste Ajustado)', fontsize=16)
ax.axis('off')
plt.show()
```

#### Manipulação Básica de Imagens para Plotagem (NumPy e OpenCV)

Para preparar dados para plotagem ou para realizar tarefas complexas de visão computacional, NumPy e OpenCV são bibliotecas indispensáveis. Enquanto Matplotlib atua como a camada de visualização, NumPy e OpenCV fornecem as ferramentas robustas para a manipulação e processamento de dados e imagens.

##### Manipulação de Arrays NumPy

NumPy é a base para a computação numérica em Python, e suas funções de manipulação de arrays são cruciais para o pré-processamento de dados de imagem.

- **numpy.reshape()**: Esta função permite alterar a forma (dimensões) de um array sem modificar seus dados subjacentes. É extremamente útil para transformar arrays 1D em 2D (ou vice-versa) ou para reorganizar as dimensões de um array para que se ajustem aos requisitos de uma função específica.
- **numpy.transpose()**: Utilizada para permutar os eixos de um array. Para um array 2D, ela executa a transposição de matriz padrão, trocando linhas por colunas. Para arrays de N dimensões, o parâmetro axes permite especificar a nova ordem dos eixos, oferecendo controle preciso sobre a reorganização dos dados.
- **numpy.stack() vs. numpy.vstack() / numpy.hstack()**:
  - **numpy.stack()**: Esta função une uma sequência de arrays ao longo de um novo eixo, o que significa que ela aumenta a dimensão do array resultante. Todos os arrays de entrada devem ter a mesma forma.
  - **numpy.vstack()**: Empilha arrays verticalmente, linha por linha, ao longo do primeiro eixo. É funcionalmente equivalente a stack(axis=0).
  - **numpy.hstack()**: Empilha arrays horizontalmente, coluna por coluna, ao longo do segundo eixo (ou do primeiro para arrays 1D).

É fundamental compreender que muitas operações NumPy, como reshape e transpose, frequentemente retornam uma view do array original, em vez de uma cópia independente. Isso significa que nenhuma nova memória é alocada, o que é altamente eficiente para grandes conjuntos de dados. No entanto, uma consequência direta é que qualquer modificação feita na view também afetará o array original. Para garantir que uma operação crie uma cópia de dados completamente independente, `array.copy()` deve ser explicitamente chamado. Esta distinção é vital para evitar efeitos colaterais inesperados em pipelines de dados complexos e para otimizar o desempenho em grandes datasets.

```python
# Exemplo de reshape
arr_1d = np.arange(12)
print(f"Array 1D: {arr_1d}, Shape: {arr_1d.shape}")
arr_2d = arr_1d.reshape((3, 4))
print(f"Array 2D (reshape): \n{arr_2d}, Shape: {arr_2d.shape}")

# Exemplo de transpose
arr_transposed = arr_2d.transpose()
print(f"Array 2D Transposed: \n{arr_transposed}, Shape: {arr_transposed.shape}")

# Exemplo de stack (cria nova dimensão)
a = np.array([2, 3, 1])
b = np.array([4, 5, 6])
stacked_arr = np.stack((a, b))
print(f"Array Stacked (nova dimensão): \n{stacked_arr}, Shape: {stacked_arr.shape}")

# Exemplo de vstack (empilha verticalmente)
x = np.array([[2, 3], [1, 4]])
y = np.array([[5, 6], [7, 8]])
vstack_arr = np.vstack((x, y))
print(f"Array vstacked: \n{vstack_arr}, Shape: {vstack_arr.shape}")

# Exemplo de hstack (empilha horizontalmente)
hstack_arr = np.hstack((x, y))
print(f"Array hstacked: \n{hstack_arr}, Shape: {hstack_arr.shape}")
```

##### Funções NumPy para Manipulação de Arrays (Pré-processamento)

| Função NumPy | Propósito | Parâmetros Chave | Comportamento (View/Copy) |
|--------------|-----------|------------------|---------------------------|
| reshape() | Altera as dimensões de um array sem alterar os dados. | shape (tupla de ints, -1 para inferir). | Retorna uma view se possível, senão uma cópia. |
| transpose() | Permuta os eixos de um array. | axes (tupla de ints para ordem dos eixos). | Retorna uma view se possível, senão uma cópia. |
| stack() | Une arrays ao longo de um novo eixo. | axis (onde o novo eixo é inserido). | Retorna uma cópia. |
| vstack() | Empilha arrays verticalmente (linha por linha). | tup (sequência de arrays). | Retorna uma cópia. |
| hstack() | Empilha arrays horizontalmente (coluna por coluna). | tup (sequência de arrays). | Retorna uma cópia. |

##### Funções OpenCV para Processamento de Imagens

OpenCV (cv2) é uma biblioteca de código aberto amplamente utilizada para tarefas de visão computacional e processamento de imagens. Suas funções são otimizadas para desempenho e operam nativamente em arrays NumPy, o que facilita uma integração fluida com o Matplotlib para visualização.

- **cv2.imread() e cv2.imshow()**:
  - **cv2.imread(filepath)**: Carrega uma imagem de um arquivo para um array NumPy. Por padrão, o OpenCV lê imagens no formato BGR (Azul, Verde, Vermelho), que difere do formato RGB (Vermelho, Verde, Azul) comumente usado por Matplotlib e outras bibliotecas.
  - **cv2.imshow(window_name, image_array)**: Esta função é usada para exibir uma imagem em uma janela dedicada do OpenCV. No entanto, é crucial notar que `cv2.imshow()` não funciona diretamente no Google Colab de forma interativa. O Colab é um ambiente baseado em navegador e não possui uma interface gráfica de usuário (GUI) para renderizar janelas do OpenCV. Para visualizar imagens processadas com OpenCV no Colab, é necessário converter a imagem para o formato RGB (se for BGR) e então usar `plt.imshow()` do Matplotlib.

- **cv2.calcHist()**: Calcula o histograma de uma imagem, fornecendo uma representação da distribuição de intensidade dos pixels ou dos canais de cor. Os parâmetros chave incluem `images` (uma lista de arrays de imagem), `channels` (os índices dos canais a serem analisados, e.g., `[0]` para azul, `[1]` para verde, `[2]` para vermelho em BGR), `mask` (uma máscara opcional para calcular o histograma apenas em uma região específica), `histSize` (o número de bins para cada dimensão do histograma) e `ranges` (o intervalo de valores de pixel a serem considerados). O resultado de `cv2.calcHist()` é um array NumPy que pode ser facilmente plotado com `plt.plot()` para visualização.

- **cv2.equalizeHist()**: Esta função é utilizada para melhorar o contraste de uma imagem em escala de cinza. Ela opera esticando o histograma de intensidade da imagem para cobrir toda a faixa dinâmica disponível, resultando em uma imagem com contraste aprimorado. A função aceita uma imagem de entrada (que deve estar em escala de cinza) e retorna a imagem com o histograma equalizado.

- **Rotação de Imagens (cv2.getRotationMatrix2D e cv2.warpAffine)**:
  - **cv2.getRotationMatrix2D(center, angle, scale)**: Calcula uma matriz de transformação 2x3 que pode ser usada para rotação, escala e translação de uma imagem. O `angle` é especificado em graus, com valores positivos indicando rotação anti-horária.
  - **cv2.warpAffine(src, M, dsize)**: Aplica a transformação afim definida pela matriz M (obtida de getRotationMatrix2D ou outras funções) a uma imagem de origem src, produzindo uma imagem de destino com o tamanho especificado por dsize.

A principal consideração ao trabalhar com OpenCV no Google Colab é a sinergia entre as bibliotecas e as limitações do ambiente. Embora o OpenCV seja a ferramenta de escolha para processar imagens (leitura, manipulação, aplicação de filtros), o Matplotlib é a ferramenta ideal para visualizar essas imagens no ambiente baseado em navegador do Colab. A saída de quase todas as funções do OpenCV é um array NumPy, que é o formato de entrada nativo para `plt.imshow()`. Isso estabelece um pipeline de trabalho muito eficiente: `cv2.imread` -> `cv2.processamento` -> `plt.imshow`. A necessidade de converter imagens de BGR para RGB para exibição com Matplotlib e a impossibilidade de usar `cv2.imshow` diretamente no Colab são pontos cruciais que devem ser compreendidos para um fluxo de trabalho eficaz.

```python
# Exemplo de carregamento, histograma e equalização de imagem
# Baixar uma imagem de exemplo (se não tiver uma localmente)
!wget -O sample_image.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Clouds_over_the_Atlantic_Ocean.jpg/640px-Clouds_over_the_Atlantic_Ocean.jpg

# Carregar imagem com OpenCV (lê em BGR por padrão)
img_bgr = cv2.imread('sample_image.jpg')
if img_bgr is None:
    print("Erro: Imagem não encontrada. Verifique o caminho.")
else:
    # Converter de BGR para RGB para exibição com Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Converter para escala de cinza para equalização de histograma
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Equalizar histograma
    img_equalized = cv2.equalizeHist(img_gray)
    
    # Calcular histograma do canal de luminância
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # Plotar imagens e histograma
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title('Escala de Cinza')
    axes[1].axis('off')
    
    axes[2].imshow(img_equalized, cmap='gray')
    axes[2].set_title('Histograma Equalizado')
    axes[2].axis('off')
    plt.show()
    
    # Plotar o histograma calculado
    plt.figure(figsize=(8, 5))
    plt.plot(hist_gray, color='black')
    plt.title('Histograma da Imagem em Escala de Cinza')
    plt.xlabel('Intensidade do Pixel')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    # Exemplo de Rotação de Imagem
    if img_bgr is not None:
        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        angle = 45  # Rotação de 45 graus
        scale = 1.0  # Sem escala
        
        # Obter a matriz de rotação
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Aplicar a rotação
        rotated_img_bgr = cv2.warpAffine(img_bgr, M, (w, h))
        rotated_img_rgb = cv2.cvtColor(rotated_img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(rotated_img_rgb)
        plt.title(f'Imagem Rotacionada {angle} Graus')
        plt.axis('off')
        plt.show()
```

##### Funções OpenCV para Processamento de Imagens (Pré-processamento)

| Função OpenCV | Propósito | Parâmetros Chave | Saída (Formato) | Compatibilidade Colab |
|---------------|-----------|------------------|-----------------|----------------------|
| imread() | Carrega uma imagem de arquivo. | filename, flags (ex: cv2.IMREAD_COLOR). | numpy.ndarray (BGR). | Sim (para leitura). |
| imshow() | Exibe uma imagem em uma janela. | window_name, image_array. | None. | Não (use plt.imshow no Colab). |
| calcHist() | Calcula o histograma de uma imagem. | images, channels, mask, histSize, ranges. | numpy.ndarray (float32). | Sim (o resultado pode ser plotado com Matplotlib). |
| equalizeHist() | Equaliza o histograma de uma imagem em escala de cinza. | src (imagem de entrada), dst (imagem de saída). | numpy.ndarray (escala de cinza). | Sim (o resultado pode ser plotado com Matplotlib). |
| getRotationMatrix2D() | Calcula a matriz de rotação 2D. | center, angle, scale. | numpy.ndarray (matriz 2x3). | Sim. |
| warpAffine() | Aplica uma transformação afim a uma imagem. | src, M (matriz de transformação), dsize (tamanho de saída). | numpy.ndarray. | Sim (o resultado pode ser plotado com Matplotlib). |

#### Interatividade e Salvamento de Plots

O modelo de execução do Google Colab com `%matplotlib inline` implica que os gráficos são renderizados como imagens estáticas no momento em que a célula é executada. Diferentemente de um ambiente Python interativo local que pode abrir uma janela de gráfico dinamicamente atualizável, no Colab, cada célula de plotagem é um "instantâneo" do estado do gráfico. Isso significa que comandos de plotagem em células subsequentes não afetarão gráficos que já foram exibidos em células anteriores. Para modificar um gráfico, todas as alterações e chamadas de plotagem devem ser realizadas dentro da mesma célula onde o gráfico foi criado. Esta característica do ambiente incentiva a encapsulação de um gráfico completo em uma única célula para maior clareza e previsibilidade.

Para salvar os gráficos gerados, as funções `plt.savefig('nome_do_arquivo.png')` ou `fig.savefig('nome_do_arquivo.pdf')` podem ser utilizadas. Os arquivos serão salvos no sistema de arquivos temporário do ambiente Colab e podem ser baixados posteriormente para uso local.

```python
# Exemplo de salvamento de plot
x = np.linspace(0, 5, 100)
y = x**2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label='y = x^2')
ax.set_title('Gráfico para Salvar')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.grid(True)

# Salvar o gráfico como PNG
plt.savefig('meu_primeiro_grafico.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'meu_primeiro_grafico.png'")
plt.show()  # Exibir o gráfico após salvar
```


## Referências citadas

1. AlexNet and ImageNet: The Birth of Deep Learning - Pinecone, acessado em agosto 12, 2025, https://www.pinecone.io/learn/series/image-search/imagenet/
2. What is LeNet? - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/computer-vision/what-is-lenet/
3. Remote Sensing Image Change Detection Based on Deep Learning: Multi-Level Feature Cross-Fusion with 3D-Convolutional Neural Networks - MDPI, acessado em agosto 12, 2025, https://www.mdpi.com/2076-3417/14/14/6269
4. Enhanced hybrid CNN and transformer network for remote sensing image change detection, acessado em agosto 12, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11933460/
5. (PDF) Change detection from remotely sensed images: From pixel ..., acessado em agosto 12, 2025, https://www.researchgate.net/publication/258791369_Change_detection_from_remotely_sensed_images_From_pixel-based_to_object-based_approaches
6. NumPy: the absolute basics for beginners, acessado em agosto 12, 2025, https://numpy.org/doc/stable/user/absolute_beginners.html
7. Scalars, Vectors, Matrices and Tensors - Linear Algebra for Deep ..., acessado em agosto 12, 2025, https://www.quantstart.com/articles/scalars-vectors-matrices-and-tensors-linear-algebra-for-deep-learning-part-1/
8. 2.1. Data Manipulation — Dive into Deep Learning 1.0.3 documentation, acessado em agosto 12, 2025, https://d2l.ai/chapter_preliminaries/ndarray.html
9. Computation on Arrays: Broadcasting | Python Data Science Handbook, acessado em agosto 12, 2025, https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
10. Learn Batches | Tensors - Codefinity, acessado em agosto 12, 2025, https://codefinity.com/courses/v2/a668a7b9-f71f-420f-89f1-71ea7e5abbac/72783c6a-1699-4821-b1b0-824cef4ef6b3/aca9c244-8471-4273-9812-d330de1a9cf2
11. numpy.matrix() in Python. I understand that learning data science ..., acessado em agosto 12, 2025, https://medium.com/@amit25173/numpy-matrix-in-python-5896e1cacf3e
12. Numpy matrix.transpose() - Python - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/python/python-numpy-matrix-transpose/
13. Perceptron - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/Perceptron
14. AI Winter and Funding Challenges - The ARF - Advertising Research Foundation, acessado em agosto 12, 2025, https://thearf.org/ai-handbook/ai-winter-and-funding-challenges/
15. The Cycles of AI Winters: A Historical Analysis and Modern Perspective | by Ferhat Sarikaya, acessado em agosto 12, 2025, https://medium.com/@ferhatsarikaya/the-cycles-of-ai-winters-a-historical-analysis-and-modern-perspective-776ffadd2025
16. How the backpropagation algorithm works - Neural networks and ..., acessado em agosto 12, 2025, http://neuralnetworksanddeeplearning.com/chap2.html
17. AlexNet and ImageNet Explained - YouTube, acessado em agosto 12, 2025, https://www.youtube.com/watch?v=c_u4AHNjOpk&pp=0gcJCfwAo7VqN5tD
18. ImageNet Dataset - Ultralytics YOLO Docs, acessado em agosto 12, 2025, https://docs.ultralytics.com/datasets/classify/imagenet/
19. Understanding ImageNet: A Key Resource for Computer Vision and AI Research - Zilliz, acessado em agosto 12, 2025, https://zilliz.com/glossary/imagenet
20. LeNet - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/LeNet
21. Introduction to Alexnet Architecture - Analytics Vidhya, acessado em agosto 12, 2025, https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/
22. Unlocking VGGNet's Power in Deep Learning - Number Analytics, acessado em agosto 12, 2025, https://www.numberanalytics.com/blog/ultimate-guide-vggnet-deep-learning
23. ResNet Explained - Vanishing Gradients, Skip Connections, and Code Implementation | Computer Vision - YouTube, acessado em agosto 12, 2025, https://www.youtube.com/watch?v=OE3XNTBy0hA
24. Segment Anything | Meta AI, acessado em agosto 12, 2025, https://segment-anything.com/
25. Residual neural network - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/Residual_neural_network
26. What is EfficientNet? | SKY ENGINE AI, acessado em agosto 12, 2025, https://www.skyengine.ai/blog/what-is-efficientnet
27. Segment Anything Model (SAM) - The Complete Guide - Viso Suite, acessado em agosto 12, 2025, https://viso.ai/deep-learning/segment-anything-model-sam-explained/
28. Boosting Segment Anything Model Towards Open-Vocabulary Learning - arXiv, acessado em agosto 12, 2025, https://arxiv.org/html/2312.03628v2
29. Top Multimodal Vision Models - Roboflow, acessado em agosto 12, 2025, https://roboflow.com/model-feature/multimodal-vision
30. (PDF) ADVANCEMENTS AND CHALLENGES OF REAL-TIME DATA IN REMOTE SENSING SCENE CLASSIFICATION WITH DEEP LEARNING TECHNIQUES - ResearchGate, acessado em agosto 12, 2025, https://www.researchgate.net/publication/389278699_ADVANCEMENTS_AND_CHALLENGES_OF_REAL-TIME_DATA_IN_REMOTE_SENSING_SCENE_CLASSIFICATION_WITH_DEEP_LEARNING_TECHNIQUES
31. IMPLEMENTATION OF HYBRID DEEP LEARNING CNN MODEL FOR MULTISPECTRAL SATELLITE IMAGE CLASSIFICATION IN LAND CHANGE DETECTION, acessado em agosto 12, 2025, http://www.jatit.org/volumes/Vol103No6/33Vol103No6.pdf
32. Machine Learning Planet High Resolution Training Data for Medium ..., acessado em agosto 12, 2025, https://www.earthdata.nasa.gov/about/competitive-programs/access/ml-planet-data
33. Digital applications unlock remote sensing AI foundation ... - Frontiers, acessado em agosto 12, 2025, https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2025.1520242/full
34. Self-Supervised Learning for Scene Classification in Remote ... - MDPI, acessado em agosto 12, 2025, https://www.mdpi.com/2072-4292/14/16/3995
35. [2504.00901] A Decade of Deep Learning for Remote Sensing Spatiotemporal Fusion: Advances, Challenges, and Opportunities - arXiv, acessado em agosto 12, 2025, https://arxiv.org/abs/2504.00901
36. Google Colab, acessado em agosto 12, 2025, https://research.google.com/colaboratory/faq.html
37. Welcome To Colab - Colab, acessado em agosto 12, 2025, https://colab.research.google.com/
38. Making the Most of your Colab Subscription - Google, acessado em agosto 12, 2025, https://colab.research.google.com/notebooks/pro.ipynb
39. Enable default runtimes with GPUs | Colab Enterprise - Google Cloud, acessado em agosto 12, 2025, https://cloud.google.com/colab/docs/default-runtimes-with-gpus
40. Tracking GPU Memory Usage - Colab, acessado em agosto 12, 2025, https://colab.research.google.com/github/kannankumar/data-diary/blob/master/_notebooks/2020-04-22-Tracking_GPU_Memory_Usage.ipynb
41. monitor cpu, gpu, memory usage in colab (pro) - Stack Overflow, acessado em agosto 12, 2025, https://stackoverflow.com/questions/75371860/monitor-cpu-gpu-memory-usage-in-colab-pro
42. Mounting Google Drive in Google Colab | by Rushi Chaudhari - Medium, acessado em agosto 12, 2025, https://medium.com/@rushic24/mounting-google-drive-in-google-colab-5ecd1d3b735a
43. What's the best way to access Google Drive files in Colab? - Latenode community, acessado em agosto 12, 2025, https://community.latenode.com/t/whats-the-best-way-to-access-google-drive-files-in-colab/13015
44. Hadamard product (matrices) - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
45. Hadamard Product: A Complete Guide to Element-Wise Matrix Multiplication - DataCamp, acessado em agosto 12, 2025, https://www.datacamp.com/tutorial/hadamard-product
46. Python for Data Analysis, 3E - 4 NumPy Basics: Arrays and ..., acessado em agosto 12, 2025, https://wesmckinney.com/book/numpy-basics
47. Broadcasting — NumPy v2.3 Manual, acessado em agosto 12, 2025, https://numpy.org/doc/stable/user/basics.broadcasting.html
48. What Is a Gradient in Machine Learning? - MachineLearningMastery.com, acessado em agosto 12, 2025, https://machinelearningmastery.com/gradient-in-machine-learning/
49. Can someone please explain the intuition behind gradient, curl, and divergence. - Reddit, acessado em agosto 12, 2025, https://www.reddit.com/r/math/comments/33pqn2/can_someone_please_explain_the_intuition_behind/
50. Linear regression: Gradient descent | Machine Learning - Google for Developers, acessado em agosto 12, 2025, https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
51. Gradient Descent Algorithm in Machine Learning - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
52. Understanding Probability Distributions for Machine Learning with Python - MachineLearningMastery.com, acessado em agosto 12, 2025, https://machinelearningmastery.com/understanding-probability-distributions-machine-learning-python/
53. Math for ML: Probability Distributions You need to know | by Rayan Yassminh - Medium, acessado em agosto 12, 2025, https://medium.com/@ryassminh/math-for-ml-probability-distributions-you-need-to-know-b8e644f0dc6f
54. What is Dropout Regularization? Find out :) - Kaggle, acessado em agosto 12, 2025, https://www.kaggle.com/code/pavansanagapati/what-is-dropout-regularization-find-out
55. The Role of Dropout in Neural Networks | by Amit Yadav | Biased-Algorithms - Medium, acessado em agosto 12, 2025, https://medium.com/biased-algorithms/the-role-of-dropout-in-neural-networks-fffbaa77eee7
56. Enhancing Monte Carlo Dropout. Advancing Uncertainty Estimation | by Shiro Matsumoto, acessado em agosto 12, 2025, https://shrmtmt.medium.com/beyond-average-predictions-embracing-variability-with-heteroscedastic-loss-in-deep-learning-f098244cad6f
57. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning - arXiv, acessado em agosto 12, 2025, https://arxiv.org/pdf/1506.02142