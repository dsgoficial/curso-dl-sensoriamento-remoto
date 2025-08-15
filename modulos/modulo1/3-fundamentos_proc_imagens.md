# Processamento de Imagens

Este módulo estabelece as bases essenciais para a compreensão e aplicação de Deep Learning em tarefas de Visão Computacional, com foco particular no Sensoriamento Remoto. Iniciaremos com os fundamentos do processamento digital de imagens, compreendendo como as imagens são representadas e manipuladas computacionalmente usando a biblioteca NumPy. Em seguida, faremos uma transição para os conceitos fundamentais de Redes Neurais Artificiais (RNAs), culminando na introdução prática e teórica do PyTorch, a principal ferramenta que utilizaremos para construir e treinar modelos de Deep Learning. Ao final deste módulo, os alunos terão uma base sólida tanto em processamento de imagens clássico quanto nos pilares das redes neurais e do framework PyTorch, preparando-os para módulos mais avançados em Redes Convolucionais e aplicações específicas em Sensoriamento Remoto.

### Seção 1: Fundamentos de Processamento de Imagens com NumPy

Esta seção explora as representações digitais de imagens e as operações fundamentais para manipulá-las, utilizando o NumPy como ferramenta principal. A compreensão desses conceitos é crucial, pois as imagens são o principal tipo de dado em Deep Learning aplicado à Visão Computacional.

#### 1.1. Representação e Manipulação Básica de Imagens

**1.1.1. Imagens como Matrizes NumPy: Grayscale, RGB e Multiespectral**

Imagens digitais são, em sua essência, arrays de valores de pixels. A biblioteca NumPy, com sua capacidade de manipular arrays multidimensionais (`ndarray`), é a ferramenta fundamental para representar e processar essas imagens.[1, 2] A forma como uma imagem é representada em um array NumPy depende de seu tipo, que pode variar de tons de cinza a imagens multiespectrais complexas.

As **imagens em tons de cinza (grayscale)** são as mais simples, contendo apenas informações de intensidade, sem cor.[1] Cada pixel na imagem é representado por um único valor, que tipicamente varia de 0 (preto) a 255 (branco).[1] Em NumPy, essas imagens são armazenadas como arrays 2D, com a forma `(altura, largura)`.[1, 2] Sua simplicidade as torna eficientes para muitas aplicações práticas de processamento de imagem.[1]

As **imagens RGB (coloridas)** são compostas por três canais de cor: Vermelho (Red), Verde (Green) e Azul (Blue).[1] Cada pixel é representado por um trio de valores, um para cada canal de cor, geralmente também variando de 0 a 255.[1] Essas imagens são armazenadas como arrays NumPy 3D, com a forma `(altura, largura, 3)`.[1, 2] As imagens RGB são o formato padrão para o armazenamento e processamento de cores em dispositivos digitais e são amplamente utilizadas em tarefas como segmentação baseada em cores, reconhecimento facial, detecção de objetos e redes neurais convolucionais (CNNs).[1]

As **imagens multiespectrais** representam uma extensão das imagens RGB, contendo dezenas, centenas ou até milhares de canais.[1] Elas são representadas como arrays NumPy com a forma `(altura, largura, canais)`, onde o número de canais (`N_Canais`) é maior que 3.[1, 3] Cada canal corresponde a uma banda espectral diferente ou a uma leitura de sensor específica, capturando informações que se estendem além do espectro visível (por exemplo, infravermelho ou ultravioleta).[1, 3] No contexto do Sensoriamento Remoto, as imagens multiespectrais são de importância crucial, pois cada banda espectral fornece informações únicas sobre a superfície terrestre, permitindo análises aprofundadas de características como vegetação, água e tipos de solo.[1, 3] A compreensão da representação multiespectral como um array NumPy de forma `(altura, largura, canais)` é, portanto, uma implicação direta para a complexidade dos dados de Sensoriamento Remoto, influenciando o design de modelos de Deep Learning e a necessidade de técnicas de pré-processamento específicas para esses dados.

Os **tipos de dados** para os valores de pixels são frequentemente inteiros sem sinal de 8 bits (`np.uint8`), que podem armazenar valores de 0 a 255.[1, 4] Este tipo de dado é eficiente em termos de memória e é o formato padrão para muitas imagens. No entanto, para operações matemáticas mais complexas, como convoluções ou cálculos de gradientes em redes neurais, ou para uso em algoritmos de Deep Learning, é comum converter esses valores para tipos de ponto flutuante (por exemplo, `float32`, `float64`). Essa conversão é frequentemente acompanhada pela **normalização** dos valores de pixel para um intervalo específico, como  ou [-1, 1], o que ajuda a estabilizar o treinamento de modelos de Deep Learning e a reduzir o impacto de variações de iluminação.[5, 6] Bibliotecas como PIL (Pillow), Matplotlib e OpenCV facilitam o carregamento de imagens em arrays NumPy e a conversão entre esses formatos e tipos de dados.[4, 7]

Em relação às **convenções de coordenadas**, em NumPy e em bibliotecas como `scikit-image`, as imagens 2D (tons de cinza) são indexadas por `(linha, coluna)` ou `(r, c)`, com a origem `(0, 0)` localizada no canto superior esquerdo.[2, 8, 9] Esta convenção é análoga à indexação de matrizes em álgebra linear. Para imagens multicanais, a dimensão do canal é tipicamente a última dimensão, seguindo o formato `(linha, coluna, canal)`, embora algumas funções permitam especificar a posição do canal através de um argumento `channel_axis`.[2, 8, 9] Esta convenção difere das coordenadas Cartesianas `(x, y)` tradicionais, onde `x` é horizontal, `y` é vertical e a origem é frequentemente no canto inferior esquerdo.[2, 8, 9] A escolha da representação da imagem não é apenas uma formalidade, mas um fator determinante para a complexidade do pipeline de processamento e a arquitetura do modelo de Deep Learning, especialmente no contexto de dados de Sensoriamento Remoto.

A tabela a seguir sumariza as principais características de cada tipo de representação de imagem:

**Tabela: Comparativo de Representação de Imagens (Grayscale, RGB, Multiespectral)**

| Característica | Imagem Grayscale | Imagem RGB | Imagem Multiespectral |
| :-------------------- | :------------------------ | :-------------------------- | :---------------------------------- |
| **Dimensão NumPy** | `(Altura, Largura)` | `(Altura, Largura, 3)` | `(Altura, Largura, N_Canais)` |
| **Canais** | 1 (Intensidade) | 3 (Vermelho, Verde, Azul) | N_Canais (Ex: Visível, Infravermelho, etc.) |
| **Valores de Pixel** | 0-255 (Intensidade) | 0-255 por canal | 0-255 (ou outros ranges, dependendo do sensor) por canal |
| **Uso Comum** | Processamento básico, detecção de bordas | Fotografia, Visão Computacional geral | Sensoriamento Remoto, Imagem Médica, Análise de Materiais |
| **Exemplo de Aplicação** | Filtragem de ruído, binarização | Classificação de objetos, reconhecimento facial | Classificação de uso do solo, detecção de mudanças, análise de vegetação |

Esta tabela serve como um guia de referência rápido e claro, consolidando informações cruciais sobre a estrutura dos dados de imagem que serão manipulados ao longo do curso, facilitando a identificação das diferenças e implicações de cada representação.

**Exemplo de Código 1.1.1: Criação e Manipulação Básica de Imagens com NumPy**
O código a seguir demonstra como criar imagens em tons de cinza, RGB e simular uma imagem multiespectral usando NumPy. Também mostra a conversão de tipos de dados e a normalização.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Imagem em tons de cinza (Grayscale)
# Representada como um array 2D (Altura, Largura)
altura, largura = 100, 150
imagem_grayscale = np.random.randint(0, 256, size=(altura, largura), dtype=np.uint8)
print(f"Shape da imagem grayscale: {imagem_grayscale.shape}")
print(f"Tipo de dado da imagem grayscale: {imagem_grayscale.dtype}")

plt.figure(figsize=(6, 3))
plt.imshow(imagem_grayscale, cmap='gray')
plt.title('Imagem Grayscale (Simulada)')
plt.axis('off')
plt.show()

# 2. Imagem RGB (Colorida)
# Representada como um array 3D (Altura, Largura, 3)
imagem_rgb = np.random.randint(0, 256, size=(altura, largura, 3), dtype=np.uint8)
print(f"\nShape da imagem RGB: {imagem_rgb.shape}")
print(f"Tipo de dado da imagem RGB: {imagem_rgb.dtype}")

plt.figure(figsize=(6, 3))
plt.imshow(imagem_rgb)
plt.title('Imagem RGB (Simulada)')
plt.axis('off')
plt.show()

# 3. Imagem Multiespectral
# Representada como um array 3D (Altura, Largura, N_Canais)
n_canais = 7 # Exemplo: 7 bandas espectrais
imagem_multiespectral = np.random.randint(0, 256, size=(altura, largura, n_canais), dtype=np.uint8)
print(f"\nShape da imagem multiespectral: {imagem_multiespectral.shape}")
print(f"Tipo de dado da imagem multiespectral: {imagem_multiespectral.dtype}")

# Exibindo uma das bandas da imagem multiespectral (ex: banda 0)
plt.figure(figsize=(6, 3))
plt.imshow(imagem_multiespectral[:, :, 0], cmap='viridis') # Usando um colormap para visualização
plt.title('Banda 0 da Imagem Multiespectral (Simulada)')
plt.axis('off')
plt.show()

# 4. Conversão de tipo de dado e Normalização
# Convertendo para float32 e normalizando para o intervalo 
imagem_rgb_float = imagem_rgb.astype(np.float32) / 255.0
print(f"\nTipo de dado da imagem RGB (float): {imagem_rgb_float.dtype}")
print(f"Valor mínimo após normalização : {imagem_rgb_float.min()}")
print(f"Valor máximo após normalização : {imagem_rgb_float.max()}")

# Normalização para o intervalo [-1, 1]
imagem_rgb_normalizada_neg1_1 = (imagem_rgb.astype(np.float32) / 127.5) - 1.0
print(f"Valor mínimo após normalização [-1, 1]: {imagem_rgb_normalizada_neg1_1.min()}")
print(f"Valor máximo após normalização [-1, 1]: {imagem_rgb_normalizada_neg1_1.max()}")

plt.figure(figsize=(6, 3))
plt.imshow(imagem_rgb_float) # imshow pode lidar com floats em 
plt.title('Imagem RGB Normalizada ')
plt.axis('off')
plt.show()
````

#### 1.1.2. Operações Fundamentais com NumPy: Slicing, Reshape e Normalização

A manipulação eficiente de arrays NumPy é essencial para o processamento de imagens em Deep Learning. Operações como fatiamento (slicing), remodelagem (reshape) e normalização são cruciais para preparar e transformar os dados de imagem.

O **fatiamento (slicing)** em NumPy estende o conceito básico de fatiamento do Python para arrays N-dimensionais, permitindo a seleção, extração e modificação de regiões ou componentes específicos de arrays de imagem.[10, 11] A sintaxe básica é `array[start:stop:step]` aplicada a cada dimensão.

  * `start`: O índice inicial (inclusive). Se omitido, assume 0 para passo positivo e `n-1` para passo negativo.
  * `stop`: O índice final (exclusive). Se omitido, assume `n` para passo positivo e `-n-1` para passo negativo.
  * `step`: O tamanho do passo (e.g., `2` para selecionar a cada dois elementos). Se omitido, o padrão é `1`. Um passo negativo inverte a ordem da seleção.[11]

Por exemplo, para recortar uma imagem ou selecionar uma área específica, podem-se especificar os intervalos de linhas e colunas, como `imagem[50:150, 100:200]` para uma região de interesse (ROI).[10, 11] É fundamental compreender que o fatiamento em NumPy geralmente retorna uma *view* (visão) do array original, e não uma cópia independente. Isso implica que qualquer modificação feita na *view* afetará diretamente o array original. Para garantir uma cópia independente, é necessário usar explicitamente o método `.copy()`.[10, 11] Essa característica é importante para a otimização de memória, mas exige atenção para evitar efeitos colaterais indesejados em pipelines de processamento de dados.

A operação de **remodelagem (reshape)** permite alterar a forma (dimensões) de um array NumPy sem modificar seus dados subjacentes.[12, 13] A função `np.reshape(array, nova_forma)` é utilizada, e uma dimensão pode ser especificada como `-1`, permitindo que o NumPy infira automaticamente o tamanho dessa dimensão com base no comprimento total do array e nas outras dimensões.[12, 13] O parâmetro `order` controla a ordem de leitura e escrita dos elementos (`'C'` para ordem C-like, `'F'` para Fortran-like, `'A'` para automático), o que pode impactar a eficiência da memória e o desempenho computacional devido à localidade dos dados.[12, 13] Em aplicações de imagem, o `reshape` é frequentemente usado para achatar uma imagem 2D ou 3D em um vetor 1D (por exemplo, para entrada em Perceptrons Multicamadas - MLPs) ou para reorganizar as dimensões dos canais para formatos esperados por certas bibliotecas ou modelos de Deep Learning.[14, 15]

A **normalização** é o processo de escalonamento dos valores de pixels de uma imagem para um intervalo específico, tipicamente entre  ou [-1, 1]. Sua importância é multifacetada:

  * **Reduz o impacto de variações de iluminação e condições de aquisição de imagem:** Garante que o modelo não seja indevidamente influenciado por diferenças de brilho ou contraste entre as imagens.
  * **Melhora a velocidade de convergência e a estabilidade de algoritmos de Machine Learning:** Valores de entrada em uma faixa consistente evitam que gradientes se tornem muito grandes ou muito pequenos, o que pode levar a problemas de gradiente explosivo ou evanescente.
  * **Aprimora a interpretabilidade dos resultados:** Ao reduzir a influência de características dominantes, todas as características contribuem de forma mais equitativa para o aprendizado do modelo.

As técnicas mais comuns incluem:

  * **Min-Max Scaling:** Escala os valores para um novo intervalo usando a fórmula `(x - min) / (max - min)`, onde `min` e `max` são os valores mínimo e máximo de pixel na imagem ou no conjunto de dados. Para escalar para o intervalo , a fórmula é `(pixel_value - min_pixel_value) / (max_pixel_value - min_pixel_value)`.
  * **Padronização (Standardization):** Transforma os valores de pixel para que tenham média zero e variância unitária, utilizando a fórmula `(x - média) / desvio_padrão`. Esta técnica é útil quando a distribuição dos dados é aproximadamente Gaussiana.
    A implementação pode ser realizada com operações NumPy vetorizadas, que são rápidas e eficientes, ou através de bibliotecas como `sklearn.preprocessing.MinMaxScaler`.

As operações de manipulação de arrays NumPy, como fatiamento, remodelagem e normalização, não são apenas ferramentas para carregar e manipular imagens, mas são a base para construir pipelines de dados eficientes e robustos. Modelos de Deep Learning, especialmente aqueles que lidam com imagens de Sensoriamento Remoto, processam grandes volumes de dados. Um pré-processamento ineficiente ou incorreto pode levar a problemas de memória, lentidão no treinamento e até mesmo à divergência do modelo. A compreensão desses detalhes técnicos é crucial para a otimização e a estabilidade dos pipelines de Deep Learning, que são pré-requisitos para o treinamento bem-sucedido de modelos, especialmente com os grandes e complexos datasets de Sensoriamento Remoto.

**Exemplo de Código 1.1.2: Operações Fundamentais com NumPy**
O código a seguir demonstra o uso de fatiamento, remodelagem e normalização em arrays NumPy, simulando operações em imagens.

```python
import numpy as np
import matplotlib.pyplot as plt

# Criando uma imagem RGB simulada (Altura x Largura x Canais)
imagem_original = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(imagem_original)
plt.title('Imagem Original')
plt.axis('off')

# 1. Fatiamento (Slicing)
# Extraindo uma Região de Interesse (ROI)
roi = imagem_original[50:150, 100:250, :] # Linhas 50-149, Colunas 100-249, todos os canais
print(f"Shape da ROI: {roi.shape}")

plt.subplot(1, 3, 2)
plt.imshow(roi)
plt.title('ROI (Fatiamento)')
plt.axis('off')

# Selecionando um canal específico (ex: canal Vermelho)
canal_vermelho = imagem_original[:, :, 0]
print(f"Shape do canal Vermelho: {canal_vermelho.shape}")

plt.subplot(1, 3, 3)
plt.imshow(canal_vermelho, cmap='Reds_r') # Usando um colormap para visualizar o canal
plt.title('Canal Vermelho')
plt.axis('off')
plt.tight_layout()
plt.show()

# Demonstração de view vs. copy
imagem_teste = np.array([[1, 2], [3, 4]], dtype=np.uint8)
view_teste = imagem_teste[0, :]
view_teste[:] = 99 # Modifica a view
print(f"\nImagem original após modificar a view: \n{imagem_teste}") # A imagem original é alterada

imagem_teste_copy = np.array([[1, 2], [3, 4]], dtype=np.uint8)
copy_teste = imagem_teste_copy[0, :].copy()
copy_teste[:] = 99 # Modifica a cópia
print(f"Imagem original após modificar a cópia: \n{imagem_teste_copy}") # A imagem original NÃO é alterada

# 2. Remodelagem (Reshape)
# Achatar a imagem para um vetor 1D (comum para MLPs)
imagem_achatada = imagem_original.reshape(-1)
print(f"\nShape da imagem achatada: {imagem_achatada.shape}")

# Remodelar para adicionar uma dimensão de batch (1, H, W, C)
imagem_batch = imagem_original[np.newaxis, :, :, :]
print(f"Shape da imagem com dimensão de batch: {imagem_batch.shape}")

# Remodelar para o formato (C, H, W) (comum em PyTorch)
# Primeiro, precisamos transpor as dimensões
imagem_chw = imagem_original.transpose((2, 0, 1))
print(f"Shape da imagem transposta (C, H, W): {imagem_chw.shape}")

# 3. Normalização
# Convertendo para float para normalização
imagem_float = imagem_original.astype(np.float32)

# Min-Max Scaling para 
min_val = imagem_float.min()
max_val = imagem_float.max()
imagem_normalizada_0_1 = (imagem_float - min_val) / (max_val - min_val)
print(f"\nMin-Max Normalização  - Min: {imagem_normalizada_0_1.min()}, Max: {imagem_normalizada_0_1.max()}")

# Padronização (Standardization) para média 0 e desvio padrão 1
media = imagem_float.mean()
desvio_padrao = imagem_float.std()
imagem_padronizada = (imagem_float - media) / desvio_padrao
print(f"Padronização - Média: {imagem_padronizada.mean():.4f}, Desvio Padrão: {imagem_padronizada.std():.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagem_normalizada_0_1)
plt.title('Imagem Normalizada ')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagem_padronizada[:,:,0], cmap='gray') # Exibindo apenas um canal para visualização
plt.title('Imagem Padronizada (Canal 0)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

#### 1.2. Filtros Espaciais e Convolução

Esta seção introduz os conceitos de filtros espaciais e a operação de convolução, que são fundamentais tanto para o processamento de imagens clássico quanto para as Redes Neurais Convolucionais (CNNs).

**1.2.1. Conceitos de Convolução 2D: Sliding Window e Padding**

A **convolução 2D** é uma operação que atua sobre duas "sinais" bidimensionais: uma imagem de entrada e um "kernel" (ou filtro), produzindo uma terceira imagem de saída.[16, 17, 18] O funcionamento intuitivo da convolução envolve a ideia de uma "janela deslizante" (sliding window): o kernel, que é uma pequena matriz de pesos, desliza sobre a imagem de entrada. Em cada posição, os valores correspondentes da porção da imagem sob a janela e do kernel são multiplicados elemento a elemento e, em seguida, somados. O resultado dessa soma ponderada se torna o novo valor do pixel na imagem de saída. Essa operação é fundamental para extrair características locais da imagem, como bordas e texturas.

O **stride (passo)** define a distância que a janela deslizante (o kernel) se move a cada passo, tanto na direção horizontal quanto na vertical. Um `stride` maior que 1 faz com que a janela pule pixels, resultando em uma imagem de saída com dimensões reduzidas. Essa característica é utilizada em CNNs para reduzir a dimensionalidade espacial dos mapas de características (feature maps), de forma análoga ao pooling. No entanto, em camadas convolucionais típicas, um `stride=(1, 1)` é mais comum para preservar a maior quantidade possível de informações espaciais.

O **padding (preenchimento)** aborda como as bordas da imagem são tratadas durante a operação de convolução. Para pixels localizados nas bordas, o kernel pode não se encaixar completamente dentro dos limites da imagem. Existem dois tipos comuns de padding:

  * **`'valid'` (sem preenchimento):** A janela de convolução permanece inteiramente dentro da imagem de entrada. A consequência é que a imagem de saída encolhe, e essa redução de tamanho pode limitar o número de camadas que a rede neural pode conter, especialmente quando as entradas são pequenas.
  * **`'same'` (com preenchimento):** A imagem de entrada é preenchida com zeros (ou outros valores) ao redor de suas bordas, com o objetivo de garantir que o tamanho da imagem de saída seja o mesmo que o tamanho da entrada. Isso evita a redução dimensional a cada camada convolucional.

A escolha entre `'valid'` e `'same'` padding envolve trade-offs: enquanto `'same'` padding ajuda a manter a dimensionalidade, pode diluir a influência dos pixels nas bordas da imagem original.[19]

A convolução 2D, com seus conceitos de sliding window, stride e padding, é a operação fundamental que define as camadas convolucionais em Redes Neurais Convolucionais (CNNs). A compreensão desses conceitos estabelece uma relação direta entre o processamento de imagens clássico e o Deep Learning. As decisões sobre `padding` e `stride` não são apenas detalhes de implementação, mas escolhas arquitetônicas cruciais que afetam o tamanho dos mapas de características e, consequentemente, a profundidade e a capacidade de aprendizado de uma CNN. Assim, esta seção serve como a base conceitual direta para as CNNs, permitindo que os alunos compreendam que os "filtros" em CNNs são análogos aos kernels clássicos, e que os parâmetros de convolução são ferramentas de design de arquitetura de rede.

**Exemplo de Código 1.2.1: Convolução 2D Simples (do zero)**
Este exemplo demonstra a implementação de uma convolução 2D básica, sem otimizações, para ilustrar o conceito de janela deslizante.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d # Para validação/comparação

def convolve2D_from_scratch(image, kernel, stride=1, padding='valid'):
    """
    Implementa a operação de convolução 2D manualmente.
    Suporta imagens em tons de cinza (2D) e RGB (3D).
    Para RGB, aplica o kernel a cada canal separadamente.

    Parâmetros:
    - image: np.array, shape (H, W) ou (H, W, C) - Imagem.
    - kernel: np.array, shape (K, K) - Filtro/kernel.
    - stride: int, passo para mover o filtro.
    - padding: str, 'valid' (sem preenchimento) ou 'same' (preenchimento com zeros).

    Retorna:
    - output: np.array, resultado após aplicar o filtro.
    """
    if image.ndim == 3: # Lida com imagens RGB aplicando convolução a cada canal
        channels = image.shape[2]
        output_channels = []
        for c in range(channels):
            # Chamada recursiva para cada canal
            output_channels.append(convolve2D_from_scratch(image[:, :, c], kernel, stride, padding))
        return np.stack(output_channels, axis=-1) # Empilha os resultados de volta em um array 3D

    # Assume imagem em tons de cinza (2D) a partir daqui
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Determinar o tamanho do padding
    if padding == 'same':
        # Calcula o padding necessário para que a saída tenha o mesmo tamanho da entrada (para stride=1)
        # Para stride > 1, o cálculo de 'same' é mais complexo e pode não resultar no mesmo tamanho exato.
        # Aqui, usamos a aproximação comum para 'same' padding.
        pad_h = (kernel_height - 1) // 2
        pad_w = (kernel_width - 1) // 2
        image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    else: # 'valid'
        pad_h, pad_w = 0, 0
        image_padded = image

    # Dimensões da imagem após o padding
    padded_height, padded_width = image_padded.shape

    # Calcular as dimensões da saída
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    # Inicializar o array de saída. Usamos float32 para evitar problemas de overflow/underflow
    # e para manter a precisão dos cálculos.
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Realizar a convolução usando slicing e broadcasting (vetorização)
    # Iteramos sobre as posições de saída e, para cada posição, extraímos a região
    # correspondente da imagem acolchoada e aplicamos a multiplicação elemento a elemento
    # e a soma com o kernel.
    for r in range(output_height):
        for c in range(output_width):
            # Extrair a região da imagem coberta pelo kernel usando slicing
            region = image_padded[r * stride : r * stride + kernel_height,
                                  c * stride : c * stride + kernel_width]
            # Multiplicação elemento a elemento e soma (operação vetorizada do NumPy)
            output[r, c] = np.sum(region * kernel)
            
    return output

# Criando uma imagem de teste simples (tons de cinza)
image = np.array([[15, 20, 21, 22, 23],
                  [30, 35, 36, 37, 38],
                  [45, 50, 51, 52, 53],
                  [60, 65, 66, 67, 68],
                  [75, 80, 81, 82, 83]], dtype=np.uint8)

# Kernel de exemplo (filtro de média 3x3)
kernel_mean = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.float32) / 9.0

# Aplicar convolução com padding='valid' (saída encolhe)
output_valid = convolve2D_from_scratch(image, kernel_mean, stride=1, padding='valid')
print("Output com padding='valid':\n", output_valid)
print(f"Shape da saída (valid): {output_valid.shape}")

# Aplicar convolução com padding='same' (saída mantém o tamanho)
output_same = convolve2D_from_scratch(image, kernel_mean, stride=1, padding='same')
print("\nOutput com padding='same':\n", output_same)
print(f"Shape da saída (same): {output_same.shape}")

# Aplicar convolução com stride=2 e padding='valid'
output_stride2 = convolve2D_from_scratch(image, kernel_mean, stride=2, padding='valid')
print("\nOutput com stride=2 e padding='valid':\n", output_stride2)
print(f"Shape da saída (stride=2, valid): {output_stride2.shape}")

# Visualização
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_valid.astype(np.uint8), cmap='gray') # Converter para uint8 para visualização
plt.title('Convolução (Valid)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_same.astype(np.uint8), cmap='gray') # Converter para uint8 para visualização
plt.title('Convolução (Same)')
plt.axis('off')
plt.tight_layout()
plt.show()

# Validação com scipy.signal.convolve2d
scipy_output_valid_signal = convolve2d(image.astype(np.float32), kernel_mean, mode='valid')
print("\nValidação com scipy.signal.convolve2d (mode='valid'):\n", scipy_output_valid_signal)
print(f"Shape da validação (valid): {scipy_output_valid_signal.shape}")
print(f"Manual e SciPy (valid) são iguais: {np.allclose(output_valid, scipy_output_valid_signal)}")

scipy_output_same_signal = convolve2d(image.astype(np.float32), kernel_mean, mode='same')
print("\nValidação com scipy.signal.convolve2d (mode='same'):\n", scipy_output_same_signal)
print(f"Shape da validação (same): {scipy_output_same_signal.shape}")
print(f"Manual e SciPy (same) são iguais: {np.allclose(output_same, scipy_output_same_signal)}")
```

#### 1.2.2. Filtros Clássicos: Média, Gaussiano, Sobel e Laplaciano

Os filtros espaciais são ferramentas essenciais no processamento de imagens para tarefas como suavização, realce e detecção de bordas. Cada filtro possui um propósito e um mecanismo de funcionamento distintos. A implementação "do zero" desses filtros nos ajuda a entender profundamente como as operações de convolução atuam sobre os pixels da imagem.

O **filtro da média (mean filter)** é um método simples e intuitivo para suavizar imagens e reduzir ruído.[24, 25, 20, 26] Ele opera substituindo o valor de cada pixel pela média dos valores de seus vizinhos, incluindo ele mesmo, dentro de uma janela (kernel) definida, como um kernel 3x3. Essa operação tem o efeito de eliminar valores de pixels que são atípicos em relação ao seu entorno, atuando como um filtro passa-baixa que reduz variações de intensidade e detalhes de alta frequência. Embora eficaz na redução de ruído, o filtro da média pode borrar as bordas da imagem.

  * **Efeito na Imagem:** O resultado é uma imagem mais "suave" ou "borrada", onde o ruído pontual é reduzido. Bordas nítidas tendem a ser suavizadas.
  * **Aplicação em Sensoriamento Remoto:** Utilizado para reduzir ruído em imagens de satélite (e.g., ruído de "sal e pimenta" ou ruído Gaussiano), suavizar pequenas variações em mapas de uso do solo, ou como um pré-processamento para outras análises onde detalhes finos não são desejados.

O **filtro Gaussiano (Gaussian filter)** é uma técnica de suavização de imagens e redução de ruído mais sofisticada que o filtro da média, pois preserva melhor as bordas. Ele funciona de maneira semelhante ao filtro da média, mas os vizinhos são ponderados por uma função Gaussiana (distribuição normal), o que significa que pixels mais próximos ao centro do kernel contribuem mais para o valor médio. As características do filtro Gaussiano são definidas pelo tamanho do kernel e pelo valor de sigma ($\sigma$), que controla a largura da distribuição Gaussiana: um sigma maior resulta em um maior embaçamento da imagem. A função Gaussiana 2D é dada por:

$G(x,y)=\frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$

Onde $x$ e $y$ são as distâncias do centro do kernel.

  * **Efeito na Imagem:** Produz um desfoque mais natural e uniforme do que o filtro da média, preservando melhor as estruturas de borda. É muito eficaz para reduzir ruído Gaussiano.
  * **Aplicação em Sensoriamento Remoto:** Frequentemente usado como um passo de pré-processamento para suavizar imagens antes de aplicar detectores de borda (como no Laplaciano de Gaussiano - LoG), para reduzir o efeito de neblina atmosférica, ou para criar mapas de densidade suavizados.

O **operador Sobel (Sobel operator)** é amplamente utilizado para detecção de bordas, enfatizando regiões de alta frequência espacial que correspondem a transições de intensidade. Tecnicamente, ele é um operador de diferenciação discreta que calcula uma aproximação do gradiente da função de intensidade da imagem. O Sobel utiliza dois kernels 3x3 (um para detectar mudanças horizontais, Gx, e outro para mudanças verticais, Gy) que são convoluídos com a imagem original.

  * **Kernel Horizontal (Gx):**
    ```
    [-1  0  1]
    [-2  0  2]
    [-1  0  1]
    ```
  * **Kernel Vertical (Gy):**
    ```
    [-1 -2 -1]
    [ 0  0  0]
    [ 1  2  1]
    ```

A magnitude do gradiente em cada ponto é então combinada (por exemplo, usando a adição Pitagórica: $G = \sqrt{G_x^2 + G_y^2}$) para determinar a intensidade da borda. O operador Sobel é menos sensível a ruído do que outros operadores de borda mais simples, mas ainda pode amplificar altas frequências.

  * **Efeito na Imagem:** O resultado é uma imagem onde as bordas são realçadas, aparecendo como linhas brancas (ou de alta intensidade) contra um fundo escuro. Ele detecta a força e a orientação das bordas.
  * **Aplicação em Sensoriamento Remoto:** Essencial para identificar feições lineares como estradas, rios, limites de campos agrícolas, ou estruturas urbanas em imagens de satélite. Pode ser usado para delinear áreas de mudança ou transição.

O **filtro Laplaciano (Laplacian filter)** é um detector de bordas que destaca regiões de rápida mudança de intensidade, calculando a segunda derivada espacial de uma imagem.[27, 20, 33, 29, 30] Devido à sua alta sensibilidade a ruído, o Laplaciano é frequentemente aplicado a uma imagem que foi previamente suavizada com um filtro Gaussiano; essa combinação é conhecida como filtro LoG (Laplacian of Gaussian). O LoG é particularmente eficaz na detecção de "zero-crossings" (pontos onde a taxa de mudança de intensidade inverte a direção), que correspondem às bordas dos objetos. Um kernel Laplaciano comum é:

```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

Ou uma variante que inclui diagonais:

```
[ 1  1  1]
[ 1 -8  1]
[ 1  1  1]
```

  * **Efeito na Imagem:** O Laplaciano realça detalhes finos e bordas, mas é extremamente sensível a ruído, que também é realçado. O LoG (Laplaciano de Gaussiano) suaviza a imagem primeiro, resultando em bordas mais limpas e menos ruído.
  * **Aplicação em Sensoriamento Remoto:** Usado para realçar texturas finas em imagens de alta resolução (e.g., padrões de vegetação, detalhes de telhados), detecção de pequenos objetos (e.g., árvores individuais, veículos), ou para afiar imagens que foram borradas.

Cada um desses filtros clássicos tem um propósito específico, como suavização ou detecção de bordas, e a intuição por trás deles é que eles realizam a *extração de características* de baixo nível. Em Deep Learning, as camadas convolucionais das CNNs aprendem kernels (filtros) que são otimizados para extrair características hierárquicas, que podem ser muito mais complexas do que as detectadas pelos filtros clássicos. Assim, a compreensão dos filtros clássicos fornece uma analogia fundamental para entender o que as CNNs estão "vendo" e aprendendo em suas camadas iniciais. A sensibilidade ao ruído de filtros como o Laplaciano também destaca a importância do pré-processamento de dados e da robustez do modelo, conceitos que se aplicam diretamente ao Deep Learning.

A tabela a seguir resume as características e aplicações dos filtros espaciais clássicos:

**Tabela: Resumo de Filtros Espaciais e Suas Aplicações**

| Filtro | Propósito Principal | Princípio de Funcionamento | Aplicações Típicas | Observações Importantes |
| :----------------- | :------------------------- | :-------------------------------------------------------- | :-------------------------------------------------------- | :---------------------------------------------------------- |
| **Média** | Suavização, Redução de Ruído | Substitui pixel pela média dos vizinhos. Filtro passa-baixa. | Remoção de ruído "salt-and-pepper", desfoque suave. | Pode borrar bordas. |
| **Gaussiano** | Suavização, Redução de Ruído | Média ponderada dos vizinhos por função Gaussiana. | Redução de ruído, pré-processamento para detecção de bordas. | Mais eficaz que a média, preserva melhor as bordas. |
| **Sobel** | Detecção de Bordas | Aproximação do gradiente 2D (primeira derivada). | Realce de bordas, segmentação, reconhecimento de formas. | Sensível à orientação (horizontal/vertical). |
| **Laplaciano** | Detecção de Bordas | Aproximação da segunda derivada espacial. | Detecção de bordas finas, realce de detalhes. | Muito sensível a ruído; frequentemente usado com Gaussiano (LoG). |

Esta tabela oferece um panorama conciso e comparativo dos filtros espaciais, facilitando a memorização e a aplicação prática desses conceitos.

**Exemplo de Código 1.2.2: Aplicação de Filtros Clássicos (Implementação do Zero)**
O código a seguir demonstra a aplicação de filtros de média, Gaussiano, Sobel e Laplaciano em uma imagem usando a função `convolve2D_from_scratch` implementada anteriormente.

```python
import numpy as np
import matplotlib.pyplot as plt

# Reutiliza a função convolve2D_from_scratch definida no Exemplo de Código 1.2.1
# (Assumindo que ela já foi executada ou está no mesmo script/notebook)
# def convolve2D_from_scratch(...):...

def display_images(images, titles, cmap='gray', figsize=(15, 5)):
    """Função auxiliar para exibir múltiplas imagens."""
    num_images = len(images)
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # Garante que a imagem esteja em formato exibível (0-255 para uint8, ou 0-1 para float)
        if images[i].dtype == np.uint8:
            plt.imshow(images[i], cmap=cmap)
        else:
            # Normaliza imagens float para exibição se não estiverem no intervalo 0-1
            display_img = images[i]
            if display_img.max() > 1.0 or display_img.min() < 0.0:
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            plt.imshow(display_img, cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def gaussian_kernel(size, sigma=1):
    """Gera um kernel Gaussiano 2D."""
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(kernel_1d, kernel_1d)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel) # Normaliza a soma para 1

# Criar uma imagem de teste (tons de cinza)
# Adicionar um pouco de ruído para ver o efeito dos filtros de suavização
image_original = np.zeros((100, 100), dtype=np.float32)
image_original[20:80, 20:80] = 200 # Quadrado branco no centro
image_original[30:70, 30:70] = 50  # Quadrado cinza dentro
# Adicionar ruído "sal e pimenta"
noise_amount = 0.02
num_salt = np.ceil(noise_amount * image_original.size * 0.5).astype(int)
coords_salt = [np.random.randint(0, s, num_salt) for s in image_original.shape]
image_original[tuple(coords_salt)] = 255
num_pepper = np.ceil(noise_amount * image_original.size * 0.5).astype(int)
coords_pepper = [np.random.randint(0, s, num_pepper) for s in image_original.shape]
image_original[tuple(coords_pepper)] = 0

# Converter para float32 para operações de filtragem para evitar problemas de overflow/underflow
image_float = image_original.astype(np.float32)

# --- 1. Filtro da Média (Mean Filter) ---
print("Aplicando Filtro da Média...")
kernel_mean = np.ones((3, 3), dtype=np.float32) / 9.0
image_mean_filtered = convolve2D_from_scratch(image_float, kernel_mean, padding='same')
print("Filtro da Média aplicado.")

# --- 2. Filtro Gaussiano (Gaussian Filter) ---
print("Aplicando Filtro Gaussiano...")
kernel_gaussian = gaussian_kernel(size=5, sigma=1.5) # Kernel 5x5 com sigma 1.5
image_gaussian_filtered = convolve2D_from_scratch(image_float, kernel_gaussian, padding='same')
print("Filtro Gaussiano aplicado.")

# --- 3. Operador Sobel (Sobel Operator) ---
print("Aplicando Operador Sobel...")
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
sobel_y_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)

grad_x = convolve2D_from_scratch(image_float, sobel_x_kernel, padding='same')
grad_y = convolve2D_from_scratch(image_float, sobel_y_kernel, padding='same')
sobel_magnitude = np.sqrt(grad_x**2 + grad_y**2)
# Normalizar para exibição (valores podem ser maiores que 255)
sobel_magnitude_display = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
print("Operador Sobel aplicado.")

# --- 4. Filtro Laplaciano (Laplacian Filter) ---
print("Aplicando Filtro Laplaciano...")
laplacian_kernel = np.array([[ 0,  1,  0],
                             [ 1, -4,  1],
                             [ 0,  1,  0]], dtype=np.float32)
image_laplacian = convolve2D_from_scratch(image_float, laplacian_kernel, padding='same')
# Normalizar para exibição (Laplaciano pode ter valores negativos)
image_laplacian_display = np.abs(image_laplacian)
image_laplacian_display = (image_laplacian_display / image_laplacian_display.max() * 255).astype(np.uint8)
print("Filtro Laplaciano aplicado.")

# --- 5. Laplaciano de Gaussiano (LoG) ---
print("Aplicando Laplaciano de Gaussiano (LoG)...")
# Primeiro, aplicar suavização Gaussiana para reduzir ruído
image_gaussian_for_log = convolve2D_from_scratch(image_float, gaussian_kernel(size=5, sigma=1.0), padding='same')
# Em seguida, aplicar Laplaciano
image_log = convolve2D_from_scratch(image_gaussian_for_log, laplacian_kernel, padding='same')
image_log_display = np.abs(image_log)
image_log_display = (image_log_display / image_log_display.max() * 255).astype(np.uint8)
print("Laplaciano de Gaussiano (LoG) aplicado.")

# Exibir resultados
display_images([image_original, image_mean_filtered, image_gaussian_filtered, sobel_magnitude_display, image_laplacian_display, image_log_display],
               ['Original com Ruído', 'Filtro da Média', 'Filtro Gaussiano', 'Operador Sobel (Mag)', 'Filtro Laplaciano', 'Laplaciano de Gaussiano'],
               cmap='gray', figsize=(18, 6))
```

#### 1.3. Segmentação Clássica de Imagens

**1.3.1. Segmentação Baseada em Pixels vs. Baseada em Regiões**

A **segmentação de imagens** é o processo de particionar uma imagem digital em múltiplos segmentos, regiões ou objetos (conjuntos de pixels). O objetivo principal é simplificar a representação da imagem, transformando-a em algo mais significativo e fácil de analisar para diversas aplicações de visão computacional. O resultado da segmentação é um conjunto de segmentos que, coletivamente, cobrem a imagem inteira, ou um conjunto de contornos extraídos da imagem. Cada pixel em uma região é similar em relação a alguma característica ou propriedade computada, como cor, intensidade ou textura.

As abordagens de segmentação podem ser broadly categorizadas em:

  * **Segmentação Baseada em Pixels:** Esta abordagem foca nas propriedades individuais de cada pixel para decidir sua classificação ou associação a um segmento. A decisão para a associação de um pixel a um segmento é baseada em regras multidimensionais, que podem considerar fatores como iluminação da imagem, ambiente e aplicação. Exemplos típicos incluem o **Thresholding (Limiarização)**, onde a decisão é baseada no valor de intensidade de um pixel em relação a um limiar predefinido, e a **Detecção de Bordas**, que identifica mudanças abruptas de intensidade entre pixels vizinhos. Embora simples e computacionalmente eficientes, essas técnicas frequentemente resultam em elementos desconectados (por exemplo, bordas não fechadas) que exigem processamento adicional para formar regiões coerentes.

  * **Segmentação Baseada em Regiões:** Em contraste, esta abordagem visa agrupar pixels adjacentes que compartilham características semelhantes, como cor, intensidade ou textura, formando regiões homogêneas. O objetivo é criar segmentos conectados e coerentes. Exemplos proeminentes incluem **Region Growing (Crescimento de Regiões)**, onde pixels vizinhos são comparados a um pixel de referência e adicionados à região se a diferença de propriedades for menor que um limiar, **K-Means** (um algoritmo de agrupamento que agrupa pixels com base na similaridade de características) e o algoritmo **Watershed**.

Além dessas abordagens clássicas, o campo da segmentação de imagens evoluiu significativamente com o advento do Deep Learning, dando origem a categorias mais avançadas:

  * **Segmentação Semântica:** Esta abordagem detecta, para *cada pixel* da imagem, a classe à qual ele pertence. Por exemplo, em uma imagem com várias árvores, todos os pixels que correspondem a qualquer árvore seriam atribuídos ao mesmo ID de classe "árvore", sem distinguir entre as árvores individuais. O foco é classificar cada pixel em uma categoria predefinida.
  * **Segmentação de Instância:** Diferente da segmentação semântica, esta abordagem identifica, para *cada pixel*, a *instância específica* do objeto a que ele pertence. Retomando o exemplo das árvores, se houver várias árvores na imagem, cada árvore individual seria segmentada como um objeto único e distinto, recebendo um ID de instância diferente. O objetivo é distinguir ocorrências individuais de objetos dentro de uma mesma classe.
  * **Segmentação Panóptica:** Esta abordagem combina as metas da segmentação semântica e de instância. Ela classifica cada pixel por sua classe e, ao mesmo tempo, distingue instâncias diferentes da mesma classe.

A distinção entre abordagens baseadas em pixels e em regiões reflete uma progressão na complexidade e no objetivo da segmentação. As técnicas de Deep Learning (semântica, de instância, panóptica) representam o estado da arte que transcende as limitações das abordagens clássicas. As abordagens clássicas, como o thresholding e a detecção de bordas, são inerentemente "pixel-based" e muitas vezes resultam em fragmentação ou dependem de limiares fixos, o que as torna menos robustas para cenas complexas. As abordagens baseadas em regiões tentam superar isso agrupando pixels, mas ainda podem ter limitações, como a super-segmentação no algoritmo Watershed.[11, 35] A segmentação semântica e de instância, impulsionadas pelo Deep Learning, são capazes de aprender características de alto nível e contextos complexos, permitindo uma segmentação muito mais precisa e robusta, superando as dificuldades das técnicas clássicas. Esta evolução demonstra a necessidade do Deep Learning para a segmentação de imagens, especialmente para tarefas complexas em Sensoriamento Remoto, como classificação de cobertura do solo e detecção de objetos.

**1.3.2. Técnicas Tradicionais: Thresholding, K-Means e Watershed**

As técnicas tradicionais de segmentação de imagens fornecem métodos fundamentais para isolar objetos de interesse, embora apresentem limitações em cenários complexos.

A **limiarização (thresholding)** é uma técnica fundamental para separar objetos de fundos, transformando imagens em tons de cinza ou coloridas em imagens binárias. A decisão de classificação de um pixel é baseada em um valor de limiar `T`: pixels com intensidade maior que `T` são atribuídos a uma classe (e.g., 1), e os demais a outra (e.g., 0).

  * **Global:** Utiliza um único limiar constante para toda a imagem, eficaz quando as distribuições de intensidade de objeto e fundo são bem distintas. O limiar pode ser determinado iterativamente ou por métodos como o de Otsu, que maximiza a variância entre as classes.
  * **Variável/Local:** O limiar é calculado e ajustado para diferentes partes da imagem, adaptando-se a variações de iluminação ou contraste. Isso pode ser feito dividindo a imagem em sub-regiões e aplicando um limiar global a cada uma, ou calculando o limiar com base nas propriedades do vizinhança de cada pixel (e.g., média e desvio padrão).
  * **Multi-nível:** Segmenta uma imagem em múltiplas regiões usando múltiplos valores de limiar, preservando mais informações da imagem original.
    A limiarização é aplicada em diversas áreas, como análise de documentos, imagens médicas e localização de objetos.

O algoritmo **K-Means** é uma técnica de aprendizado não supervisionado que particiona `n` observações em `k` clusters, onde cada observação é atribuída ao cluster com o centroide (média) mais próximo. O objetivo é minimizar a variância dentro de cada cluster (soma dos quadrados das distâncias intra-cluster - WCSS). O processo iterativo do K-Means envolve:

1.  **Inicialização:** Escolha `k` centroides aleatoriamente no conjunto de dados.
2.  **Atribuição:** Atribua cada ponto de dado ao centroide mais próximo.
3.  **Atualização:** Recalcule os centroides com base na média dos pontos atribuídos a cada cluster.
4.  **Repetição:** Repita os passos 2 e 3 até a convergência (quando as atribuições de cluster não mudam significativamente) ou até um número máximo de iterações.
    Em imagens, o K-Means é frequentemente utilizado para **quantização de cores**, reduzindo o número de cores em uma imagem para fins de compressão, agrupando cores semelhantes em um número limitado de `k` cores. Também pode ser empregado para agrupar pixels com características espectrais semelhantes em imagens multiespectrais.

O algoritmo **Watershed (Bacia Hidrográfica)** é uma técnica de segmentação baseada em morfologia matemática, que trata a imagem como um mapa topográfico. Nele, áreas de alta intensidade são interpretadas como "montanhas" e áreas de baixa intensidade como "bacias". O algoritmo simula a subida da água nas bacias, construindo "barragens" (linhas de watershed) nos pontos onde as bacias se encontram. Essas linhas de barragem representam os limites dos objetos segmentados. As vantagens do Watershed incluem a geração de linhas de borda contínuas e de um pixel de largura, além de ser computacionalmente eficiente. No entanto, sua principal desvantagem é a **super-segmentação**, ou seja, a divisão excessiva da imagem em muitas pequenas regiões, causada por ruído ou texturas finas. Para mitigar a super-segmentação, são empregadas soluções como pré-processamento (e.g., filtros morfológicos, reconstrução de gradiente) e o uso de "marcadores" (marker-based watershed) para guiar a segmentação e suprimir mínimos irrelevantes. O Watershed é aplicado na extração de objetos, em imagens médicas e na análise de componentes conectados.[11, 27]

As técnicas clássicas de segmentação, embora limitadas para cenas complexas, introduzem princípios que são relevantes ou adaptados em pipelines de Deep Learning. A super-segmentação do Watershed, por exemplo, é um problema comum que o Deep Learning pode mitigar, mas o conceito de "marcadores" para guiar a segmentação pode ser análogo ao uso de "prior knowledge" ou "attention mechanisms" em DL. O K-Means, embora não seja uma rede neural, é uma técnica de agrupamento que pode ser usada para pré-processamento de dados (como redução de cores ou agrupamento de bandas espectrais) antes de alimentar uma rede neural, ou mesmo para inicializar pesos em certas arquiteturas. O Thresholding, apesar de simples, pode ser um passo inicial rápido em pipelines de DL para isolar regiões de interesse ou criar máscaras binárias para treinamento. Essas técnicas clássicas não são obsoletas; elas fornecem uma base conceitual e, em alguns casos, podem ser integradas como etapas de pré-processamento ou pós-processamento em pipelines de Deep Learning. A compreensão de suas forças e fraquezas ajuda a justificar a necessidade de modelos mais complexos e a identificar oportunidades para abordagens híbridas.

**Exemplo de Código 1.3.2: Segmentação por Limiarização e K-Means**
Este código demonstra a segmentação de imagens usando limiarização (global e adaptativa) e o algoritmo K-Means para quantização de cores. [C\_5]

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local
from sklearn.cluster import MiniBatchKMeans # MiniBatchKMeans é mais rápido para imagens grandes
from PIL import Image

# Criar uma imagem de teste com gradiente para limiarização
# e um pouco de cor para K-Means
image_test_gray = np.linspace(0, 255, 100*100).reshape((100, 100)).astype(np.uint8)
image_test_gray[40:60, 40:60] = 200 # Adicionar um objeto mais claro

# Criar uma imagem colorida simples para K-Means
image_test_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
image_test_rgb[:50, :, 0] = 255 # Metade superior vermelha
image_test_rgb[50:, :, 1] = 255 # Metade inferior verde
image_test_rgb[20:80, 20:80, 2] = 255 # Quadrado azul no centro

plt.figure(figsize=(18, 6))

# --- Limiarização (Thresholding) ---
# 1. Limiarização Global (Otsu)
thresh_otsu = threshold_otsu(image_test_gray)
binary_otsu = image_test_gray > thresh_otsu

plt.subplot(2, 3, 1)
plt.imshow(image_test_gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_otsu, cmap='gray')
plt.title(f'Limiarização Otsu (T={thresh_otsu:.2f})')
plt.axis('off')

# 2. Limiarização Adaptativa (Local)
# block_size: tamanho da vizinhança para calcular o limiar local
# offset: constante subtraída da média/mediana ponderada
binary_adaptive = image_test_gray > threshold_local(image_test_gray, block_size=31, offset=10)

plt.subplot(2, 3, 3)
plt.imshow(binary_adaptive, cmap='gray')
plt.title('Limiarização Adaptativa (Block 31, Offset 10)')
plt.axis('off')

# --- K-Means para Quantização de Cores ---
# Remodelar a imagem para uma lista de pixels (Altura*Largura, Canais)
# e converter para float para K-Means
image_flat = image_test_rgb.reshape(-1, 3).astype(np.float32)

# Número de clusters (cores) desejado
n_colors = 4
kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0, n_init=10) # n_init para evitar warnings
kmeans.fit(image_flat)

# Obter os centroides dos clusters (as novas cores)
new_colors = kmeans.cluster_centers_.astype(np.uint8)

# Atribuir cada pixel ao seu centroide mais próximo
labels = kmeans.predict(image_flat)
quantized_image_flat = new_colors[labels]

# Remodelar de volta para a forma da imagem original
quantized_image = quantized_image_flat.reshape(image_test_rgb.shape)

plt.subplot(2, 3, 4)
plt.imshow(image_test_rgb)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(quantized_image)
plt.title(f'K-Means Quantização ({n_colors} cores)')
plt.axis('off')

# Exemplo de aplicação de K-Means para segmentação de regiões (simulada)
# Usar K-Means em uma imagem em tons de cinza para agrupar intensidades
image_gray_flat = image_test_gray.reshape(-1, 1).astype(np.float32)
kmeans_gray = MiniBatchKMeans(n_clusters=3, random_state=0, n_init=10) # 3 regiões de intensidade
kmeans_gray.fit(image_gray_flat)
labels_gray = kmeans_gray.predict(image_gray_flat)
segmented_gray = labels_gray.reshape(image_test_gray.shape)

plt.subplot(2, 3, 6)
plt.imshow(segmented_gray, cmap='viridis') # Usar um colormap para visualizar as regiões
plt.title('K-Means Segmentação (3 Regiões)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

-----

