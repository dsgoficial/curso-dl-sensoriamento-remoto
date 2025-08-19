---
sidebar_position: 7
title: "Fundamentos de Processamento de Imagens"
description: "Conceitos básicos de processamento digital de imagens, filtros e convolução com NumPy"
tags: [processamento-imagens, numpy, filtros, convolução, segmentação]
---

# Processamento de Imagens

Este módulo estabelece as bases essenciais para a compreensão e aplicação de Deep Learning em tarefas de Visão Computacional, com foco particular no Sensoriamento Remoto. Iniciaremos com os fundamentos do processamento digital de imagens, compreendendo como as imagens são representadas e manipuladas computacionalmente usando a biblioteca NumPy. Em seguida, faremos uma transição para os conceitos fundamentais de Redes Neurais Artificiais (RNAs), culminando na introdução prática e teórica do PyTorch, a principal ferramenta que utilizaremos para construir e treinar modelos de Deep Learning. Ao final deste módulo, os alunos terão uma base sólida tanto em processamento de imagens clássico quanto nos pilares das redes neurais e do framework PyTorch, preparando-os para módulos mais avançados em Redes Convolucionais e aplicações específicas em Sensoriamento Remoto.

## Seção 1: Fundamentos de Processamento de Imagens com NumPy

Esta seção explora as representações digitais de imagens e as operações fundamentais para manipulá-las, utilizando o NumPy como ferramenta principal. A compreensão desses conceitos é crucial, pois as imagens são o principal tipo de dado em Deep Learning aplicado à Visão Computacional.

### 1.1. Representação e Manipulação Básica de Imagens

**1.1.1. Imagens como Matrizes NumPy: Grayscale, RGB e Multiespectral**

Imagens digitais são, em sua essência, arrays de valores de pixels. A biblioteca NumPy, com sua capacidade de manipular arrays multidimensionais (`ndarray`), é a ferramenta fundamental para representar e processar essas imagens.[1, 2] A forma como uma imagem é representada em um array NumPy depende de seu tipo, que pode variar de tons de cinza a imagens multiespectrais complexas.

As **imagens em tons de cinza (grayscale)** são as mais simples, contendo apenas informações de intensidade, sem cor.[1] Cada pixel na imagem é representado por um único valor, que tipicamente varia de 0 (preto) a 255 (branco).[1] Em NumPy, essas imagens são armazenadas como arrays 2D, com a forma `(altura, largura)`.[1, 2] Sua simplicidade as torna eficientes para muitas aplicações práticas de processamento de imagem.[1]

As **imagens RGB (coloridas)** são compostas por três canais de cor: Vermelho (Red), Verde (Green) e Azul (Blue).[1] Cada pixel é representado por um trio de valores, um para cada canal de cor, geralmente também variando de 0 a 255.[1] Essas imagens são armazenadas como arrays NumPy 3D, com a forma `(altura, largura, 3)`.[1, 2] As imagens RGB são o formato padrão para o armazenamento e processamento de cores em dispositivos digitais e são amplamente utilizadas em tarefas como segmentação baseada em cores, reconhecimento facial, detecção de objetos e redes neurais convolucionais (CNNs).[1]

As **imagens multiespectrais** representam uma extensão das imagens RGB, contendo dezenas, centenas ou até milhares de canais.[1] Elas são representadas como arrays NumPy com a forma `(altura, largura, canais)`, onde o número de canais (`N_Canais`) é maior que 3.[1, 3] Cada canal corresponde a uma banda espectral diferente ou a uma leitura de sensor específica, capturando informações que se estendem além do espectro visível (por exemplo, infravermelho ou ultravioleta).[1, 3] No contexto do Sensoriamento Remoto, as imagens multiespectrais são de importância crucial, pois cada banda espectral fornece informações únicas sobre a superfície terrestre, permitindo análises aprofundadas de características como vegetação, água e tipos de solo.[1, 3]

Os **tipos de dados** para os valores de pixels são frequentemente inteiros sem sinal de 8 bits (`np.uint8`), que podem armazenar valores de 0 a 255.[1, 4] Este tipo de dado é eficiente em termos de memória e é o formato padrão para muitas imagens. No entanto, para operações matemáticas mais complexas, como convoluções ou cálculos de gradientes em redes neurais, ou para uso em algoritmos de Deep Learning, é comum converter esses valores para tipos de ponto flutuante (por exemplo, `float32`, `float64`). Essa conversão é frequentemente acompanhada pela **normalização** dos valores de pixel para um intervalo específico, como [0,1] ou [-1, 1], o que ajuda a estabilizar o treinamento de modelos de Deep Learning e a reduzir o impacto de variações de iluminação.[5, 6]

Em relação às **convenções de coordenadas**, em NumPy e em bibliotecas como `scikit-image`, as imagens 2D (tons de cinza) são indexadas por `(linha, coluna)` ou `(r, c)`, com a origem `(0, 0)` localizada no canto superior esquerdo.[2, 8, 9] Esta convenção é análoga à indexação de matrizes em álgebra linear. Para imagens multicanais, a dimensão do canal é tipicamente a última dimensão, seguindo o formato `(linha, coluna, canal)`, embora algumas funções permitam especificar a posição do canal através de um argumento `channel_axis`.[2, 8, 9] Esta convenção difere das coordenadas Cartesianas `(x, y)` tradicionais, onde `x` é horizontal, `y` é vertical e a origem é frequentemente no canto inferior esquerdo.[2, 8, 9]

A tabela a seguir sumariza as principais características de cada tipo de representação de imagem:

**Tabela: Comparativo de Representação de Imagens (Grayscale, RGB, Multiespectral)**

| Característica | Imagem Grayscale | Imagem RGB | Imagem Multiespectral |
| :-------------------- | :------------------------ | :-------------------------- | :---------------------------------- |
| **Dimensão NumPy** | `(Altura, Largura)` | `(Altura, Largura, 3)` | `(Altura, Largura, N_Canais)` |
| **Canais** | 1 (Intensidade) | 3 (Vermelho, Verde, Azul) | N_Canais (Ex: Visível, Infravermelho, etc.) |
| **Valores de Pixel** | 0-255 (Intensidade) | 0-255 por canal | 0-255 (ou outros ranges, dependendo do sensor) por canal |
| **Uso Comum** | Processamento básico, detecção de bordas | Fotografia, Visão Computacional geral | Sensoriamento Remoto, Imagem Médica, Análise de Materiais |
| **Exemplo de Aplicação** | Filtragem de ruído, binarização | Classificação de objetos, reconhecimento facial | Classificação de uso do solo, detecção de mudanças, análise de vegetação |

**Exemplo de Código 1.1.1: Criação e Manipulação Básica de Imagens com NumPy**

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
# Convertendo para float32 e normalizando para o intervalo [0,1]
imagem_rgb_float = imagem_rgb.astype(np.float32) / 255.0
print(f"\nTipo de dado da imagem RGB (float): {imagem_rgb_float.dtype}")
print(f"Valor mínimo após normalização [0,1]: {imagem_rgb_float.min()}")
print(f"Valor máximo após normalização [0,1]: {imagem_rgb_float.max()}")

# Normalização para o intervalo [-1, 1]
imagem_rgb_normalizada_neg1_1 = (imagem_rgb.astype(np.float32) / 127.5) - 1.0
print(f"Valor mínimo após normalização [-1, 1]: {imagem_rgb_normalizada_neg1_1.min()}")
print(f"Valor máximo após normalização [-1, 1]: {imagem_rgb_normalizada_neg1_1.max()}")

plt.figure(figsize=(6, 3))
plt.imshow(imagem_rgb_float) # imshow pode lidar com floats em [0,1]
plt.title('Imagem RGB Normalizada [0,1]')
plt.axis('off')
plt.show()
```

### 1.1.2. Operações Fundamentais com NumPy: Slicing, Reshape e Normalização

A manipulação eficiente de arrays NumPy é essencial para o processamento de imagens em Deep Learning. Operações como fatiamento (slicing), remodelagem (reshape) e normalização são cruciais para preparar e transformar os dados de imagem.

O **fatiamento (slicing)** em NumPy estende o conceito básico de fatiamento do Python para arrays N-dimensionais, permitindo a seleção, extração e modificação de regiões ou componentes específicos de arrays de imagem.[10, 11] A sintaxe básica é `array[start:stop:step]` aplicada a cada dimensão.

**Exemplo de Código 1.1.2: Operações Fundamentais com NumPy**

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

# Min-Max Scaling para [0,1]
min_val = imagem_float.min()
max_val = imagem_float.max()
imagem_normalizada_0_1 = (imagem_float - min_val) / (max_val - min_val)
print(f"\nMin-Max Normalização [0,1] - Min: {imagem_normalizada_0_1.min()}, Max: {imagem_normalizada_0_1.max()}")

# Padronização (Standardization) para média 0 e desvio padrão 1
media = imagem_float.mean()
desvio_padrao = imagem_float.std()
imagem_padronizada = (imagem_float - media) / desvio_padrao
print(f"Padronização - Média: {imagem_padronizada.mean():.4f}, Desvio Padrão: {imagem_padronizada.std():.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagem_normalizada_0_1)
plt.title('Imagem Normalizada [0,1]')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagem_padronizada[:,:,0], cmap='gray') # Exibindo apenas um canal para visualização
plt.title('Imagem Padronizada (Canal 0)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

## 1.2. Filtros Espaciais e Convolução

Esta seção introduz os conceitos de filtros espaciais e a operação de convolução, que são fundamentais tanto para o processamento de imagens clássico quanto para as Redes Neurais Convolucionais (CNNs).

### 1.2.1. Conceitos de Convolução 2D: Sliding Window e Padding

A **convolução 2D** é uma operação que atua sobre duas "sinais" bidimensionais: uma imagem de entrada e um "kernel" (ou filtro), produzindo uma terceira imagem de saída.[16, 17, 18] O funcionamento intuitivo da convolução envolve a ideia de uma "janela deslizante" (sliding window): o kernel, que é uma pequena matriz de pesos, desliza sobre a imagem de entrada. Em cada posição, os valores correspondentes da porção da imagem sob a janela e do kernel são multiplicados elemento a elemento e, em seguida, somados. O resultado dessa soma ponderada se torna o novo valor do pixel na imagem de saída.

**Exemplo de Código 1.2.1: Convolução 2D Simples (do zero)**

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

    # Inicializar o array de saída
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Realizar a convolução
    for r in range(output_height):
        for c in range(output_width):
            # Extrair a região da imagem coberta pelo kernel usando slicing
            region = image_padded[r * stride : r * stride + kernel_height,
                                  c * stride : c * stride + kernel_width]
            # Multiplicação elemento a elemento e soma
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

# Visualização
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_valid.astype(np.uint8), cmap='gray')
plt.title('Convolução (Valid)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_same.astype(np.uint8), cmap='gray')
plt.title('Convolução (Same)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### 1.2.2. Filtros Clássicos: Média, Gaussiano, Sobel e Laplaciano

Os filtros espaciais são ferramentas essenciais no processamento de imagens para tarefas como suavização, realce e detecção de bordas. Cada filtro possui um propósito e um mecanismo de funcionamento distintos.

**Exemplo de Código 1.2.2: Aplicação de Filtros Clássicos**

```python
import numpy as np
import matplotlib.pyplot as plt

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

# Converter para float32 para operações de filtragem
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
# Normalizar para exibição
sobel_magnitude_display = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
print("Operador Sobel aplicado.")

# --- 4. Filtro Laplaciano (Laplacian Filter) ---
print("Aplicando Filtro Laplaciano...")
laplacian_kernel = np.array([[ 0,  1,  0],
                             [ 1, -4,  1],
                             [ 0,  1,  0]], dtype=np.float32)
image_laplacian = convolve2D_from_scratch(image_float, laplacian_kernel, padding='same')
# Normalizar para exibição
image_laplacian_display = np.abs(image_laplacian)
image_laplacian_display = (image_laplacian_display / image_laplacian_display.max() * 255).astype(np.uint8)
print("Filtro Laplaciano aplicado.")

# Exibir resultados
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

images = [image_original, image_mean_filtered, image_gaussian_filtered, 
          sobel_magnitude_display, image_laplacian_display]
titles = ['Original com Ruído', 'Filtro da Média', 'Filtro Gaussiano', 
          'Operador Sobel (Mag)', 'Filtro Laplaciano']

for i, (img, title) in enumerate(zip(images, titles)):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(title)
    axes[i].axis('off')

# Remove the last empty subplot
axes[-1].remove()

plt.tight_layout()
plt.show()
```
