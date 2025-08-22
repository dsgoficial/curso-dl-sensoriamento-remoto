---
sidebar_position: 2
title: "A Arquitetura das CNNs"
description: "Os Blocos de Construção de um Modelo de Visão Computacional"
tags: [cnn, arquitetura, convolução, pooling, filtros, feature maps, pytorch]
---

# 2. A Arquitetura das CNNs: Os Blocos de Construção de um Modelo de Visão

As Redes Neurais Convolucionais foram concebidas para superar as limitações do MLP, introduzindo uma arquitetura especializada que respeita e explora a estrutura multidimensional dos dados de imagem. A CNN opera como um sistema hierárquico, onde cada camada é responsável por uma sub-tarefa específica.⁸ A estrutura mais comum de uma CNN é composta por três tipos principais de camadas: a camada convolucional, a camada de pooling e a camada totalmente conectada, cada uma com um papel distinto e interligado para processar e classificar dados visuais.

## 2.1. A Camada Convolucional: O Coração da Extração Hierárquica de Características

A camada convolucional é o componente central e o bloco de construção fundamental de uma CNN, onde a maioria dos cálculos ocorre. Seu objetivo principal é extrair características de forma hierárquica, progredindo da detecção de traços simples, como bordas e texturas, para a identificação de representações mais complexas, como formas e objetos inteiros.

O mecanismo de operação baseia-se em um filtro (também chamado de kernel), que é uma pequena matriz de pesos aprendidos. Este filtro percorre (ou "convolui") a imagem de entrada, pixel por pixel ou em saltos definidos, realizando uma operação de produto escalar entre os valores de sua matriz e a sub-região da imagem que ele cobre. O resultado de cada operação é um único valor na matriz de saída, chamada de mapa de características (feature map). Este mapa de características é uma representação visual de onde uma característica específica, que o filtro foi treinado para detectar, está presente na imagem de entrada.

O uso de filtros compartilhados em toda a imagem⁴ é a principal inovação que resolve a explosão de parâmetros dos MLPs. Em vez de cada neurônio ter seus próprios pesos individuais, como ocorre em uma camada totalmente conectada, um único conjunto de pesos (o filtro) é usado repetidamente para escanear a imagem inteira. Esta abordagem não apenas reduz drasticamente o número de parâmetros, tornando a rede muito mais eficiente e escalável para imagens de alta resolução, mas também confere à rede a propriedade de invariância à translação (translation invariance). Isso significa que a rede é capaz de reconhecer um objeto ou característica, como um olho ou uma borda, independentemente de sua posição na imagem de entrada. A rede não precisa ser treinada em todas as posições possíveis de um objeto, pois o filtro compartilhado é capaz de detectá-lo em qualquer lugar.

### Exemplo de Convolução com PyTorch

A função nn.Conv2d do PyTorch é a ferramenta ideal para realizar a convolução em tensores de imagem. Ela permite especificar os canais de entrada (in_channels), os canais de saída (out_channels, que correspondem ao número de filtros), o tamanho do filtro (kernel_size), o passo (stride) e o preenchimento (padding).

```python
import torch
import torch.nn as nn

# Simular uma entrada de imagem com 1 canal de cor (escala de cinza), altura de 8 e largura de 8
# Formato: (batch_size, channels, height, width)
input_tensor = torch.randn(1, 1, 8, 8)

# Definir a camada de convolução: 1 canal de entrada, 16 filtros de saída, kernel de 3x3
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)

# Realizar a operação de convolução
output_tensor = conv_layer(input_tensor)

# Imprimir as dimensões da saída
print(f"Dimensões da entrada: {input_tensor.shape}")
print(f"Dimensões da saída após a convolução: {output_tensor.shape}")

# Saída esperada: Dimensões da entrada: torch.Size([1, 1, 8, 8])
# Saída esperada: Dimensões da saída após a convolução: torch.Size([1, 16, 6, 6])
```

Neste exemplo, o kernel_size=3 por padrão tem um stride de 1 e padding de 0. A dimensão da saída é calculada por (I - K + 1), onde I é o tamanho da entrada (8) e K é o tamanho do kernel (3), resultando em (8 - 3 + 1) = 6 para a altura e largura. A dimensão de profundidade (canais) da saída é 16, correspondente ao número de filtros aplicados.

## 2.2. A Camada de Pooling: Redução de Dimensionalidade e Aumento de Eficiência

Posicionada tipicamente após uma ou mais camadas convolucionais, a camada de pooling (ou subamostragem) tem como principal objetivo reduzir a dimensionalidade espacial dos mapas de características. Essa redução, também conhecida como downsampling, ajuda a diminuir a carga computacional, o número de parâmetros e o risco de superajuste (overfitting).

O mecanismo de operação de pooling envolve um filtro que percorre a entrada e aplica uma função de agregação aos valores de cada sub-região que ele cobre. Diferentemente da camada convolucional, o filtro de pooling não possui pesos aprendidos. Os dois tipos mais comuns são:

- **Max Pooling**: Seleciona o valor máximo de cada sub-região para preencher a matriz de saída. Esta é a técnica mais utilizada, pois preserva as características mais proeminentes (aquelas que geraram a maior ativação), tornando a representação mais robusta a pequenas variações e distorções na imagem.

- **Average Pooling**: Calcula a média dos valores de cada sub-região para a matriz de saída.

Embora a camada de pooling resulte em uma "perda de informação", essa perda é controlada e, em muitos casos, desejável. O objetivo é manter apenas as características mais discriminativas e relevantes para a tarefa de classificação, descartando o restante e consolidando a representação da imagem.

### Exemplo Visual de Max e Average Pooling

Considere uma matriz de entrada 4x4. A operação de pooling com um filtro 2x2 e um stride de 2 será aplicada a esta matriz.

**Matriz de Entrada (4x4):**
```
[[1, 3, 2, 4],
 [5, 2, 6, 1],
 [3, 1, 7, 5],
 [8, 4, 3, 2]]
```

**Max Pooling (filtro 2x2, stride 2):** O filtro desliza pela matriz, e em cada janela 2x2, o valor máximo é selecionado.

```
[[1, 3], -> max(1, 3, 5, 2) = 5
 [5, 2]]

[[2, 4], -> max(2, 4, 6, 1) = 6
 [6, 1]]

[[3, 1], -> max(3, 1, 8, 4) = 8
 [8, 4]]

[[7, 5], -> max(7, 5, 3, 2) = 7
 [3, 2]]
```

**Saída do Max Pooling (2x2):**
```
[[5, 6],
 [8, 7]]
```

**Average Pooling (filtro 2x2, stride 2):** O filtro desliza, e em cada janela 2x2, a média dos valores é calculada.

**Saída do Average Pooling (2x2):**
```
[[2.75, 3.25],
 [4.0, 4.25]]
```

O Max Pooling é geralmente preferido, pois tende a reter as características mais proeminentes, ou seja, aquelas que geraram a maior ativação na camada convolucional, tornando a representação mais robusta a pequenas variações e distorções na imagem. O Average Pooling, por outro lado, mantém mais informações sobre os elementos menos importantes, misturando-os, o que pode resultar em uma saída menos extrema.¹⁵

### Exemplo de Pooling em PyTorch

A função nn.MaxPool2d é usada para aplicar o Max Pooling, enquanto nn.AvgPool2d é usada para o Average Pooling. Ambas aceitam parâmetros como kernel_size e stride para controlar a operação.

```python
import torch
import torch.nn as nn

# Simular um mapa de características de entrada com 1 canal, 4x4
feature_map = torch.tensor([[[1., 3., 2., 4.],
                            [5., 2., 6., 1.],
                            [3., 1., 7., 5.],
                            [8., 4., 3., 2.]]])

# Max Pooling com kernel de 2x2 e stride de 2
max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
output_max_pool = max_pool_layer(feature_map)
print(f"Saída do Max Pooling: {output_max_pool.shape}")
print(output_max_pool)

# Average Pooling com kernel de 2x2 e stride de 2
avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg_pool = avg_pool_layer(feature_map)
print(f"\nSaída do Average Pooling: {output_avg_pool.shape}")
print(output_avg_pool)
```

## 2.3. A Camada Totalmente Conectada: A Classificação Baseada em Características

A camada totalmente conectada (fully connected, FC) é tipicamente a última camada de uma CNN. Seu papel é integrar as características de alto nível extraídas e refinadas pelas camadas anteriores (convolucional e pooling) para realizar a previsão ou classificação final. A entrada para esta camada é um vetor achatado (flattened) dos mapas de características finais, transformando a saída multidimensional das camadas convolucionais em um formato compatível com uma rede neural tradicional.

Nesta camada, cada neurônio se conecta a todos os neurônios da camada anterior, operando de forma semelhante a um MLP tradicional. Enquanto as camadas convolucionais e de pooling geralmente utilizam a função de ativação ReLU, a camada FC comumente emprega a função de ativação Softmax, especialmente para problemas de classificação. A função Softmax converte as saídas dos neurônios em uma distribuição de probabilidade, produzindo um valor entre 0 e 1 para cada classe de saída, indicando a probabilidade de a entrada pertencer a essa classe.

A CNN opera em uma hierarquia clara: as camadas convolucionais iniciais se concentram em características simples como cores e bordas, enquanto as camadas mais profundas combinam essas características para reconhecer elementos ou formas maiores, como partes de um objeto. A camada FC final utiliza essa representação abstrata e de alto nível para tomar a decisão de classificação.

## 2.4. Funções de Ativação: O Papel da Não-Linearidade

Uma parte crucial da arquitetura de uma CNN é a aplicação de funções de ativação. Elas são responsáveis por introduzir a não-linearidade na rede, permitindo que o modelo aprenda relações complexas e não-lineares entre as características da imagem. A função de ativação mais comum após uma operação de convolução é a ReLU (Unidade Linear Retificada), que simplesmente define todos os valores negativos como zero. A aplicação de uma função de ativação é feita logo após a operação de convolução, garantindo que a rede possa aprender e identificar padrões complexos. As camadas de pooling e de flattening não utilizam funções de ativação, pois suas funções são estritamente de redução de dimensionalidade e reestruturação de dados, respectivamente, e não requerem não-linearidades.³⁸ No final da rede, nas camadas totalmente conectadas, uma função como a Softmax é tipicamente usada para converter as saídas em uma distribuição de probabilidade para a classificação final.