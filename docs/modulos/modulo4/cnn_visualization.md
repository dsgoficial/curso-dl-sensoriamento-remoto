---
sidebar_position: 6
title: "Visualização e Interpretação do Aprendizado da CNN"
description: "Feature Maps, Filtros e Técnicas de Interpretabilidade em CNNs"
tags: [visualização, feature maps, filtros, interpretabilidade, xai, vgg16]
---

# 6. Visualização e Interpretação do Aprendizado da CNN

Uma das características mais valiosas das CNNs é a sua interpretabilidade. Diferentemente de modelos de "caixa preta", o que a CNN aprende pode ser visualizado e analisado. A visualização dos mapas de características e dos filtros da rede fornece uma compreensão profunda de como a CNN processa a informação e toma decisões, um processo central para o campo da "Inteligência Artificial Explicável" (XAI).

## 6.1. O que são e como Visualizar os Feature Maps (Mapas de Características)

Um feature map é o resultado de um filtro convolucional aplicado a uma camada de entrada.¹¹ Ele serve como uma representação visual de onde uma característica específica, como uma borda ou uma textura, foi detectada na imagem de entrada.¹⁰ Visualizar os mapas de características em diferentes profundidades da rede revela a natureza hierárquica do aprendizado da CNN.¹¹

### Hierarquia das Características

- **Camadas Iniciais**: Os feature maps das camadas iniciais ativam-se para características simples e de baixo nível, como arestas, cantos e gradientes de cor. Eles mostram a resposta da rede a elementos visuais básicos.

- **Camadas Intermediárias**: À medida que os dados avançam, os mapas de características das camadas intermediárias começam a combinar as características simples para formar padrões mais complexos, como texturas ou partes de objetos, como o contorno de uma roda ou a forma de um olho.

- **Camadas Profundas**: Os mapas de características nas camadas mais profundas representam conceitos altamente abstratos e complexos, ativando-se para objetos inteiros, como um carro, um rosto ou um cachorro.¹¹

Essa capacidade de visualizar o que a rede "está olhando" para tomar uma decisão é crucial para a interpretabilidade do modelo.¹¹ Ao examinar quais partes de uma imagem ativam fortemente mapas de características específicos, os desenvolvedores podem depurar o modelo, verificar se ele está se concentrando em informações relevantes e, em última análise, aumentar sua precisão e confiabilidade.¹¹

## 6.2. Visualização Prática dos Feature Maps com PyTorch

Para visualizar os feature maps de uma rede, podemos utilizar um modelo pré-treinado, como o VGG16, disponível na biblioteca torchvision.³¹ O VGG16 foi treinado no dataset ImageNet, que contém milhões de imagens e 1000 categorias, o que faz com que a rede tenha aprendido a detectar uma vasta gama de características.³¹

A visualização envolve os seguintes passos:

1. Carregar um modelo pré-treinado (torchvision.models.vgg16(pretrained=True)).
2. Extrair as camadas convolucionais do modelo.
3. Carregar e pré-processar uma imagem de entrada para que ela tenha as dimensões e normalização esperadas pelo modelo.
4. Realizar o forward pass camada a camada, capturando a saída de cada camada convolucional.
5. Visualizar a saída de cada camada (os feature maps) usando bibliotecas como matplotlib.³¹

### Exemplo de Código para Visualização de Feature Maps

```python
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# 1. Carregar um modelo VGG16 pré-treinado
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()  # Coloca o modelo em modo de avaliação

# 2. Definir as transformações da imagem
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Carregar uma imagem de exemplo
image_path = 'https://upload.wikimedia.org/wikipedia/commons/b/b3/Mops_oct09_cropped.jpg'
response = requests.get(image_path)
image = Image.open(BytesIO(response.content)).convert('RGB')
input_tensor = img_transforms(image).unsqueeze(0)  # Adiciona a dimensão do batch

# 4. Extrair os feature maps de cada camada convolucional
feature_maps = []
layer_names = []

# Percorre as camadas do modelo
for name, layer in model.features.named_children():
    if isinstance(layer, torch.nn.Conv2d):
        input_tensor = layer(input_tensor)
        feature_maps.append(input_tensor)
        layer_names.append(name)

# 5. Visualizar os feature maps
for i, feature_map_tensor in enumerate(feature_maps):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    for j in range(6):
        # Move o tensor para a CPU e converte para array numpy
        activation = feature_map_tensor.squeeze(0)[j].detach().cpu().numpy()
        axes[j].imshow(activation, cmap='viridis')
        axes[j].axis('off')
    fig.suptitle(f'Feature Maps da Camada: {layer_names[i]}')
    plt.show()
```

Este código mostra como as ativações mudam de camada para camada. Nas camadas iniciais, você veria padrões de alto contraste, como bordas. Nas camadas mais profundas, os padrões se tornariam mais complexos e abstratos, mostrando as representações de alto nível que a rede aprendeu para classificar a imagem.³¹

## 6.3. Visualizando os Filtros Aprendidos com PyTorch

Além de visualizar os mapas de características, os próprios filtros (os kernels com pesos aprendidos) podem ser visualizados como pequenas imagens.³⁴ A visualização de um filtro revela a característica específica que ele foi treinado para detectar. Um filtro pode se parecer com um detector de borda vertical, outro com um detector de borda horizontal, e outro pode ter a aparência de um círculo ou de um gradiente de cor. Essa capacidade da CNN de "aprender" automaticamente os detectores de características²² é o que a diferencia de abordagens mais antigas de visão computacional que exigiam a criação manual desses filtros. A visualização dos filtros confirma que a rede não está apenas memorizando dados, mas está aprendendo a extrair representações visuais significativas da imagem.³⁵

Para visualizar os filtros de uma camada convolucional em um modelo PyTorch, você pode acessar diretamente os pesos (weight) da camada.³⁶ Os pesos são tensores que podem ser convertidos para um formato de imagem e plotados com bibliotecas como o matplotlib. No caso de múltiplas camadas de entrada, é possível visualizar cada canal em um subplot separado.

### Exemplo de Código para Visualização de Filtros

```python
import torch
from torchvision import models
import matplotlib.pyplot as plt

# 1. Carregar um modelo pré-treinado VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# 2. Acessar a primeira camada convolucional
first_conv_layer = model.features[0]

# 3. Extrair os pesos do filtro
# shape: (out_channels, in_channels, kernel_height, kernel_width)
filters_tensor = first_conv_layer.weight.data.clone().detach().cpu()

# 4. Visualizar os primeiros 8 filtros
fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i in range(8):
    # Pega o i-ésimo filtro (com todos os canais de entrada)
    # Normaliza os valores para visualização
    filter_i = filters_tensor[i]
    filter_i = (filter_i - filter_i.min()) / (filter_i.max() - filter_i.min())
    
    # Transpõe para (height, width, channels) para o imshow
    filter_i = filter_i.permute(1, 2, 0)
    
    axes[i].imshow(filter_i)
    axes[i].set_title(f'Filtro {i+1}')
    axes[i].axis('off')

plt.suptitle('Filtros Aprendidos da Primeira Camada Convolucional do VGG16')
plt.show()
```

Este código acessa a primeira camada convolucional do modelo VGG16, extrai os pesos dos seus filtros e os plota. Nas primeiras camadas, os filtros geralmente se parecem com detectores de bordas e cores.³⁴
