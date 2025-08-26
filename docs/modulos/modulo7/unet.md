---
sidebar_position: 4
title: "U-Net: Precisão em Segmentação Médica e Sensoriamento Remoto"
description: "Arquitetura U-Net com conexões de salto para segmentação precisa em PyTorch"
tags: [unet, segmentação, conexões-de-salto, deep-learning, pytorch]
---

# A Arquitetura U-Net para Segmentação Semântica e sua Implementação em PyTorch

## Seção 1: Introdução à Segmentação Semântica e a Ascensão da U-Net

A segmentação semântica representa uma das tarefas mais fundamentais e desafiadoras no campo da visão computacional. Diferentemente da classificação de imagens, que atribui um único rótulo a uma imagem inteira, ou da detecção de objetos, que identifica e localiza objetos com caixas delimitadoras (bounding boxes), a segmentação semântica eleva o nível de granularidade ao classificar cada pixel de uma imagem com um rótulo de classe.¹ O resultado dessa tarefa é um mapa de segmentação, que essencialmente recria a imagem original com cada pixel codificado por cor para representar a sua classe semântica, formando máscaras de segmentação.¹ Essa precisão pixel a pixel é crucial para uma ampla gama de aplicações de alta relevância, como a análise de imagens médicas², onde a identificação de tumores ou órgãos é vital, em veículos autônomos, para diferenciar a estrada de pedestres e outros veículos, e no sensoriamento remoto, para mapear o uso do solo com alta precisão.³

A U-Net emergiu como uma arquitetura seminal para a segmentação de imagens em 2015, proposta por Olaf Ronneberger, Philipp Fischer e Thomas Brox em seu artigo intitulado "U-Net: Convolutional Networks for Biomedical Image Segmentation".⁵ O modelo foi desenvolvido com um foco específico no campo biomédico, que enfrenta um desafio único: a escassez de imagens anotadas.² A U-Net foi projetada para ser treinada de ponta a ponta a partir de um número muito limitado de imagens, superando os métodos mais avançados da época, como as redes convolucionais de "sliding-window".⁵ Sua arquitetura inovadora, baseada nas Fully Convolutional Networks (FCNs) mas com modificações e extensões cruciais, permitiu que ela alcançasse um desempenho superior, estabelecendo-se rapidamente como um padrão-ouro para a segmentação de imagens.⁶

## Seção 2: A Arquitetura U-Net em Profundidade: Teoria e Estrutura

A arquitetura da U-Net é notável por sua estrutura simétrica em forma de "U", que é composta por duas partes principais: um caminho contrativo, ou codificador (encoder), e um caminho expansivo, ou decodificador (decoder).² Essa configuração permite que a rede capture o contexto de alto nível e, em seguida, use essa informação para realizar a localização precisa em nível de pixel.

### 2.1. A Estrutura em "U": Caminho Contrativo (Encoder) e Caminho Expansivo (Decoder)

O **caminho contrativo** é uma rede convolucional típica, que opera de maneira semelhante a um extrator de características. Ele é composto por uma sucessão de blocos, onde cada bloco aplica duas camadas convolucionais de 3x3 (cada uma seguida por uma ativação ReLU) e, em seguida, uma operação de max pooling de 2x2 com um stride de 2 para o downsampling.² Com cada etapa de downsampling, a resolução espacial da imagem é reduzida pela metade, enquanto o número de canais de características é dobrado. Esse processo permite que a rede capture informações contextuais de alto nível e representações de características cada vez mais abstratas, embora à custa da perda de detalhes espaciais finos.⁶

O **caminho expansivo**, por sua vez, é projetado para restaurar a resolução espacial do mapa de características, permitindo uma segmentação precisa. Cada etapa neste caminho começa com uma operação de upsampling, que pode ser uma convolução transposta⁹ ou "up-convolution", que dobra as dimensões espaciais do mapa de características e reduz pela metade o número de canais. Após essa operação, o mapa de características upsampled é concatenado com o mapa de características correspondente do caminho contrativo.⁶ Essa concatenação é um passo crítico para a precisão da rede e é seguida por duas camadas convolucionais de 3x3 com ativações ReLU, que aprendem a sintetizar as informações combinadas. A resolução do mapa de características continua a ser restaurada até que a saída final tenha as mesmas dimensões espaciais da imagem de entrada. A camada final é uma convolução de 1x1 que mapeia o número de canais de volta para o número de classes a serem segmentadas.¹¹

### 2.2. A Essência da U-Net: As Conexões de Salto (Skip Connections)

As conexões de salto são a inovação central e o recurso mais definidor da U-Net, sendo cruciais para o seu sucesso em tarefas de segmentação de imagens.⁵ Elas atuam como atalhos de alta resolução, conectando diretamente os mapas de características do caminho contrativo aos mapas de características correspondentes no caminho expansivo.⁷ O mecanismo por trás dessas conexões é a concatenação, onde as características de alta resolução do codificador são fundidas com as características upsampled do decodificador.¹⁰

A principal função dessas conexões é combater a perda de informação espacial que é inerente às operações de pooling e downsampling.¹² Sem as conexões de salto, o decodificador teria que reconstruir a imagem apenas a partir dos mapas de características de baixa resolução e alto nível semântico, resultando em limites de objeto imprecisos e borrados.¹³ Ao fornecer ao decodificador as características de alta resolução do codificador, as conexões de salto permitem que o modelo reconstrua com precisão as localizações e limites de objetos, o que é vital para uma segmentação de qualidade.¹⁰

A arquitetura das conexões de salto na U-Net, que utiliza a concatenação em vez de uma fusão mais simples, fornece uma "orientação" de localização explícita para o processo de upsampling. Essa fusão de informações de alto nível (contexto) e baixo nível (detalhe espacial) é a razão direta pela qual a U-Net se tornou tão eficaz em tarefas que exigem limites de objeto precisos. Além disso, as conexões de salto também criam um caminho de fluxo de gradiente mais curto durante o treinamento com retropropagação. Isso ajuda a mitigar o problema do gradiente evanescente e permite o treinamento de redes mais profundas, uma vantagem herdada de arquiteturas como as redes residuais.¹²

### 2.3. Componentes-Chave e Estratégias de Treinamento

Os blocos de construção da U-Net são as duplas convoluções de 3x3, cada uma seguida por uma função de ativação ReLU. Uma particularidade do trabalho original é que as convoluções não usavam padding, o que resultava em uma saída com dimensões espaciais menores que a entrada.⁸ Na prática moderna, a implementação da U-Net frequentemente usa padding para manter as dimensões do mapa de características consistentes, garantindo que a saída final tenha o mesmo tamanho que a imagem de entrada.¹¹ Essa adaptação simplifica o pipeline e elimina a necessidade de pós-processamento, tornando a implementação mais prática e didática.

A U-Net foi desenvolvida para ser eficiente com poucos dados de treinamento, um requisito crucial para a segmentação biomédica.² Para contornar a limitação de dados anotados, a arquitetura faz uso extensivo de técnicas de **data augmentation**.² Estratégias como rotação, zoom, espelhamento, adição de ruído gaussiano e ajustes de brilho e contraste são aplicadas aleatoriamente aos dados de entrada. Essas transformações sintéticas expandem o conjunto de treinamento, aumentando a robustez do modelo e reduzindo o risco de overfitting.² Além disso, para processar imagens de alta resolução que não cabem na memória da GPU, a U-Net utiliza uma **estratégia de sobreposição de "tiles"** (Overlap-Tile Strategy). A imagem é dividida em blocos menores com uma região de sobreposição, o que garante a continuidade da segmentação e previne imprecisões nas bordas dos tiles.²

## Seção 3: Implementação da U-Net do Zero em PyTorch

A implementação de uma arquitetura como a U-Net em uma biblioteca de aprendizado profundo como o PyTorch segue um fluxo de trabalho padronizado, que pode ser dividido em etapas claras: 1) Preparação e carregamento dos dados, 2) Definição da arquitetura do modelo, 3) Configuração da função de perda e do otimizador, 4) A execução do loop de treinamento e, finalmente, 5) A avaliação do modelo e a inferência.¹⁵ Abaixo, é apresentada a construção modular do modelo em PyTorch.

### 3.1. Construção do Modelo: Código e Explicações

A arquitetura pode ser construída de forma modular, com cada componente encapsulado em classes e funções reutilizáveis.

#### 3.1.1. O Bloco Convolucional Duplo (double_conv)

O bloco fundamental da U-Net é uma sequência de duas camadas convolucionais de 3x3, cada uma seguida por uma ativação ReLU. Para evitar a repetição de código, esse bloco pode ser encapsulado em uma função ou classe, uma prática que aumenta a clareza e a manutenibilidade do código.¹¹

```python
import torch
import torch.nn as nn

def double_convolution(in_channels, out_channels):
    """
    Função para criar um bloco de dupla convolução.
    Nesta implementação, o padding é usado para manter as dimensões espaciais.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op
```

Essa função usa `padding=1` nas camadas de convolução. Esta é uma escolha de engenharia importante, pois assegura que as dimensões espaciais do mapa de características de saída sejam idênticas às da entrada, eliminando a necessidade de pós-processamento ou recorte.¹¹ O `inplace=True` na ativação ReLU otimiza o uso de memória ao modificar o tensor de entrada no local.

#### 3.1.2. A Classe Completa UNet (nn.Module)

A arquitetura completa é definida em uma classe que herda de `nn.Module`. Os métodos `__init__` e `forward` constroem e definem o fluxo de dados da rede, respectivamente.

```python
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Caminho Contrativo (Encoder)
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        
        # Caminho Expansivo (Decoder)
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_1 = double_convolution(1024, 512)
        
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256)
        
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)
        
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)
        
        # Camada de Saída
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )
    
    def forward(self, x):
        # Downsampling path
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        
        down_9 = self.down_convolution_5(down_8)
        
        # Upsampling path com skip connections
        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        
        # Camada final
        out = self.out(x)
        return out
```

A classe UNet demonstra a estrutura modular. As camadas do caminho contrativo (downsampling) são encadeadas com operações de max pooling. O caminho expansivo (upsampling) utiliza `nn.ConvTranspose2d` e a concatenação (`torch.cat`) para fundir as características de alta resolução com as de baixa resolução.⁹

### 3.2. O Processo de Treinamento e Avaliação

#### Funções de Perda para Segmentação

A escolha da função de perda é um fator crítico para o sucesso de um modelo de segmentação. Enquanto a `CrossEntropyLoss` é uma escolha padrão para problemas de classificação multi-classe, ela pode ser inadequada para tarefas de segmentação, especialmente quando há um forte desequilíbrio de classes.¹⁸ Em muitos conjuntos de dados de segmentação (por exemplo, a segmentação de tumores em imagens médicas), a classe de interesse (o tumor) ocupa uma fração minúscula dos pixels totais em comparação com a classe de fundo. Uma função de perda como a `CrossEntropyLoss` poderia ser dominada pelos pixels de fundo, levando o modelo a uma solução trivial onde ele simplesmente prevê a classe majoritária para tudo, resultando em um modelo inutilizável.

Para resolver esse problema, a **Dice Loss** é uma alternativa mais robusta e preferível.¹⁹ A Dice Loss é baseada no Coeficiente de Dice, uma métrica de sobreposição que mede a similaridade entre duas amostras. A função de perda, definida como `1 - Dice Coefficient`, força o modelo a maximizar a sobreposição entre a máscara de segmentação predita e o ground truth, o que é ideal para lidar com o problema de desequilíbrio de classes. A `GeneralizedDiceLoss` é uma extensão que pode ser utilizada em casos ainda mais extremos de desequilíbrio, ao dar mais peso a classes com menor representatividade.¹⁹

#### Estrutura do Loop de Treinamento

O loop de treinamento em PyTorch é um processo iterativo que envolve a passagem de dados através do modelo e a atualização de seus parâmetros.¹⁵ O fluxo padrão para cada época de treinamento inclui:

1. **Iteração sobre os dados**: O loop percorre os dados em lotes (batches) do DataLoader.
2. **Passada para a frente (Forward Pass)**: O lote de imagens de entrada é alimentado ao modelo para gerar uma previsão (`output = model(data)`).
3. **Cálculo da Perda**: A previsão é comparada com o ground truth (target) usando a função de perda escolhida (`loss = criterion(output, target)`).
4. **Limpeza de Gradientes**: Os gradientes acumulados do passo anterior são zerados (`optimizer.zero_grad()`).
5. **Retropropagação (Backward Pass)**: O gradiente da perda em relação aos parâmetros do modelo é calculado (`loss.backward()`).
6. **Atualização de Parâmetros**: O otimizador ajusta os pesos do modelo na direção que minimiza a perda (`optimizer.step()`).

Essa estrutura é a base do treinamento de redes neurais, e o PyTorch fornece todas as ferramentas necessárias.¹⁵ A conveniência de frameworks de alto nível como o PyTorch Lightning é notável para abstrair e simplificar a gestão desse loop de treinamento.²⁰

## Seção 4: Comparativo de Arquiteturas: U-Net vs. FCN vs. SegNet

Para entender a relevância da U-Net, é crucial contextualizá-la em relação a outras arquiteturas seminais de segmentação. As Fully Convolutional Networks (FCNs) e a SegNet representam avanços significativos que, em diferentes graus, influenciaram a U-Net e foram seus contemporâneos. A comparação entre elas destaca as inovações e compensações de cada abordagem.

### 4.1. Fully Convolutional Network (FCN)

A FCN, proposta em 2014, foi a primeira rede neural a demonstrar a segmentação em nível de pixel de forma eficiente, substituindo as camadas densas finais de redes de classificação por camadas convolucionais de 1x1.³ Essa mudança permitiu que a rede processasse imagens de qualquer tamanho e produzisse um mapa de segmentação com as mesmas dimensões da entrada.³ No entanto, o método de upsampling da FCN, que utiliza a deconvolução, e sua abordagem de fusão de características, que combina mapas de diferentes resoluções, podem resultar em mapas de segmentação com limites borrados e uma precisão inferior em cenários complexos.²²

### 4.2. SegNet

A SegNet compartilha a arquitetura de codificador-decodificador da U-Net. Seu codificador é inspirado na arquitetura VGG16, com 13 camadas convolucionais.²³ A inovação central da SegNet reside em sua abordagem de upsampling, que utiliza os índices de max-pooling armazenados durante o caminho contrativo. O decodificador usa esses índices para realizar uma operação de "unpooling" não linear, reconstruindo a posição exata dos pixels de maior ativação.²⁴ Esse método é altamente eficiente em termos de memória e computação, pois evita o aprendizado de parâmetros na fase de upsampling, uma desvantagem das deconvoluções usadas pela FCN.²⁴ O SegNet demonstrou um desempenho superior ao FCN na restauração de detalhes e no tratamento de limites, superando-o e, em alguns casos, a própria U-Net, devido à precisão de sua técnica de unpooling com índices.²²

### 4.3. Tabela Comparativa Detalhada

A tabela a seguir resume as diferenças-chave entre as três arquiteturas, fornecendo um guia visual para o seu curso.

**Tabela 1: Comparativo de Arquiteturas de Segmentação Semântica**

| Característica | U-Net | FCN | SegNet |
|---|---|---|---|
| **Arquitetura** | Encoder-Decoder em forma de U² | Somente convolucional³ | Encoder-Decoder (similar à VGG16)²³ |
| **Método de Upsampling** | Convolução Transposta⁹ | Deconvolução²⁴ | Unpooling com índices de Max-Pooling²⁴ |
| **Conexões de Salto** | Concatenação de mapas de características¹⁰ | Fusão de mapas de múltiplas camadas²² | Ausente (substituído por índices de unpooling)²⁴ |
| **Vantagens** | Preserva detalhes espaciais, alto desempenho com poucos dados, adaptabilidade² | Processa imagens de qualquer tamanho, eficiente³ | Eficiente em memória e computação, excelente em bordas e detalhes²² |
| **Limitações** | Pode ser computacionalmente intensiva, original com saída menor que a entrada¹¹ | Limites de segmentação borrados, menor acurácia em contextos complexos²² | Pode ter erros de classificação em ambientes urbanos complexos²² |

## Seção 5: Variações e Avanços na Arquitetura U-Net

A U-Net serviu como inspiração para uma série de arquiteturas subsequentes que buscaram aprimorar seu desempenho, focando em problemas como a dificuldade de treinar redes muito profundas e a ineficiência das conexões de salto.

### 5.1. Res-UNet

A Res-UNet é uma variação que integra o conceito de aprendizado residual (ResNet) na arquitetura U-Net.²⁵ Onde a U-Net original utiliza blocos de dupla convolução, a Res-UNet os substitui por blocos residuais. A principal vantagem dos blocos residuais é a criação de conexões de salto de "curto" alcance, que permitem que o gradiente flua mais facilmente através da rede.¹² Ao adicionar essas conexões dentro dos blocos de convolução, a Res-UNet se torna mais robusta a problemas de gradiente evanescente e permite a construção de redes muito mais profundas sem o risco de degradação de desempenho. Isso a torna mais fácil de treinar e potencialmente mais poderosa para tarefas complexas.²⁵

### 5.2. Attention U-Net

A Attention U-Net é outro avanço significativo, que adiciona um mecanismo de atenção às conexões de salto da U-Net.²⁶ A motivação para essa variação é a observação de que as conexões de salto da U-Net original, embora cruciais, podem transferir uma grande quantidade de características redundantes e irrelevantes, já que as camadas iniciais do codificador contêm representações de baixo nível e fracas.¹³

Para resolver isso, a Attention U-Net introduz **Attention Gates (AGs)**, que são módulos de atenção espacial implementados nas conexões de salto.¹³ A função principal dos AGs é suprimir ativamente as ativações em regiões irrelevantes da imagem, permitindo que a rede se concentre apenas nas estruturas-alvo. Essa abordagem de "soft attention" funciona ao ponderar diferentes partes do mapa de características, atribuindo pesos maiores a regiões de alta relevância.¹³ A implementação dos AGs é diferenciável, o que permite que sejam treinados com retropropagação padrão, aprendendo a identificar as partes mais importantes da imagem.¹³ Resultados empíricos demonstram que a Attention U-Net supera consistentemente a U-Net original, com um aumento marginal de parâmetros e tempo de inferência.¹³

## Seção 6: Conclusão e Recomendações para o Curso

A U-Net se estabeleceu como uma arquitetura fundamental para a segmentação semântica, notável por sua robustez, adaptabilidade e eficiência com dados limitados. Sua estrutura de codificador-decodificador e, em particular, suas conexões de salto de concatenação, revolucionaram a forma como as redes neurais lidam com a fusão de informações contextuais e espaciais. A U-Net não é apenas um modelo; é uma base conceitual que inspirou uma série de variantes, como a Res-UNet e a Attention U-Net, que continuam a expandir os limites da segmentação de imagens.

Para a elaboração de um curso de deep learning, a U-Net é a escolha ideal como modelo introdutório para a segmentação. Sua arquitetura intuitiva em "U" e a clareza de seus conceitos (downsampling para contexto, upsampling para localização, e conexões de salto para precisão) facilitam a compreensão dos alunos.

Para a prática, recomenda-se o uso de conjuntos de dados de segmentação semântica publicamente disponíveis, como o **Cityscapes** ou o **PASCAL VOC**.¹ Esses datasets fornecem imagens com anotações de segmentação detalhadas, ideais para o treinamento e a avaliação de modelos.²⁷

Além disso, para um aprendizado aprofundado, o curso deve enfatizar a importância de decisões de engenharia, como a escolha da função de perda. Deve-se ensinar que a **Dice Loss** é frequentemente superior à `CrossEntropyLoss` para problemas de segmentação com desequilíbrio de classes, uma vez que sua natureza de sobreposição penaliza o modelo de forma mais efetiva, forçando-o a segmentar corretamente a classe minoritária. Por fim, o curso pode apresentar as variações da U-Net, como a Res-UNet e a Attention U-Net, como o próximo passo para os alunos que desejam se aprofundar na vanguarda da pesquisa em visão computacional.

## Referências citadas

1. What Is Semantic Segmentation? | IBM, acessado em agosto 25, 2025, https://www.ibm.com/think/topics/semantic-segmentation
2. U-Net: A Comprehensive Guide to Its Architecture and Applications, acessado em agosto 25, 2025, https://viso.ai/deep-learning/u-net-a-comprehensive-guide-to-its-architecture-and-applications/
3. Understanding FCN Fully Convolutional Network in Machine Vision ..., acessado em agosto 25, 2025, https://www.unitxlabs.com/resources/fcn-fully-convolutional-network-machine-vision-system-guide/
4. (PDF) Comparison of Fully Convolutional Networks (FCN) and U-Net for Road Segmentation from High Resolution Imageries - ResearchGate, acessado em agosto 25, 2025, https://www.researchgate.net/publication/344976276_Comparison_of_Fully_Convolutional_Networks_FCN_and_U-Net_for_Road_Segmentation_from_High_Resolution_Imageries
5. U-Net: Convolutional Networks for Biomedical Image Segmentation | Request PDF, acessado em agosto 25, 2025, https://www.researchgate.net/publication/305193694_U-Net_Convolutional_Networks_for_Biomedical_Image_Segmentation
6. U-Net - Wikipedia, acessado em agosto 25, 2025, https://en.wikipedia.org/wiki/U-Net
7. UNet Architecture Explained In One Shot [TUTORIAL] - Kaggle, acessado em agosto 25, 2025, https://www.kaggle.com/code/akshitsharma1/unet-architecture-explained-in-one-shot-tutorial/notebook
8. The U-Net : A Complete Guide | Medium, acessado em agosto 25, 2025, https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8
9. U-Net Architecture Explained: A Simple Guide with PyTorch Code | by Abhishek - Medium, acessado em agosto 25, 2025, https://medium.com/@AIchemizt/u-net-architecture-explained-a-simple-guide-with-pytorch-code-fc33619f2b75
10. U-Net Architecture Explained - GeeksforGeeks, acessado em agosto 25, 2025, https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
11. Implementing UNet from Scratch Using PyTorch - DebuggerCafe, acessado em agosto 25, 2025, https://debuggercafe.com/unet-from-scratch-using-pytorch/
12. What are Skip Connections in Deep Learning? - Analytics Vidhya, acessado em agosto 25, 2025, https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/
13. A detailed explanation of the Attention U-Net | by Robin Vinod | TDS ..., acessado em agosto 25, 2025, https://medium.com/data-science/a-detailed-explanation-of-the-attention-u-net-b371a5590831
14. Cook your First U-Net in PyTorch - Medium, acessado em agosto 25, 2025, https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3
15. Learning PyTorch: The Basic Program Structure | by Dagang Wei - Medium, acessado em agosto 25, 2025, https://medium.com/@weidagang/learning-pytorch-the-basic-program-structure-ed5723118b67
16. milesial/Pytorch-UNet: PyTorch implementation of the U-Net for image semantic segmentation with high quality images - GitHub, acessado em agosto 25, 2025, https://github.com/milesial/Pytorch-UNet
17. How to Implement UNet in PyTorch for Image Segmentation from Scratch? - Bhimraj Yadav, acessado em agosto 25, 2025, https://bhimraj.com.np/blog/pytorch-unet-image-segmentation-implementation
18. PyTorch Loss Functions: The Ultimate Guide - neptune.ai, acessado em agosto 25, 2025, https://neptune.ai/blog/pytorch-loss-functions
19. wolny/pytorch-3dunet: 3D U-Net model for volumetric ... - GitHub, acessado em agosto 25, 2025, https://github.com/wolny/pytorch-3dunet
20. UNet|Semantic Segmentation|PyTorch Lightning - Kaggle, acessado em agosto 25, 2025, https://www.kaggle.com/code/nikhilxb/unet-semantic-segmentation-pytorch-lightning
21. Supporting Fully Convolutional Networks (and U-Net) for ... - Datature, acessado em agosto 25, 2025, https://datature.io/blog/supporting-fully-convolutional-networks-and-u-net-for-image-segmentation
22. (PDF) Performance and Analysis of FCN, U-Net, and SegNet in ..., acessado em agosto 25, 2025, https://www.researchgate.net/publication/388315318_Performance_and_Analysis_of_FCN_U-Net_and_SegNet_in_Remote_Sensing_Image_Segmentation_Based_on_the_LoveDA_Dataset
23. www.researchgate.net, acessado em agosto 25, 2025, https://www.researchgate.net/publication/378672931_SegNet_Network_Architecture_for_Deep_Learning_Image_Segmentation_and_Its_Integrated_Applications_and_Prospects#:~:text=SegNet's%20architecture%20consists%20of%20an,level%20features%20from%20input%20images.
24. (PDF) SegNet Network Architecture for Deep Learning Image ..., acessado em agosto 25, 2025, https://www.researchgate.net/publication/378672931_SegNet_Network_Architecture_for_Deep_Learning_Image_Segmentation_and_Its_Integrated_Applications_and_Prospects
25. nikhilroxtomar/Deep-Residual-Unet: ResUNet, a semantic ... - GitHub, acessado em agosto 25, 2025, https://github.com/nikhilroxtomar/Deep-Residual-Unet
26. Attention UNET in PyTorch - Idiot Developer, acessado em agosto 25, 2025, https://idiotdeveloper.com/attention-unet-in-pytorch/
27. Best Datasets for Training Semantic Segmentation Models | Keymakr, acessado em agosto 25, 2025, https://keymakr.com/blog/best-datasets-for-training-semantic-segmentation-models/