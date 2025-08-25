---
sidebar_position: 3
title: "Família VGG"
description: "Análise da família VGG e sua filosofia de design uniforme com filtros 3x3 para construção de redes neurais profundas"
tags: [vgg, arquitetura-uniforme, profundidade, simplicidade, filtros-3x3, pytorch]
---

# A Família de Arquiteturas VGG

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qFkwYpW3LRFhp25qPB1B3Rnppm1U0fmo?usp=sharing)

## 1. Introdução: A Essência da VGG e o Contexto Histórico do Deep Learning em Visão Computacional

A história moderna da visão computacional, no contexto do deep learning, pode ser rastreada a partir do surgimento das redes neurais convolucionais (CNNs). No início da década de 2010, arquiteturas como a LeNet-5 e, de forma mais proeminente, a AlexNet em 2012, lançaram as bases para a aplicação de redes neurais profundas em desafios de reconhecimento de larga escala. A AlexNet, que popularizou o uso de CNNs no ImageNet Large Scale Visual Recognition Challenge (ILSVRC), foi pioneira ao demonstrar o poder da profundidade na extração de features, utilizando camadas de ativação ReLU e dropout. No entanto, sua arquitetura era caracterizada pelo uso de filtros convolucionais de grandes dimensões (como 11x11 e 5x5), o que conferia uma certa complexidade e um design menos uniforme.

O ano de 2014 marcou um ponto de virada na pesquisa em visão computacional com o ILSVRC daquele ano, onde a VGG, proposta pelo Visual Geometry Group da Universidade de Oxford, e a GoogLeNet competiram por destaque. Embora a GoogLeNet tenha superado a VGG na tarefa de classificação, alcançando um erro Top-5 de 6.66% contra os 7.32% da VGG, a VGG obteve uma vitória significativa na tarefa de localização, com um erro de 25.32%.

Apesar de não ser a vencedora absoluta em classificação, a VGG se tornou um modelo seminal na comunidade de pesquisa e educação. A relevância da VGG não se baseou unicamente em sua performance, mas em sua elegância e simplicidade de design. Enquanto a GoogLeNet introduziu um módulo complexo e com filtros paralelos, a VGG demonstrou que um princípio de design extremamente simples e uniforme – empilhar camadas de convolução de 3x3 – poderia alcançar resultados de ponta. Essa prova de conceito foi notavelmente didática e mais replicável. Ela solidificou a crença de que a profundidade da rede é um fator mais crucial para o desempenho do que a largura ou a complexidade do design por filtro, uma ideia que se tornaria central para o desenvolvimento de arquiteturas subsequentes como a ResNet.

## 2. Princípios de Design: A Filosofia da Simplicidade Profunda

O design da VGG é notável por sua abordagem sistemática e uniforme. Em contraste com as arquiteturas predecessoras, a VGG baseou-se quase que exclusivamente no uso de filtros convolucionais de 3x3. A cada camada, o passo (stride) era fixado em 1 pixel, e o preenchimento (padding) era aplicado para preservar a resolução espacial da entrada, permitindo a empilhagem de múltiplas camadas sem redução dimensional prematura.

Um aspecto frequentemente debatido da VGG é sua eficiência de parâmetros. Embora a arquitetura seja famosa por ter um número massivo de parâmetros (cerca de 138 milhões para a VGG-16 e 144 milhões para a VGG-19), o princípio de design de usar filtros 3x3 era, na verdade, uma estratégia para reduzir a complexidade por camada. Ao empilhar duas camadas convolucionais de 3x3 com a mesma entrada e saída, o campo receptivo efetivo é equivalente ao de um único filtro de 5x5. No entanto, essa abordagem utiliza menos parâmetros:

2 * (3 * 3) = 18 parâmetros por canal de entrada e saída, em comparação com os (5 * 5) = 25 parâmetros de um único filtro 5x5. A economia se torna ainda mais acentuada ao comparar três filtros 3x3 (campo receptivo de 7x7) com um único filtro 7x7.

A VGG aproveitou essa eficiência por filtro para aumentar a profundidade da rede exponencialmente. No entanto, o número total de parâmetros e o custo computacional tornaram-se extraordinariamente altos devido a essa profundidade e, principalmente, às camadas totalmente conectadas finais, que sozinhas contêm a maior parte dos parâmetros do modelo. O resultado foi um modelo com uma arquitetura de bloco eficiente, mas um número total de parâmetros e um custo computacional extremamente altos. Essa característica demonstra uma dualidade no design da VGG: uma abordagem inteligente e parcimoniosa no nível do bloco, que culminou em uma arquitetura final pesada e computacionalmente intensiva. A introdução de uma função de ativação não-linear, como a ReLU, após cada camada convolucional, também foi uma escolha crucial, pois introduziu a não-linearidade necessária e melhorou o tempo de computação em comparação com as funções tanh ou sigmoide, comuns em modelos anteriores.

## 3. Anatomia da Família VGG: Análise Detalhada das Arquiteturas

A família VGG é composta por variantes que diferem principalmente no número de camadas convolucionais. Todas as variantes utilizam uma entrada de imagem RGB de tamanho fixo de (224, 224, 3) e têm a mesma estrutura de camadas totalmente conectadas na saída. O pré-processamento padrão inclui a subtração do valor RGB médio, calculado sobre o conjunto de treinamento.

### 3.1. VGG-16 (Configuração D): A Arquitetura Clássica

A VGG-16, a variante mais conhecida, é composta por 16 camadas com peso, sendo 13 camadas convolucionais e 3 camadas totalmente conectadas. A arquitetura é organizada em cinco blocos de convolução e pooling. A progressão de canais de filtro é a seguinte:

- **Bloco 1:** Duas camadas convolucionais com 64 filtros de 3x3, seguidas por uma camada de max-pooling de 2x2.
- **Bloco 2:** Duas camadas convolucionais com 128 filtros de 3x3, seguidas por uma camada de max-pooling de 2x2.
- **Bloco 3:** Três camadas convolucionais com 256 filtros de 3x3, seguidas por uma camada de max-pooling de 2x2.
- **Bloco 4:** Três camadas convolucionais com 512 filtros de 3x3, seguidas por uma camada de max-pooling de 2x2.
- **Bloco 5:** Três camadas convolucionais com 512 filtros de 3x3, seguidas por uma camada de max-pooling de 2x2.

Após a última camada de pooling, o mapa de features resultante, de dimensão (7, 7, 512), é achatado (flattened) em um vetor de 25088 elementos. Este vetor alimenta a seção do classificador, composta por três camadas totalmente conectadas (FC): as duas primeiras com 4096 neurônios e a última com 1000 neurônios, correspondendo às classes do ImageNet, com uma ativação softmax final.

### 3.2. VGG-19 (Configuração E): A Variante Mais Profunda

A VGG-19 é uma extensão da VGG-16, aumentando a profundidade da rede para 19 camadas com peso, sendo 16 camadas convolucionais e 3 camadas totalmente conectadas. A arquitetura é idêntica à VGG-16, exceto pela adição de uma camada convolucional no terceiro bloco e uma em cada um dos dois últimos blocos (agora com 4 camadas de convolução cada). As camadas totalmente conectadas de saída permanecem as mesmas.

### Tabela 1: Resumo das Configurações da Família VGG

| Modelo | Número de Camadas (Peso) | Número de Camadas Convolucionais | Número de Camadas de Pooling | Total de Parâmetros (Aprox.) |
|--------|--------------------------|----------------------------------|------------------------------|------------------------------|
| VGG-11 | 11 | 8 | 5 | 133 milhões |
| VGG-13 | 13 | 10 | 5 | 133 milhões |
| VGG-16 | 16 | 13 | 5 | 138 milhões |
| VGG-19 | 19 | 16 | 5 | 144 milhões |

### 3.3. O Papel da Normalização em Lotes (BN): Estabilidade e Performance

Uma observação crucial para o estudo da VGG é que a Normalização em Lotes (Batch Normalization - BN) não fazia parte da arquitetura original, mas se tornou um aprimoramento padrão em implementações modernas. A BN é uma técnica que estabiliza as distribuições dos inputs de cada camada, o que acelera a convergência durante o treinamento e reduz a dependência de inicializações de peso precisas. A inclusão da BN permite o uso de taxas de aprendizado mais altas e melhora a robustez do treinamento.

A comunidade de pesquisa de deep learning frequentemente aprimora modelos históricos ao longo do tempo. As implementações oficiais do PyTorch oferecem variantes com BN (vgg16_bn, vgg19_bn), e os resultados de desempenho confirmam que essas versões superam as originais. Isso demonstra que as arquiteturas de deep learning não são estáticas, mas sim uma base sobre a qual a comunidade constrói melhorias. A VGG-BN é, portanto, a abordagem preferencial para tarefas de aprendizado por transferência ou para ser utilizada como backbone em novas arquiteturas.

### Tabela 2: Comparação de Desempenho (VGG vs. VGG-BN)

| Modelo | Erro Top-1 (%) | Erro Top-5 (%) |
|--------|----------------|----------------|
| VGG16 | 28.41 | 9.62 |
| VGG16_BN | 26.63 | 8.50 |
| VGG19 | 27.62 | 9.12 |
| VGG19_BN | 25.76 | 8.15 |

## 4. Análise Crítica: Vantagens e Deficiências da Família VGG

### 4.1. Vantagens: A Simplicidade que Conquista

- **Simplicidade e Replicabilidade:** A principal vantagem da arquitetura VGG é sua uniformidade e estrutura simples. O design repetitivo de blocos facilita o entendimento e a replicação do modelo, tornando-o um excelente ponto de partida para estudantes e pesquisadores. Essa clareza arquitetural contrasta com a complexidade de modelos contemporâneos como o GoogLeNet.

- **Extração de Features Robusta:** A profundidade da VGG, combinada com o uso consistente de filtros pequenos, permite que a rede aprenda features hierárquicas ricas e generalizáveis. As camadas mais profundas são capazes de capturar representações mais abstratas e semanticamente significativas.

- **Excelência em Transfer Learning:** Devido à sua capacidade de extração de features robusta e à sua arquitetura comprovada, os modelos VGG pré-treinados no ImageNet são amplamente utilizados como extratores de features para transfer learning. A VGG é frequentemente ajustada para tarefas de classificação, detecção de objetos e segmentação, aproveitando o conhecimento prévio adquirido em um grande conjunto de dados.

### 4.2. Deficiências: O Custo da Profundidade

- **Alto Custo Computacional e de Memória:** A VGG é notória por sua ineficiência. O treinamento é extremamente lento e a inferência é mais lenta em comparação com arquiteturas mais modernas. Com cerca de 138 milhões de parâmetros para a VGG-16, o modelo ocupa um espaço considerável em disco (528 MB) e requer recursos computacionais substanciais, como GPUs de alto desempenho e grandes quantidades de memória.

- **Problemas de Treinamento:** A grande profundidade e o número de parâmetros da VGG contribuem para o problema de gradientes explosivos, tornando o treinamento mais difícil e instável, especialmente sem o auxílio da normalização em lotes.

- **Limitações de Aplicação:** Devido ao seu alto custo computacional e de memória, a VGG não é prática para aplicações em dispositivos com recursos limitados, como smartphones e sistemas embarcados.

O design da VGG, embora revolucionário, também revelou desafios práticos que impulsionaram a pesquisa subsequente. A ineficiência de parâmetros e a dificuldade de treinamento em redes muito profundas motivaram o desenvolvimento de novas arquiteturas que buscaram resolver essas limitações. Por exemplo, a ResNet abordou o problema do gradiente de forma elegante com suas "conexões de atalho" (skip connections), que permitem a passagem de informações através de múltiplas camadas, aliviando o problema do gradiente de desvanecimento e explosão. A Inception, por sua vez, focou em maior eficiência computacional e capacidade de extração de features em múltiplas escalas por meio de seus módulos paralelos. A VGG não é apenas um marco histórico, mas uma ponte crítica que provou a eficácia da profundidade e, ao mesmo tempo, apresentou os desafios que a próxima geração de arquiteturas se propôs a superar.

### Tabela 3: Vantagens e Deficiências da Família VGG

| Categoria | Vantagens | Deficiências |
|-----------|-----------|--------------|
| **Design** | - Arquitetura simples e uniforme, fácil de entender e replicar - Uso de filtros 3x3 que permitem a simulação de campos receptivos maiores com menos parâmetros | - Grande número de parâmetros, principalmente nas camadas FC - Sensibilidade a problemas de gradiente (explosivo/desvanecente) em redes muito profundas |
| **Performance** | - Excelente extração de features hierárquicas e abstratas - Bom desempenho em tarefas de classificação e localização de objetos | - Lento para treinar e para inferência devido à sua profundidade e ao volume de parâmetros - Menor precisão em comparação com modelos mais modernos, como ResNet e Inception, que possuem designs mais complexos |
| **Custo e Recursos** | - Boa baseline para comparação com novas arquiteturas - Modelos pré-treinados são eficazes para transfer learning | - Requer recursos de hardware significativos (GPU, memória) - O grande tamanho do arquivo de pesos (528 MB) dificulta a implantação em dispositivos de borda |

## 5. Implementação Prática da VGG em PyTorch

### 5.1. Da Teoria à Prática: O Padrão de Design em PyTorch

A implementação de modelos de redes neurais em PyTorch tipicamente segue um padrão de design orientado a objetos. O modelo é construído como uma classe que herda de `torch.nn.Module`. Dentro dessa classe, dois métodos são essenciais: `__init__`, onde as camadas do modelo são definidas e inicializadas, e `forward`, que especifica a sequência e o fluxo de dados através dessas camadas.

As camadas fundamentais utilizadas na implementação da VGG em PyTorch incluem:

- `nn.Conv2d`: Camadas convolucionais, que recebem o número de canais de entrada e saída, o tamanho do kernel, o passo e o preenchimento.
- `nn.ReLU`: A função de ativação, que introduz não-linearidade.
- `nn.MaxPool2d`: A camada de pooling, que reduz as dimensões espaciais do mapa de features.
- `nn.Linear`: As camadas totalmente conectadas, que realizam a classificação final.
- `nn.Dropout`: Uma técnica de regularização para evitar overfitting.
- `nn.BatchNorm2d`: A camada de normalização em lotes, para estabilizar o treinamento e acelerar a convergência.
- `nn.Sequential`: Um contêiner que permite agrupar múltiplas operações em uma única etapa, facilitando a organização dos blocos da VGG.

### 5.2. Implementando VGG do Zero: Uma Abordagem Modular e Parametrizável

Uma abordagem robusta e didática para implementar a VGG do zero é usar uma função auxiliar para criar as camadas de extração de features de forma dinâmica. Em vez de listar cada camada manualmente, o que é repetitivo e não escalável para toda a família VGG, uma função `_make_layers` pode iterar sobre uma lista de configurações (cfg). Essa metodologia permite que o instrutor demonstre a implementação de toda a família VGG (VGG11, VGG13, VGG16, VGG19) com uma única classe, alterando apenas a lista de configuração.

O código a seguir exemplifica essa abordagem:

```python
import torch
import torch.nn as nn

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def VGG16(num_classes=1000):
    return VGG('VGG16', num_classes)

def VGG19(num_classes=1000):
    return VGG('VGG19', num_classes)
```

A passagem `forward` é crucial para conectar as camadas. Nela, a entrada é processada sequencialmente pelas camadas de extração de features, seguida pelo pooling adaptativo, e então a saída é achatada para a entrada do classificador totalmente conectado.

### 5.3. Abordagem Profissional: Utilizando Modelos Pré-treinados da torchvision

Na prática, a forma mais eficiente de usar a VGG é através dos modelos pré-treinados fornecidos pela biblioteca `torchvision`. Esses modelos já foram treinados no vasto conjunto de dados ImageNet, e seus pesos podem ser reutilizados para novas tarefas, o que economiza tempo e recursos computacionais. Para usar esses modelos pré-treinados, é essencial normalizar os dados de entrada com os valores `mean = [0.485, 0.456, 0.406]` e `std = [0.229, 0.224, 0.225]` utilizados no treinamento original.

O código a seguir demonstra como carregar um modelo VGG-16 pré-treinado e adaptá-lo para uma tarefa de transfer learning:

```python
import torchvision.models as models

# Carregar o modelo VGG-16 pré-treinado com BN
model_vgg16_bn = models.vgg16_bn(pretrained=True)

# A camada de classificador original da VGG tem 1000 saídas
print(model_vgg16_bn.classifier)

# Exemplo de adaptação para 2 classes (para uma tarefa de classificação binária)
# Congelar as camadas de extração de features para não treinar novamente
for param in model_vgg16_bn.features.parameters():
    param.requires_grad = False

# Modificar a última camada do classificador para se adequar ao número de classes da nova tarefa
num_ftrs = model_vgg16_bn.classifier.in_features
model_vgg16_bn.classifier = nn.Linear(num_ftrs, 2)

# O modelo agora está pronto para ser treinado na nova tarefa com apenas 2 classes
print(model_vgg16_bn.classifier)
```

Essa abordagem capitaliza a capacidade da VGG de atuar como um robusto extrator de features, mostrando sua utilidade contínua no ecossistema atual do deep learning.

## Referências

1. VGG-Net Architecture Explained - GeeksforGeeks, acessado em agosto 24, 2025, https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

2. Very Deep Convolutional Networks for Large-Scale Image Recognition Vol 12 Issue 08, acessado em agosto 24, 2025, https://www.researchgate.net/publication/390956657_Very_Deep_Convolutional_Networks_for_Large-Scale_Image_Recognition_Vol_12_Issue_08

3. VGGNet - Wikipedia, acessado em agosto 24, 2025, https://en.wikipedia.org/wiki/VGGNet

4. VGG-16 | CNN model - GeeksforGeeks, acessado em agosto 24, 2025, https://www.geeksforgeeks.org/computer-vision/vgg-16-cnn-model/

5. Benefits of VGG (Visual Geometry Group) Networks - BytePlus, acessado em agosto 24, 2025, https://www.byteplus.com/en/topic/401678

6. Everything you need to know about VGG16 | by Great Learning - Medium, acessado em agosto 24, 2025, https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918

7. Understanding the VGG19 Architecture - OpenGenus IQ, acessado em agosto 24, 2025, https://iq.opengenus.org/vgg19-architecture/

8. VGG from Scratch with PyTorch – Step-by-Step Guide - MangoHost, acessado em agosto 24, 2025, https://mangohost.net/blog/vgg-from-scratch-with-pytorch-step-by-step-guide/

9. Writing VGG from Scratch in PyTorch | DigitalOcean, acessado em agosto 24, 2025, https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch

10. Very Deep Convolutional Networks for Large-Scale Image ..., acessado em agosto 24, 2025, https://www.researchgate.net/publication/319770291_Very_Deep_Convolutional_Networks_for_Large-Scale_Image_Recognition

11. vgg-nets - PyTorch, acessado em agosto 24, 2025, https://pytorch.org/hub/pytorch_vision_vgg/

12. VGG vs ResNet vs Inception vs MobileNet | Kaggle, acessado em agosto 24, 2025, https://www.kaggle.com/discussions/getting-started/433540

13. Reducing Computational Complexity in CNNs: A Focus on VGG19 ..., acessado em agosto 24, 2025, https://thesai.org/Downloads/Volume16No6/Paper_105-Reducing_Computational_Complexity_in_CNNs.pdf

14. (PDF) Improving the Performance of VGG Through Different Granularity Feature Combinations - ResearchGate, acessado em agosto 24, 2025, https://www.researchgate.net/publication/346306126_Improving_the_Performance_of_VGG_Through_Different_Granularity_Feature_Combinations

15. ResNet Vs EfficientNet vs VGG Vs NN - DEV Community, acessado em agosto 24, 2025, https://dev.to/saaransh_gupta_1903/resnet-vs-efficientnet-vs-vgg-vs-nn-2hf5

16. VGG — Torchvision main documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/main/models/vgg.html

17. vgg16_bn — Torchvision 0.12 documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/0.12/generated/torchvision.models.vgg16_bn.html

18. VGG — Torchvision 0.22 documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/0.22/models/vgg.html

19. Understanding VGG Networks: A Comprehensive Guide - BytePlus, acessado em agosto 24, 2025, https://www.byteplus.com/en/topic/401679

20. A comparison between VGG16, VGG19 and ResNet50 architecture ..., acessado em agosto 24, 2025, https://www.researchgate.net/publication/363129766_A_comparison_between_VGG16_VGG19_and_ResNet50_architecture_frameworks_for_Image_Classification