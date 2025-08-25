---
sidebar_position: 4
title: "ResNet: Redes Residuais e Skip Connections"
description: "Arquitetura revolucionária que solucionou o problema do gradiente desvanecente através de conexões residuais"
tags: [resnet, redes-residuais, skip-connections, gradiente-desvanecente, aprendizagem-residual, pytorch]
---

# Arquiteturas ResNet e sua Implementação em PyTorch

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ikHEdPPz5WjaBWvLg8PrF0O0RUme7Zc6?usp=sharing)

## Capítulo 1: O Contexto Histórico e a Motivação para o ResNet

Avanços em visão computacional e outras áreas do aprendizado profundo dependem, em grande parte, da capacidade de treinar redes neurais cada vez mais profundas. Antes da introdução das Redes Residuais (ResNets), esse objetivo encontrava dois obstáculos técnicos significativos que limitavam a escalabilidade dos modelos. A compreensão desses desafios é fundamental para apreciar a inovação que o ResNet trouxe.

### 1.1. O Dilema do Aprofundamento da Rede: O Problema do Gradiente Desvanecente

O primeiro e mais crítico problema enfrentado por redes neurais profundas era o fenômeno do gradiente desvanecente. Durante o processo de treinamento, o algoritmo de retropropagação (backpropagation) calcula os gradientes da função de perda em relação aos pesos do modelo para realizar as atualizações necessárias. Em uma rede profunda, esses gradientes são calculados por meio de uma longa cadeia de multiplicações de derivadas, uma para cada camada percorrida. O problema surge quando essas derivadas são valores pequenos, como é o caso das funções de ativação sigmoid ou tanh, cujas derivadas se aproximam de zero para entradas muito grandes ou muito pequenas.

A multiplicação repetida desses valores pequenos faz com que o gradiente nas camadas iniciais da rede se torne exponencialmente pequeno, tendendo a zero.

A consequência prática deste problema é que as atualizações de peso nas camadas mais próximas à entrada se tornam negligenciáveis ou inexistentes, paralisando o aprendizado dessas camadas. Essas camadas iniciais são cruciais para o sucesso do modelo, pois são responsáveis por extrair as características de baixo nível da entrada, como bordas, texturas e cores. A falha em aprender essas representações fundamentais compromete o desempenho geral da rede e, na prática, impunha um limite na profundidade das redes que podiam ser treinadas de forma eficaz. O gradiente desvanecente não era apenas uma curiosidade matemática, mas uma limitação de engenharia que impedia a escalabilidade dos modelos para níveis de desempenho superiores.

### 1.2. O Fenômeno de Degradação

Outro desafio observado nos modelos pré-ResNet foi o problema de degradação. Contrariamente ao overfitting, onde o desempenho em dados de validação diminui enquanto a precisão em dados de treinamento continua a aumentar, a degradação se manifesta quando a adição de camadas extras a uma rede profunda leva a um aumento no erro tanto nos dados de treinamento quanto nos de teste. Embora, teoricamente, adicionar camadas extras a uma rede mais rasa pudesse resultar em uma solução de, no mínimo, a mesma qualidade (simplesmente aprendendo o mapeamento de identidade para as camadas adicionadas), essa otimização se mostrou incrivelmente difícil na prática para as arquiteturas de rede neural convencionais.

### 1.3. O Cenário Pré-ResNet

No período que precedeu o ResNet, a pesquisa em visão computacional era dominada por arquiteturas como a VGG e a Inception. A VGGNet era notável por sua simplicidade e profundidade, utilizando blocos empilhados de camadas convolucionais de 3x3. Embora poderosa, ela sofria dos problemas de gradiente e degradação, e era conhecida por ser computacionalmente muito cara devido ao grande número de parâmetros e FLOPs (Floating Point Operations).

A arquitetura Inception (do GoogLeNet), por outro lado, buscava a eficiência computacional através de um design mais complexo que utilizava convoluções fatoradas e múltiplos caminhos paralelos para extrair recursos em diferentes escalas dentro de um único bloco. Embora fosse computacionalmente mais eficiente do que a VGG, sua arquitetura intrincada e o grande número de hiperparâmetros dificultavam a modificação e a adaptação para novas tarefas, tornando-a menos flexível para engenheiros e pesquisadores.

O sucesso do ResNet pode ser atribuído não apenas à sua eficácia em resolver os problemas de degradação e gradiente, mas também à elegância de sua solução. Ele adotou a abordagem simples e empilhada da VGG e adicionou um único e poderoso truque: a conexão de atalho. Essa inovação permitiu superar os principais problemas da época sem introduzir a complexidade de engenharia da Inception, o que foi fundamental para sua adoção generalizada.

## Capítulo 2: O Princípio Fundamental do ResNet: Aprendizagem Residual

O cerne da arquitetura ResNet reside no conceito de aprendizado residual, uma abordagem que reparametrizou a forma como as redes neurais aprendem. Em vez de uma camada tentar aprender diretamente um mapeamento complexo H(x), ela aprende a função residual F(x), que representa as perturbações necessárias para ir da entrada x para a saída desejada H(x).

### 2.1. Conexões de Salto (Skip Connections) e Mapeamento de Identidade

A inovação é implementada através das conexões de atalho, também conhecidas como conexões de salto ou residuais. Essas conexões criam um caminho de desvio que ignora uma ou mais camadas e adiciona a entrada de um bloco diretamente à sua saída.

Matematicamente, o mapeamento de um bloco residual é expresso como:

**H(x) = F(x) + x**

Nesta formulação, x é o vetor de entrada para o bloco, e F(x) é a função residual que representa as camadas convolucionais e as ativações dentro do bloco. A intuição por trás dessa abordagem é que, se o mapeamento ideal for simplesmente o mapeamento de identidade, a rede pode facilmente aprender a função residual F(x)=0. Em uma rede convencional, as camadas teriam que aprender um mapeamento de identidade complexo, uma tarefa que se mostrou desafiadora. A conexão de atalho torna essa tarefa trivial, garantindo que o desempenho de uma rede profunda seja, no mínimo, tão bom quanto o de uma rede mais rasa. Essa capacidade de passar a informação original para a frente previne a degradação de desempenho à medida que a rede se aprofunda.

### 2.2. Como as Conexões Residuais Mitigam o Gradiente Desvanecente

O grande poder das conexões residuais se revela durante a retropropagação. A regra da cadeia para o gradiente da função de perda L com relação à entrada x do bloco residual é dada por:

**∂L/∂x = ∂L/∂H · ∂H/∂x = ∂L/∂H · (1 + ∂F/∂x)**

A presença do termo 1 no gradiente do mapeamento do bloco, 1+∂F/∂x, é a chave para mitigar o problema do gradiente desvanecente. Mesmo que a derivada da função residual, ∂F/∂x, seja muito pequena, o gradiente total não se anulará, pois o termo 1 garante um caminho de fluxo direto e desobstruído. Concretamente, isso significa que uma parte do gradiente flui diretamente de volta através da conexão de atalho, fornecendo um fluxo robusto e desimpedido para as camadas iniciais.

Essa dualidade funcional da conexão de atalho é o que a torna tão impactante. Ela não apenas facilita o aprendizado do mapeamento de identidade na passagem para a frente, mas também age como um "caminho de fluxo de informações" de via dupla, permitindo que os gradientes fluam para trás sem serem atenuados e que as representações de características de baixo nível fluam para a frente para serem reutilizadas pelas camadas mais profundas. Essa dupla funcionalidade torna o ResNet excepcionalmente robusto e permitiu o treinamento de redes neurais com centenas de camadas.

## Capítulo 3: As Arquiteturas ResNet Clássicas: Uma Análise Detalhada

A família ResNet é composta por diversas arquiteturas, diferenciadas principalmente pelo número de camadas e pelo tipo de bloco de construção que utilizam. As duas variantes mais fundamentais de blocos são o Bloco Básico e o Bloco Gargalo.

### 3.1. O Bloco Básico (BasicBlock)

O Bloco Básico é o bloco de construção mais simples da família ResNet, utilizado em arquiteturas mais rasas, como o ResNet-18 e o ResNet-34. Ele consiste em uma sequência de duas camadas convolucionais de 3x3, com normalização de lote (batch normalization) e ativação ReLU aplicadas entre elas. A entrada para o bloco é adicionada à saída dessas duas camadas antes de uma ativação ReLU final. A simplicidade e a eficácia desse bloco tornaram-no uma escolha popular para modelos com menor complexidade, adequados para tarefas onde a eficiência computacional é uma prioridade.

### 3.2. O Bloco Gargalo (Bottleneck)

Para redes mais profundas (ResNet-50, ResNet-101 e ResNet-152), o Bloco Básico se torna computacionalmente ineficiente. A solução de design para este problema foi o Bloco Gargalo, que adota uma estrutura de três camadas para otimizar o tempo de treinamento. A estrutura consiste em:

1. Uma camada convolucional de 1x1 que reduz a dimensionalidade dos canais de entrada.
2. Uma camada convolucional de 3x3 que opera com o número de canais reduzido.
3. Uma segunda camada convolucional de 1x1 que restaura a dimensionalidade original dos canais, ou a expande, se necessário.

Essa configuração de "gargalo" reduz drasticamente o número de operações, especialmente na custosa camada convolucional de 3x3. O Bloco Gargalo é um exemplo de um compromisso de design: aumenta a profundidade da rede (3 camadas por bloco) para obter maior precisão, mas com uma complexidade computacional por bloco que é gerenciável, tornando o treinamento de redes muito profundas viável em grande escala. O fato de que o ResNet-50 supera o ResNet-34 com um aumento marginal em FLOPs, apesar de usar mais camadas, valida a eficácia dessa troca.

### 3.3. Tabela 1: Comparativo das Arquiteturas ResNet Clássicas

A tabela a seguir resume as principais características das arquiteturas ResNet clássicas, fornecendo uma referência para a relação entre profundidade, tipo de bloco, complexidade computacional e desempenho.

| Arquitetura | Tipo de Bloco | Número de Camadas | Parâmetros (M) | FLOPs (G) | Acurácia Top-1 (ImageNet-1K) |
|-------------|---------------|------------------|----------------|-----------|------------------------------|
| ResNet-18   | Básico        | 18               | 11.69          | 3.66      | 71.0%                        |
| ResNet-34   | Básico        | 34               | 21.80          | 7.36      | 74.6%                        |
| ResNet-50   | Gargalo       | 50               | 25.56          | 8.19      | 76.5%                        |
| ResNet-101  | Gargalo       | 101              | 44.55          | 15.52     | 77.6%                        |
| ResNet-152  | Gargalo       | 152              | 60.19          | 23.05     | 78.3%                        |

*Dados baseados em implementações padrão, com performance e complexidade podendo variar entre diferentes frameworks.*

## Capítulo 4: Variações e Evoluções da Família ResNet

O princípio de aprendizado residual provou ser tão poderoso que deu origem a várias variantes e arquiteturas que o incorporaram. As mais notáveis são as Wide Residual Networks e as ResNeXts.

### 4.1. Wide Residual Networks (WRN): A Prioridade da Largura sobre a Profundidade

As Wide Residual Networks (WRN) surgiram de uma observação contra-intuitiva: aumentar a profundidade não é a única, e talvez não a melhor, maneira de melhorar a performance. Os autores do WRN argumentaram que modelos mais rasos e mais largos (com mais canais de filtro) podem superar modelos mais profundos e mais finos, com maior eficiência computacional. A arquitetura WRN introduz um "fator de alargamento" (widening factor) k que multiplica o número de canais em cada camada convolucional, aumentando a largura do modelo.

Essa filosofia desafia diretamente a suposição de que "mais profundidade é sempre melhor", que era a principal motivação para o ResNet original. Isso destaca uma tendência na pesquisa de arquitetura de redes neurais, onde a exploração de múltiplas dimensões de escalabilidade (profundidade, largura, e, como veremos a seguir, cardinalidade) se tornou fundamental para obter o melhor desempenho.

### 4.2. ResNeXt: A Introdução da Cardinalidade

O ResNeXt, abreviação de Residual Networks with Aggregated Transformations, é um modelo híbrido que fundiu as ideias de dois paradigmas de sucesso: as conexões de atalho do ResNet e a abordagem de múltiplos caminhos do Inception. A inovação central do ResNeXt é a introdução da "cardinalidade" como uma nova dimensão de escalabilidade, além da profundidade e da largura.

A cardinalidade refere-se ao número de caminhos de transformação paralelos dentro de cada bloco residual. Em vez de aumentar a complexidade de um único caminho (por exemplo, com mais filtros), o ResNeXt usa convoluções agrupadas para dividir a entrada em múltiplos caminhos idênticos, que são processados em paralelo e depois agregados. Essa estratégia permite que o modelo aprenda representações de características mais ricas e diversificadas sem um aumento proporcional no custo computacional. A existência do ResNeXt demonstra a natureza evolutiva da pesquisa em deep learning, onde as arquiteturas de ponta são frequentemente o resultado da combinação e refinamento de conceitos comprovados. O fato de o ResNeXt-50 superar o ResNet-50 com uma complexidade de parâmetros e FLOPs similar valida a eficácia da abordagem de cardinalidade.

### 4.3. Tabela 2: Performance e Eficiência das Variações ResNet

A tabela a seguir compara o desempenho e a complexidade de um modelo ResNet clássico com sua contraparte ResNeXt, destacando a eficiência da cardinalidade.

| Modelo | Parâmetros (M) | FLOPs (G) | Erro de Validação Top-1 (%) |
|--------|----------------|-----------|----------------------------|
| ResNet-50 | 25.56 | 8.19 | 23.9% |
| ResNeXt-50 (32x4d) | ~25.03 | ~4.1 | 22.2% |

*Dados baseados no desempenho no conjunto de dados ImageNet-1K. A cardinalidade do ResNeXt-50 é de 32, com largura de 4 canais por grupo (4d).*

## Capítulo 5: Vantagens e Deficiências da Arquitetura ResNet

O ResNet é uma arquitetura revolucionária, mas, como qualquer modelo, possui um conjunto de vantagens e limitações que devem ser consideradas na prática.

### 5.1. Vantagens

- **Mitigação do Gradiente Desvanecente**: A principal e mais importante vantagem do ResNet é sua capacidade de permitir que gradientes fluam diretamente através da rede por meio das conexões de atalho, resolvendo um problema que por muito tempo limitou a profundidade das redes neurais.

- **Permite Redes Extremamente Profundas**: Ao resolver o problema do gradiente desvanecente e de degradação, o ResNet tornou possível treinar redes com mais de cem camadas (como o ResNet-101 e -152) e até modelos com mais de 1000 camadas, alcançando precisões que eram inatingíveis antes.

- **Melhor Fluxo de Gradiente e Convergência**: O caminho de gradiente desobstruído facilita a otimização e a convergência mais rápida durante o treinamento, tornando o processo mais estável.

- **Reutilização de Características**: As conexões de atalho permitem que a rede reutilize diretamente as características de camadas anteriores, o que é um fator-chave para melhorar a generalização do modelo.

- **Ampla Aplicabilidade**: O princípio residual provou ser tão fundamental que foi adotado por outras arquiteturas, incluindo os modelos Transformers, que são o padrão atual em processamento de linguagem natural. A incapacidade de treinar modelos de Transformer muito profundos sem conexões residuais reforça a universalidade do princípio.

### 5.2. Deficiências

- **Custo Computacional e de Memória**: Embora o Bloco Gargalo tenha melhorado a eficiência, as arquiteturas ResNet mais profundas ainda exigem um número substancial de parâmetros e FLOPs, o que se traduz em um alto custo computacional e de memória para treinamento e inferência, especialmente para implantação em dispositivos com recursos limitados.

- **Risco de Overfitting**: Dada a sua alta capacidade de representação, as versões muito profundas do ResNet podem sofrer de overfitting em conjuntos de dados menores, a menos que sejam aplicadas técnicas de regularização adequadas.

- **Retornos Decrescentes**: A partir de uma certa profundidade, a adição de mais camadas pode não gerar ganhos de precisão significativos. Encontrar o equilíbrio ideal entre profundidade e eficiência é um desafio de design para algumas aplicações.

## Capítulo 6: Implementação Prática em PyTorch

Para a implementação em PyTorch, existem duas abordagens principais: uma rápida e prática usando a biblioteca torchvision e uma abordagem de baixo nível para uma compreensão mais profunda da arquitetura.

### 6.1. Iniciação Rápida: Utilizando torchvision.models

A maneira mais eficiente de utilizar o ResNet na prática é por meio da biblioteca torchvision.models, que fornece implementações pré-construídas e pesos pré-treinados em grandes conjuntos de dados como o ImageNet. O conceito de transfer learning permite que você use um modelo pré-treinado como um extrator de características e adapte a camada de classificação final para sua nova tarefa, economizando tempo e recursos computacionais.

```python
import torch
import torch.nn as nn
from torchvision import models

def get_resnet_pretrained(model_name, num_classes):
    """
    Carrega um modelo ResNet pré-treinado e o adapta para uma nova tarefa.
    
    Args:
        model_name (str): Nome do modelo ResNet ('resnet18', 'resnet50', etc.).
        num_classes (int): Número de classes para a nova tarefa de classificação.
    
    Returns:
        nn.Module: O modelo ResNet adaptado.
    """
    # Carrega o modelo com pesos pré-treinados no ImageNet
    # A opção 'weights' é a forma moderna de carregar pesos pré-treinados
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("Modelo não suportado. Escolha 'resnet18' ou 'resnet50'.")
    
    # Congela os pesos do modelo pré-treinado para que não sejam atualizados durante o treinamento
    for param in model.parameters():
        param.requires_grad = False
    
    # Substitui a camada de classificação final (fc) para a nova tarefa
    # A nova camada 'fc' terá os pesos inicializados aleatoriamente e será treinada
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Exemplo de uso para uma tarefa com 10 classes
# model_adapted = get_resnet_pretrained('resnet18', num_classes=10)
# print(model_adapted)
```

### 6.2. Construindo ResNet do Zero: Uma Análise Estrutural

Para uma compreensão completa, é essencial entender como os blocos e a arquitetura ResNet são construídos a partir de componentes básicos do PyTorch. A implementação oficial da torchvision fornece a base para esta análise.

A classe principal ResNet é uma subclasse de nn.Module que orquestra a montagem das camadas. As classes BasicBlock e Bottleneck são também subclasses de nn.Module que definem a estrutura de um único bloco residual.

#### O Bloco Básico (BasicBlock)

O BasicBlock para ResNet-18 e ResNet-34 consiste em duas camadas convolucionais de 3x3. A conexão de atalho é uma simples adição da entrada à saída da segunda convolução, após a normalização de lote. Se houver uma alteração na dimensionalidade (por exemplo, na dimensão espacial ou no número de canais), uma camada de downsampling é adicionada ao caminho de atalho.

```python
import torch
import torch.nn as nn

# Bloco de construção do ResNet-18 e ResNet-34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Primeira camada convolucional: 3x3
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Segunda camada convolucional: 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Adiciona a entrada ('identity') à saída do bloco
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

#### O Bloco Gargalo (Bottleneck)

O Bottleneck para ResNet-50, -101 e -152 é mais complexo, com três convoluções: 1x1, 3x3 e 1x1. O expansion aqui é 4, significando que o número de canais na saída do bloco (planes) é 4 vezes maior do que os canais intermediários (width).

```python
# Bloco de construção do ResNet-50, ResNet-101 e ResNet-152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        width = planes  # Ajuste para a notação original do código
        
        # Primeira camada: 1x1 para reduzir canais
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # Segunda camada: 3x3
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        # Terceira camada: 1x1 para restaurar canais
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Adiciona a entrada ('identity') à saída do bloco
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

#### A Classe Principal ResNet

A classe ResNet monta a arquitetura completa, empilhando múltiplos blocos residuais. A função _make_layer é uma função auxiliar que cria uma sequência de blocos residuais. Para construir um modelo específico, você passa o tipo de bloco e a contagem de camadas por estágio para o construtor.

### 6.3. Tabela 3: Mapeamento de Parâmetros para Construção de Camadas

Esta tabela fornece a "receita" para construir as arquiteturas ResNet clássicas a partir das classes de bloco.

| Arquitetura | block | layers |
|-------------|--------|--------|
| ResNet-18 | BasicBlock | [2, 2, 2, 2] |
| ResNet-34 | BasicBlock | [3, 4, 6, 3] |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] |
| ResNet-101 | Bottleneck | [3, 4, 23, 3] |
| ResNet-152 | Bottleneck | [3, 8, 36, 3] |

## Implementações Completas das Arquiteturas ResNet

### Implementação Completa do ResNet-18

```python
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        
        # Camada convolucional inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Camadas residuais
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Exemplo de uso
# model = ResNet18(num_classes=10)
# print(model)
```

### Implementação Completa do ResNet-34

```python
class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.inplanes = 64
        
        # Camada convolucional inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Camadas residuais - configuração [3, 4, 6, 3]
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

### Implementação Completa do ResNet-50

```python
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.inplanes = 64
        
        # Camada convolucional inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Camadas residuais - configuração [3, 4, 6, 3] com Bottleneck
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

### Implementação Completa do ResNet-101

```python
class ResNet101(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet101, self).__init__()
        self.inplanes = 64
        
        # Camada convolucional inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Camadas residuais - configuração [3, 4, 23, 3] com Bottleneck
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)  # 23 blocos na layer3
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

### Implementação Completa do ResNet-152

```python
class ResNet152(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet152, self).__init__()
        self.inplanes = 64
        
        # Camada convolucional inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Camadas residuais - configuração [3, 8, 36, 3] com Bottleneck
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 8, stride=2)   # 8 blocos na layer2
        self.layer3 = self._make_layer(Bottleneck, 256, 36, stride=2)  # 36 blocos na layer3
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # Pooling e classificação
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

### Factory Function para Criar Qualquer Arquitetura ResNet

```python
def create_resnet(architecture, num_classes=1000):
    """
    Factory function para criar qualquer arquitetura ResNet.
    
    Args:
        architecture (str): Nome da arquitetura ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes (int): Número de classes de saída
    
    Returns:
        nn.Module: O modelo ResNet solicitado
    """
    architectures = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }
    
    if architecture not in architectures:
        raise ValueError(f"Arquitetura {architecture} não suportada. Escolha entre: {list(architectures.keys())}")
    
    block, layers = architectures[architecture]
    
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.inplanes = 64
            
            # Camada convolucional inicial
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Camadas residuais
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            
            # Pooling e classificação
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            
            # Inicialização dos pesos
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x
    
    return ResNet()

# Exemplos de uso:
# model_18 = create_resnet('resnet18', num_classes=10)
# model_50 = create_resnet('resnet50', num_classes=100)
# model_152 = create_resnet('resnet152', num_classes=1000)
```

### Exemplo de Uso Prático

```python
# Exemplo de uso completo
def main():
    # Criar diferentes arquiteturas ResNet
    models = {}
    architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    
    for arch in architectures:
        model = create_resnet(arch, num_classes=10)  # CIFAR-10 tem 10 classes
        models[arch] = model
        print(f"{arch}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parâmetros treináveis")
    
    # Teste com uma entrada exemplo
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 canais, 224x224 pixels
    
    for arch, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"{arch} output shape: {output.shape}")

if __name__ == "__main__":
    main()
```
