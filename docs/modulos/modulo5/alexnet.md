---
sidebar_position: 1
title: "AlexNet: A Revolução do Deep Learning"
description: "Arquitetura pioneira que iniciou a era moderna do deep learning em visão computacional, suas inovações e implementação em PyTorch"
tags: [alexnet, cnn, deep-learning, pytorch, imagenet, relu, dropout]
---

# AlexNet

**Template do Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LtZZjLQWm7vnf33s1okd0WmvxAvTaLe5?usp=sharing)

**Solução do Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17mvatPet-R803WzN8GVoZorWzyB3UAu8?usp=sharing)

## 1. Introdução: O Legado e a Revolução da AlexNet

A história da visão computacional é marcada por um ponto de viragem inegável: o ano de 2012. Até então, o campo era dominado por métodos tradicionais de aprendizado de máquina, com engenharia manual de características. Redes neurais profundas, embora teoricamente promissoras, eram frequentemente consideradas impraticáveis ou ineficientes para tarefas em larga escala. A crença predominante era que o treinamento de modelos com múltiplas camadas em grandes conjuntos de dados era uma tarefa computacionalmente proibitiva e suscetível a problemas como o desaparecimento do gradiente. A AlexNet, desenvolvida por Alex Krizhevsky, Ilya Sutskever e Geoffrey Hinton, da Universidade de Toronto, desafiou e reverteu essa percepção, catalisando o que hoje é conhecido como a revolução do deep learning.

A vitória da AlexNet no Desafio de Reconhecimento Visual em Larga Escala do ImageNet (ILSVRC) em 30 de setembro de 2012, foi um marco decisivo. A rede neural convolucional (CNN) alcançou uma taxa de erro top-5 de 15,3%, superando o segundo colocado em mais de 10,8 pontos percentuais. Essa performance de vanguarda não apenas garantiu sua vitória, mas também provou de forma inequívoca que as CNNs profundas eram eficazes e escaláveis para tarefas de reconhecimento de imagem em larga escala. A AlexNet demonstrou que as máquinas podiam alcançar um desempenho superior ao humano em tarefas de classificação de imagens quando a arquitetura era bem projetada e os recursos computacionais eram adequadamente aproveitados.

O sucesso da AlexNet não foi um evento isolado, mas sim o resultado da convergência de três avanços cruciais que amadureceram simultaneamente ao longo da década anterior. A primeira peça do quebra-cabeça foi a disponibilidade de dados em larga escala. O ImageNet, um gigantesco conjunto de dados com 1,2 milhão de imagens rotuladas em 1.000 categorias distintas, forneceu a base para que modelos com muitos parâmetros pudessem ser treinados sem sofrer de overfitting severo. A coleta e rotulagem desses dados, utilizando o crowdsourcing da Amazon Mechanical Turk, representaram um desafio e um feito sem precedentes na época, superando outros conjuntos de dados por ordens de magnitude.

A segunda peça foi o poder computacional. O treinamento de uma rede com 60 milhões de parâmetros e 650.000 neurônios era uma tarefa proibitiva para as CPUs da época. A AlexNet explorou o paralelismo das Unidades de Processamento Gráfico (GPUs), treinando-se em duas GPUs Nvidia GTX 580 por um período de cinco a seis dias. Esse uso pioneiro de GPUs demonstrou a importância do hardware otimizado para o deep learning, validando os investimentos em chips especializados e estabelecendo um novo ciclo de inovação que, posteriormente, levou ao desenvolvimento de aceleradores de IA como as Unidades de Processamento de Tensor (TPUs).

Finalmente, a AlexNet introduziu inovações algorítmicas que permitiram que o modelo fosse treinado de forma eficaz. O uso de Unidades Lineares Retificadas (ReLU), o método de regularização Dropout e a técnica de aumento de dados (data augmentation) foram os pilares que permitiram o treinamento bem-sucedido de uma arquitetura profunda, superando as limitações de modelos anteriores como a LeNet. Juntos, esses três elementos — dados, hardware e algoritmos — formaram a base para o modelo que não apenas venceu uma competição, mas que também redefiniu o campo da inteligência artificial.


## 2. Anatomia da AlexNet: Arquitetura e Hiperparâmetros Detalhados

A AlexNet é caracterizada por sua estrutura de oito camadas: cinco camadas convolucionais, algumas seguidas por camadas de max-pooling, e três camadas totalmente conectadas, com um total de 60 milhões de parâmetros e 650.000 neurônios. O design da rede é um exemplo clássico de arquitetura de extração de características em cascata, onde camadas iniciais aprendem características de baixo nível (como bordas e texturas) e camadas mais profundas aprendem características de alto nível (como partes de objetos e formas). A arquitetura foi cuidadosamente construída para extrair recursos hierárquicos de imagens e é detalhada na Tabela 1.

### Tabela 1: Arquitetura da AlexNet - Detalhes Camada a Camada

| Camada | Tipo | Dimensão de Entrada | Parâmetros da Camada | Dimensão de Saída |
|--------|------|-------------------|---------------------|-------------------|
| Input | Imagem | 227x227x3 | N/A | 227x227x3 |
| CONV1 | Convolução | 227x227x3 | 96 filtros de 11x11, stride=4, padding=2 | 55x55x96 |
| MAXPOOL1 | Max-Pooling | 55x55x96 | Pool size=3x3, stride=2 | 27x27x96 |
| CONV2 | Convolução | 27x27x96 | 256 filtros de 5x5, stride=1, padding=2 | 27x27x256 |
| MAXPOOL2 | Max-Pooling | 27x27x256 | Pool size=3x3, stride=2 | 13x13x256 |
| CONV3 | Convolução | 13x13x256 | 384 filtros de 3x3, stride=1, padding=1 | 13x13x384 |
| CONV4 | Convolução | 13x13x384 | 384 filtros de 3x3, stride=1, padding=1 | 13x13x384 |
| CONV5 | Convolução | 13x13x384 | 256 filtros de 3x3, stride=1, padding=1 | 13x13x256 |
| MAXPOOL3 | Max-Pooling | 13x13x256 | Pool size=3x3, stride=2 | 6x6x256 |
| FC6 | Totalmente Conectada | 9216 (256×6×6) | 4096 neurônios | 4096 |
| FC7 | Totalmente Conectada | 4096 | 4096 neurônios | 4096 |
| FC8 | Totalmente Conectada | 4096 | 1000 neurônios | 1000 |

### 2.1 O Mistério da Dimensão de Entrada

Apesar de o artigo original da AlexNet ter mencionado uma dimensão de entrada de 224x224, essa dimensão é matematicamente inconsistente com os hiperparâmetros da primeira camada convolucional. Uma imagem de 224x224 pixels, com um kernel de 11x11 e um passo (stride) de 4, resultaria em uma dimensão de saída de 54x54, e não 55x55. A dimensão de entrada correta para que os cálculos de convolução e pooling se alinhem com as dimensões de saída relatadas é de 227x227 pixels. Este detalhe, embora aparentemente pequeno, demonstra a natureza exploratória da pesquisa em 2012 e a importância da validação e replicação dos resultados. A correção dessa inconsistência pela comunidade de pesquisa subsequente destaca a importância de um escrutínio rigoroso em trabalhos científicos.

### 2.2 As Inovações Arquiteturais Fundamentais

A AlexNet não foi apenas uma rede profunda, mas uma demonstração de como as inovações arquiteturais poderiam superar os desafios do treinamento em grande escala.

**Unidades Lineares Retificadas (ReLU):** Antes da AlexNet, funções de ativação saturantes, como tanh e sigmoid, eram comuns. Essas funções sofriam do problema de gradiente de saturação, onde o gradiente se tornava próximo de zero para valores de entrada grandes, desacelerando o treinamento e dificultando o treinamento de redes profundas. A AlexNet adotou a função de ativação ReLU, definida como f(x)=max(0,x), que, por ser não saturante, acelerou significativamente a convergência do modelo e tornou o treinamento de redes mais profundas uma tarefa viável. O treinamento do AlexNet foi 6 vezes mais rápido com o uso de ReLUs do que com a função tanh.

**Dropout:** Com 60 milhões de parâmetros, a AlexNet era altamente suscetível ao overfitting, especialmente considerando que o treinamento em grande escala era um conceito relativamente novo. Para combater isso, os autores introduziram o Dropout, uma técnica de regularização em que os neurônios são desativados aleatoriamente com uma probabilidade de 0.5 durante a fase de treinamento. Isso impede que a rede dependa excessivamente de um conjunto específico de neurônios (evitando a "co-adaptação") e força a aprendizagem de características mais robustas e generalizáveis.

**Aumentação de Dados (Data Augmentation):** Para mitigar o overfitting e aumentar a robustez do modelo, a AlexNet empregou duas formas de aumento de dados. A primeira consistia em extrair patches aleatórios de 224x224 (e suas reflexões horizontais) de imagens originais de 256x256, aumentando o tamanho do conjunto de treinamento em 2048 vezes. A segunda técnica envolvia a alteração aleatória dos valores RGB de cada pixel. Essas estratégias foram essenciais para garantir que o modelo, com seus muitos parâmetros, generalizasse bem para novas imagens.

**Uso de Múltiplas GPUs:** Um dos aspectos mais notáveis do AlexNet foi sua implementação distribuída em duas GPUs Nvidia GTX 580. A divisão do modelo não foi uma decisão de design ideal por si só, mas sim uma solução pragmática para uma limitação de hardware. Na época, uma única GPU não tinha memória suficiente (3 GB) para armazenar todos os 60 milhões de parâmetros e os estados de ativação necessários para o treinamento. Essa restrição técnica forçou os autores a encontrar uma solução arquitetural inovadora que não apenas superou o obstáculo, mas também dobrou o poder de processamento disponível, demonstrando como as limitações do hardware podem inspirar novas abordagens de software.

## 3. Análise Crítica: Vantagens e Deficiências da AlexNet

### 3.1. Principais Vantagens

A AlexNet representou um salto qualitativo em relação aos modelos de visão computacional anteriores. Sua principal vantagem foi a precisão sem precedentes que alcançou, estabelecendo um novo padrão e demonstrando o poder de modelos mais profundos para aprender características complexas. A utilização de ReLUs e GPUs tornou o treinamento de redes profundas viável e rápido, acelerando a convergência em comparação com abordagens que usavam funções de ativação tradicionais. Além disso, o emprego estratégico de Dropout, aumento de dados e pooling sobreposto (que reduziu a taxa de erro top-5 em 0,3%) foi fundamental para melhorar a generalização e combater o overfitting, um problema crônico em redes com grande número de parâmetros. O sucesso da AlexNet validou a pesquisa em deep learning, inspirando a criação de uma nova geração de arquiteturas.

### 3.2. Limitações e Desafios

Apesar de seu sucesso, a AlexNet possuía desafios significativos que se tornaram evidentes com o tempo e serviram de catalisador para as inovações que se seguiram. O mais notável foi o seu custo computacional e a dependência de hardware de ponta. O treinamento de cinco a seis dias em GPUs caras era um gargalo para a pesquisa e tornava o modelo inacessível para muitos. Com aproximadamente 60 milhões de parâmetros, o modelo era considerado enorme para sua época. Esse tamanho excessivo resultava em altos requisitos de memória e de computação, tornando a AlexNet impraticável para implantação em dispositivos com recursos limitados ou para aplicações em tempo real.

O desempenho do modelo, embora revolucionário, ainda tinha limitações. A taxa de erro top-5 de 15,3% significava que um número substancial de imagens ainda era classificado incorretamente. O modelo também demonstrava sensibilidade a imagens de baixa resolução, fundos complexos e variações visuais significativas, além de ter uma capacidade limitada de compreensão contextual e vulnerabilidade a exemplos adversariais.

Essas deficiências, longe de serem apenas falhas, abriram novos caminhos de pesquisa. A necessidade de modelos mais eficientes e compactos, com menos parâmetros, impulsionou o desenvolvimento de arquiteturas como o GoogLeNet. A busca por redes ainda mais profundas, que pudessem aprender características mais complexas, levou à solução do problema do gradiente de saturação em larga escala, o que culminou na invenção das redes residuais (ResNet). A análise desses desafios mostra que as limitações de um modelo de ponta podem se tornar os objetivos de pesquisa para a próxima geração de inovações.

## 4. O Legado da AlexNet: A Evolução para Arquiteturas Modernas

A AlexNet foi o ponto de partida para a era moderna do deep learning em visão computacional. As arquiteturas que a sucederam construíram sobre suas inovações, abordando diretamente suas deficiências e empurrando os limites da performance e da eficiência.

**AlexNet vs. VGGNet (2014):** A VGGNet, desenvolvida por Simonyan e Zisserman, ficou em segundo lugar no ILSVRC de 2014. Sua principal contribuição foi a simplificação e o aprofundamento da arquitetura. A VGGNet demonstrou que o uso de múltiplos filtros pequenos (3x3) em camadas sequenciais poderia replicar a função de filtros maiores, como o 11x11 da AlexNet. Essa abordagem permitiu a criação de redes mais profundas, com 16 a 19 camadas, e estabeleceu a profundidade como um fator crucial para a performance. No entanto, a VGGNet tinha 138 milhões de parâmetros, tornando-a ainda mais pesada que a AlexNet.

**AlexNet vs. GoogLeNet/Inception (2014):** O GoogLeNet, vencedor do ILSVRC de 2014, focou na eficiência e na redução de parâmetros. Sua inovação central foi o "módulo Inception", que utilizava convoluções em paralelo de diferentes tamanhos (1x1, 3x3, 5x5) e pooling para capturar características em múltiplas escalas. A GoogLeNet também empregou convoluções de 1x1 como "camadas de gargalo" para reduzir a dimensionalidade. O resultado foi uma rede de 22 camadas, muito mais profunda que a AlexNet, mas com apenas 4 milhões de parâmetros, uma redução dramática em relação aos 60 milhões da AlexNet.

**AlexNet vs. ResNet (2015):** A ResNet, vencedora do ILSVRC de 2015, abordou o problema do gradiente de saturação em redes ultra-profundas de uma forma fundamental. Enquanto a AlexNet tornou possível o treinamento de redes com 8 camadas, a ResNet permitiu a criação de redes com centenas de camadas (como a ResNet-152), com uma profundidade 20 vezes maior que a AlexNet. A principal inovação foram as "skip connections" (ou "conexões de atalho"), que permitiam que os dados de uma camada anterior fossem adicionados à saída de uma camada posterior, garantindo que o gradiente fluísse eficientemente através da rede. A ResNet alcançou uma taxa de erro top-5 de 3,57%, superando o desempenho de nível humano no conjunto de dados do ImageNet.

### Tabela 2: AlexNet vs. Arquiteturas Posteriores

| Arquitetura | Ano | Taxa de Erro Top-5 | Número de Parâmetros | Principal Inovação |
|-------------|-----|-------------------|---------------------|-------------------|
| AlexNet | 2012 | 15.3% | ~60 milhões | ReLU, Dropout, GPU Acceleration |
| VGGNet | 2014 | ~7.3% | ~138 milhões | Uniformidade, uso de 3x3 kernels |
| GoogLeNet | 2014 | 6.67% | ~4 milhões | Módulo Inception, eficiência de parâmetros |
| ResNet | 2015 | 3.57% | ~25 milhões (ResNet-50) | Skip connections (aprendizagem residual) |


## 5. Implementação Prática da AlexNet em PyTorch

Para um curso de deep learning, a implementação da AlexNet em PyTorch pode ser abordada de duas maneiras: utilizando o modelo pré-treinado da biblioteca torchvision para um fluxo de trabalho rápido de inferência ou construindo a arquitetura do zero para um aprendizado mais aprofundado.

### 5.1. Pré-requisitos

Certifique-se de que as seguintes bibliotecas estão instaladas:

- torch
- torchvision
- Pillow (PIL)
- numpy (opcional, mas recomendado)
- matplotlib (opcional para visualização)

### 5.2. Opção A: Utilizando a AlexNet Pré-treinada da torchvision

A abordagem mais direta é carregar o modelo pré-treinado do PyTorch Hub. O modelo alexnet foi treinado no conjunto de dados do ImageNet com 1000 classes e já possui os pesos ajustados para essa tarefa.

#### Código para Carregamento e Inferência

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Carregar o modelo pré-treinado da AlexNet
# O parâmetro 'pretrained=True' foi substituído por 'weights=models.AlexNet_Weights.IMAGENET1K_V1'
# em versões mais recentes do torchvision
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.eval()  # Definir o modelo para o modo de avaliação (desativa dropout, etc.)

# Definir as transformações de pré-processamento de imagem
# Os modelos pré-treinados esperam que as imagens sejam normalizadas com os seguintes valores
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregar e pré-processar uma imagem de exemplo (por exemplo, "cat.jpg")
try:
    img = Image.open('cat.jpg').convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Criar um mini-lote (adicionar dimensão do lote)
    
    # Executar a inferência sem calcular gradientes
    with torch.no_grad():
        output = model(input_batch)
    
    # Obter as probabilidades aplicando a função softmax
    probabilities = torch.nn.functional.softmax(output, dim=0)
    
    # Carregar os rótulos das classes do ImageNet
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Imprimir as 5 principais categorias previstas
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]:<40} {top5_prob[i].item():.4f}")
        
except FileNotFoundError:
    print("O arquivo 'cat.jpg' ou 'imagenet_classes.txt' não foi encontrado.")
```

O pré-processamento das imagens é um passo crítico. Os modelos da torchvision que foram treinados no ImageNet esperam imagens de entrada redimensionadas para 256 pixels, cortadas no centro para 224 pixels e, em seguida, normalizadas com os valores de média e desvio padrão específicos do ImageNet. Ignorar este passo de normalização resultará em um desempenho drasticamente inferior do modelo.

### 5.3. Opção B: Construindo a AlexNet do Zero (from scratch)

Para uma compreensão mais profunda da arquitetura, é instrutivo construir a AlexNet manualmente. O código a seguir mostra a estrutura básica da classe AlexNet em PyTorch, seguindo a arquitetura detalhada na Tabela 1.

#### Estrutura da Classe AlexNet

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Camadas de extração de características (features)
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Camadas totalmente conectadas (classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Alternativa: x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Exemplo de uso
alexnet_from_scratch = AlexNet(num_classes=1000)
print(alexnet_from_scratch)
```

Uma observação crucial para o instrutor do curso é que a implementação oficial da torchvision (disponível no PyTorch Hub) difere sutilmente da arquitetura original, pois é baseada em um trabalho subsequente ("One weird trick for parallelizing convolutional neural networks"). Por exemplo, a versão da torchvision usa uma camada nn.AdaptiveAvgPool2d e omite a Normalização de Resposta Local (LRN). Esta discrepância demonstra que, na prática, as implementações de código podem evoluir a partir do artigo de pesquisa original, com ajustes que melhoram o desempenho ou simplificam a arquitetura.

## Referências

8.1. Deep Convolutional Neural Networks (AlexNet) — Dive into Deep Learning 1.0.3 documentation, acessado em agosto 24, 2025, http://d2l.ai/chapter_convolutional-modern/alexnet.html
AlexNet - Wikipedia, acessado em agosto 24, 2025, https://en.wikipedia.org/wiki/AlexNet
Alex Krizhevsky - Wikipedia, acessado em agosto 24, 2025, https://en.wikipedia.org/wiki/Alex_Krizhevsky
What is AlexNet? AlexNet Architecture Explained - Great Learning, acessado em agosto 24, 2025, https://www.mygreatlearning.com/blog/alexnet-the-first-cnn-to-win-image-net/
CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and ..., acessado em agosto 24, 2025, https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
AlexNet Unraveled: The Breakthrough CNN That Changed Deep Learning Forever | by Anto Jeffrin | Medium, acessado em agosto 24, 2025, https://medium.com/@antojeffrin007/alexnet-unraveled-the-breakthrough-cnn-that-changed-deep-learning-forever-7975f3151ea4
ImageNet Classification with Deep Convolutional Neural Networks - ResearchGate, acessado em agosto 24, 2025, https://www.researchgate.net/publication/267960550_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks
AlexNet Architecture Explained. The convolutional neural network (CNN)… | by Siddhesh Bangar | Medium, acessado em agosto 24, 2025, https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5
Introduction to Alexnet Architecture - Analytics Vidhya, acessado em agosto 24, 2025, https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/
Difference between AlexNet and GoogleNet - GeeksforGeeks, acessado em agosto 24, 2025, https://www.geeksforgeeks.org/deep-learning/difference-between-alexnet-and-googlenet/
[discussion] Why was AlexNet split on two GPUs each of memory size 3GB when it can fit on 1 GB? : r/mlscaling - Reddit, acessado em agosto 24, 2025, https://www.reddit.com/r/mlscaling/comments/1gc6pvk/discussion_why_was_alexnet_split_on_two_gpus_each/
What is Alexnet? | Activeloop Glossary, acessado em agosto 24, 2025, https://www.activeloop.ai/resources/glossary/alexnet/
Evolution of Convolutional Neural Network (CNN) architectures | by ..., acessado em agosto 24, 2025, https://medium.com/@cpt1995daas/evolution-of-convolutional-neural-network-cnn-architectures-44f2109268a1
www.activeloop.ai, acessado em agosto 24, 2025, https://www.activeloop.ai/resources/glossary/alexnet/#:~:text=Some%20drawbacks%20of%20AlexNet%20include,and%20relatively%20slow%20inference%20time.
Challenges with AlexNet - BytePlus, acessado em agosto 24, 2025, https://www.byteplus.com/en/topic/401660
Evolution of CNN Architectures (AlexNet to ResNet) - ApX Machine Learning, acessado em agosto 24, 2025, https://apxml.com/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/cnn-architecture-evolution
AlexNet – PyTorch, acessado em agosto 24, 2025, https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet — Torchvision 0.22 documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.alexnet.html
AlexNet — Torchvision 0.22 documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/0.22/models/alexnet.html
alexnet — Torchvision main documentation, acessado em agosto 24, 2025, https://docs.pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
Implementing AlexNet from Scratch Using PyTorch | Daniel Paricio, acessado em agosto 24, 2025, https://danielparicio.com/posts/implementing-alexnet/