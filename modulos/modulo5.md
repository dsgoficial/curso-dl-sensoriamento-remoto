### **Módulo 5: Introdução às CNNs (2h30min)**

Este módulo introduz as Redes Neurais Convolucionais (CNNs) como a arquitetura fundamental para o processamento de imagens, superando as limitações das Redes Neurais de Múltiplas Camadas (MLPs) para esta tarefa. O conteúdo é estruturado para construir uma compreensão sólida, começando dos conceitos teóricos e culminando em uma implementação prática.

#### **5.1 Da Convolução Clássica às CNNs (1h15min)**

##### **Limitações de MLPs para Imagens** (25min)

Embora MLPs tenham sido a base de redes neurais, elas se mostram ineficientes e inviáveis para tarefas de visão computacional. Vamos explorar as duas principais razões para isso.

###### **Perda de informação espacial ao "achatar" imagens** (10min)

MLPs processam dados de entrada como um vetor unidimensional. Para usar uma imagem em uma MLP, precisamos primeiro "achatá-la" (flatten), transformando a matriz 2D de pixels em um vetor longo. A operação `flatten()` converte uma imagem de dimensões $H \times W \times C$ (altura, largura, canais) em um vetor de tamanho $H \times W \times C$.

O problema crucial é que essa operação descarta completamente a estrutura espacial da imagem. Pixels que eram vizinhos e formavam bordas, texturas ou formas são agora tratados como pontos independentes.

**Analogia:** Imagine que você tem que reconhecer uma face, mas só pode ver uma lista aleatória de todos os valores de pixels que a compõem. Sem saber a posição de cada pixel (ou seja, a estrutura 2D), a tarefa se torna impossível. Da mesma forma, uma MLP luta para entender conceitos como bordas ou texturas, que são padrões inerentemente espaciais, pois a informação de vizinhança foi perdida.

###### **Explosão de parâmetros com imagens grandes** (15min)

Vamos calcular o número de parâmetros para um MLP com uma imagem de satélite típica, por exemplo, uma imagem RGB de $224 \times 224 \times 3$ pixels.

  - **Dimensão da Imagem:** $224 \times 224 \times 3 = 150.528$ pixels.
  - **Entrada para a MLP:** Vetor de 150.528 features.

Consideremos uma primeira camada oculta com apenas 1.000 neurônios. O número de pesos (parâmetros) entre a camada de entrada e esta primeira camada oculta seria:

$\text{Parâmetros} = (\text{Número de Entradas} \times \text{Número de Neurônios na Camada Oculta}) + \text{Número de Neurônios na Camada Oculta (Bias)}$

$\text{Parâmetros} = (150.528 \times 1000) + 1000 = 150.528.000 + 1000 = 150.529.000$

Apenas na primeira camada, temos mais de 150 milhões de parâmetros\! Isso torna o treinamento **computacionalmente intratável** e, pior, a rede se torna extremamente propensa a **overfitting**, pois tem mais parâmetros do que dados suficientes para aprender padrões genéricos. A necessidade de uma arquitetura mais eficiente que explore a estrutura 2D da imagem se torna evidente.

##### **Operação de Convolução Aprendida** (30min)

O insight revolucionário por trás das CNNs é a operação de convolução aprendida.

###### **Conexão explícita com filtros clássicos do Módulo 2** (20min)

No **Módulo 2**, implementamos filtros clássicos (kernels) para extrair características específicas de uma imagem:

  - **Filtro de Sobel:** Detecta bordas horizontais e verticais.
  - **Filtro Gaussiano:** Suaviza a imagem, reduzindo o ruído.
  - **Filtro Laplaciano:** Detecta mudanças bruscas de intensidade.

O insight de Yann LeCun e outros pioneiros foi o seguinte: ao invés de projetar manualmente esses filtros para extrair características, por que não deixar que a própria rede aprenda os filtros ideais para a tarefa?

Uma camada convolucional em uma CNN é essencialmente um **banco de filtros que são aprendidos automaticamente** durante o treinamento. A rede começa com filtros aleatórios e, através do backpropagation, ajusta seus valores para minimizar o erro de previsão. Durante o treinamento, alguns filtros podem evoluir para se tornarem detectores de bordas (semelhantes ao Sobel), outros podem aprender a detectar cantos, texturas ou padrões de cores.

A operação matemática da convolução 2D é definida por:

$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)$

Onde $I$ é a matriz de entrada, $K$ é o filtro (kernel) e o resultado é uma nova matriz, chamada de **feature map**.

###### **Conceitos fundamentais: padding, stride, receptive field** (10min)

  - **Padding:** Preenchimento da imagem de entrada com zeros nas bordas para controlar as dimensões de saída. `Padding='same'` garante que a imagem de saída tenha o mesmo tamanho que a de entrada. `Padding='valid'` não adiciona preenchimento, resultando em uma saída menor.

  - **Stride:** Define o "passo" que o filtro dá ao deslizar sobre a imagem. Um `stride` de 2, por exemplo, faz com que o filtro pule 2 pixels por vez, resultando em uma saída de menor dimensão (downsampling).

A dimensão de saída $O$ de uma camada convolucional é calculada por:

$O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$

Onde $I$ é a dimensão de entrada, $K$ o tamanho do kernel, $P$ o padding e $S$ o stride.

  - **Receptive Field (RF):** A região da imagem de entrada que influencia a ativação de um neurônio específico em uma camada subsequente. Em uma camada convolucional, o RF é o tamanho do kernel. Em uma rede mais profunda, o RF de uma ativação é o tamanho da área na imagem original que a camada "vê". O RF cresce à medida que a profundidade da rede aumenta, permitindo que camadas profundas combinem características locais em padrões mais complexos.

##### **Arquitetura CNN Básica** (20min)

A arquitetura típica de uma CNN é uma sequência de blocos de construção.

###### **Camadas convolucionais, pooling e fully connected** (15min)

  - **Camadas Convolucionais (CONV):** O coração da CNN. Elas aplicam múltiplos filtros aprendidos à imagem de entrada para produzir **feature maps**. A saída de uma camada CONV é um conjunto de feature maps, cada um destacando uma característica diferente (bordas, texturas, etc.) na imagem.
  - **Camadas de Pooling (POOL):** Geralmente inseridas entre camadas CONV. Seu objetivo é reduzir a dimensionalidade espacial dos feature maps (downsampling). As operações mais comuns são o **max pooling** (seleciona o valor máximo em uma janela) e o **average pooling** (calcula a média). O pooling oferece uma forma de **invariância à translação**, tornando a rede mais robusta a pequenas variações na posição dos objetos.
  - **Camadas Totalmente Conectadas (FC):** No final da rede, os feature maps são "achatados" e conectados a uma ou mais camadas densas (MLPs). Essas camadas densas usam as características de alto nível extraídas pelas camadas CONV e POOL para realizar a tarefa final, como a classificação.

O padrão arquitetural típico de uma CNN é: `CONV -> POOL -> CONV -> POOL -> FC -> FC -> OUTPUT`.

###### **Feature maps e hierarquia de características** (5min)

Cada filtro em uma camada convolucional produz um **feature map**, que é uma representação da imagem original onde as características detectadas pelo filtro são realçadas.

  - **Camadas Iniciais:** Aprendem a detectar características simples e genéricas, como bordas, cantos e texturas.
  - **Camadas Intermediárias:** Combinam as características simples em padrões mais complexos, como formas e partes de objetos.
  - **Camadas Profundas:** Combinam as características intermediárias em representações de alto nível, como objetos completos (ex: um carro, uma casa) ou cenas inteiras.

Essa **hierarquia de características** é o conceito-chave que dá às CNNs seu poder para tarefas complexas de visão.

-----

#### **5.2 LeNet no MNIST: Primeira CNN Prática (1h15min)**

Para solidificar os conceitos teóricos, vamos implementar e analisar a arquitetura CNN pioneira, a **LeNet-5**, no clássico dataset MNIST.

##### **Implementação LeNet-5** (45min)

###### **Arquitetura detalhada e motivação histórica** (15min)

A LeNet-5, proposta por Yann LeCun em 1998, foi uma das primeiras CNNs de sucesso, projetada para reconhecimento de dígitos manuscritos. Sua arquitetura é:

1.  **Input:** Imagem de 32x32 pixels (MNIST é 28x28, então é necessário um padding).
2.  **C1 (Convolução):** 6 filtros de $5 \times 5$, `stride=1`. Output: 6 feature maps de $28 \times 28$.
3.  **S2 (Average Pooling):** Filtro de $2 \times 2$, `stride=2`. Output: 6 feature maps de $14 \times 14$.
4.  **C3 (Convolução):** 16 filtros de $5 \times 5$. Output: 16 feature maps de $10 \times 10$.
5.  **S4 (Average Pooling):** Filtro de $2 \times 2$, `stride=2`. Output: 16 feature maps de $5 \times 5$.
6.  **C5 (Convolução):** 120 filtros de $5 \times 5$ (tratada como camada FC). Output: 120 neurônios.
7.  **F6 (Fully Connected):** 84 neurônios.
8.  **Output (Fully Connected):** 10 neurônios (1 por classe, 0-9) com ativação Softmax.

###### **Implementação guiada: LeNet from scratch** (20min)

Vamos implementar a LeNet-5 usando a biblioteca **PyTorch**. O código abaixo demonstra a estrutura de classes e o método `forward()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Camada Convolucional C1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # Camada de Pooling S2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Camada Convolucional C3
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # Camada de Pooling S4
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Camada Convolucional C5 (tratada como FC)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        # Camadas Fully Connected (FC)
        self.fc6 = nn.Linear(in_features=120, out_features=84)
        self.fc7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # x shape: [batch_size, 1, 32, 32]
        x = F.relu(self.conv1(x)) # Output: [batch_size, 6, 28, 28]
        x = self.pool2(x)         # Output: [batch_size, 6, 14, 14]
        x = F.relu(self.conv3(x)) # Output: [batch_size, 16, 10, 10]
        x = self.pool4(x)         # Output: [batch_size, 16, 5, 5]
        x = F.relu(self.conv5(x)) # Output: [batch_size, 120, 1, 1]
        
        # Achata o tensor para a camada FC
        x = x.view(-1, 120)     # Output: [batch_size, 120]
        
        x = F.relu(self.fc6(x))   # Output: [batch_size, 84]
        x = self.fc7(x)           # Output: [batch_size, 10]
        return x

```

Os parâmetros-chave das camadas são:

  - `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
  - `nn.AvgPool2d(kernel_size, stride)`
  - `nn.Linear(in_features, out_features)`

O método `forward()` rastreia as dimensões do tensor em cada etapa, uma prática essencial para entender o fluxo de dados na rede.

###### **Adaptação do training loop para CNNs** (10min)

O loop de treinamento em si é similar ao de MLPs, mas a preparação dos dados de entrada é crucial. As imagens MNIST são 28x28, enquanto a LeNet-5 espera 32x32. Usamos `torchvision.transforms.Resize((32,32))` para fazer este padding.

```python
from torchvision import transforms

# Transformações de pré-processamento para o dataset MNIST
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# O dataloader fornece lotes (batches) de imagens com o formato [batch_size, 1, 32, 32]
# O método forward() lida com este formato e o reshape no final (`.view(-1, 120)`)
# o prepara para a camada FC.
```

##### **Análise Comparativa** (30min)

###### **MLP vs CNN no mesmo dataset MNIST** (15min)

Uma comparação empírica de MLP e LeNet-5 no dataset MNIST demonstra as vantagens da arquitetura convolucional.

| Arquitetura | Parâmetros | Acurácia (em%) | Vantagens |
| :--- | :--- | :--- | :--- |
| **MLP** (300/100 neurônios) | \~238.000 | \~98,00 | Simplicidade |
| **LeNet-5** | \~62.000 | **\~99,20** | Menos parâmetros, maior precisão, robustez |

A LeNet-5 atinge maior acurácia com um número de parâmetros significativamente menor, o que a torna mais eficiente e menos propensa ao overfitting.

###### **Visualização de feature maps e filtros aprendidos** (10min)

Uma das maiores vantagens das CNNs é sua interpretabilidade. Podemos extrair e visualizar os filtros e os feature maps gerados por eles.

  - **Visualização de Filtros:** Os filtros aprendidos na primeira camada da LeNet-5 frequentemente se assemelham a detectores de bordas, cantos e blobs, confirmando a teoria de que a rede aprende a extrair características de baixo nível.
  - **Visualização de Feature Maps:** Ao passar uma imagem de entrada por uma camada convolucional, podemos visualizar os feature maps resultantes. Cada feature map mostra onde o filtro correspondente foi ativado na imagem, realçando as características que ele detecta. Isso nos dá uma visão interna de como a rede "enxerga" e processa a imagem.

<!-- end list -->

```python
# Exemplo de código para visualizar filtros da primeira camada (conv1)
weights = model.conv1.weight.data
fig = plt.figure(figsize=(8, 4))
for i in range(6): # Visualiza os 6 filtros
    ax = fig.add_subplot(1, 6, i + 1)
    ax.imshow(weights[i, 0, :, :].cpu(), cmap='gray')
    ax.axis('off')
plt.show()
```

###### **Preparação conceitual: Por que CNNs são ideais para sensoriamento remoto** (5min)

O sucesso da CNN em imagens como as do MNIST pode ser diretamente extrapolado para o sensoriamento remoto. Imagens de satélite e aéreas possuem estrutura espacial crucial para a interpretação:

  - **Estradas e rios:** Características lineares.
  - **Campos agrícolas:** Texturas consistentes.
  - **Áreas urbanas:** Padrões característicos de construção.

As CNNs são ideais para essas tarefas porque conseguem aprender automaticamente esses padrões espaciais e a hierarquia de características (pixel → textura → objeto → cena) a partir dos dados, tornando-as a arquitetura de escolha para classificação de cobertura do solo, detecção de mudanças, reconhecimento de objetos e outras aplicações em sensoriamento remoto.