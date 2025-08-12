# Módulo 1: Fundamentos e Contexto (4h30min)

## 1.1 Introdução ao Curso e Evolução da IA (1h10min)

### Contextualização e Evolução da IA (50min)

### Das Origens ao Deep Learning (25min)

A jornada da Inteligência Artificial (IA) é uma tapeçaria rica em inovações, expectativas e desafios. As origens do Deep Learning podem ser traçadas até as primeiras tentativas de simular o cérebro humano.

O **Perceptron**, introduzido por Frank Rosenblatt em 1957, foi um dos primeiros algoritmos para aprendizado supervisionado de classificadores binários. Ele funcionava como um classificador linear, simulando um neurônio artificial capaz de tomar decisões simples. Rosenblatt chegou a construir uma máquina dedicada, o Mark I Perceptron, demonstrado publicamente em 1960.

No entanto, o otimismo inicial em torno do Perceptron foi abalado. As décadas de 1970 e 1980 foram marcadas pelo que ficou conhecido como "Inverno da IA", um período de desilusão e cortes de financiamento. O livro "Perceptrons" (1969), de Marvin Minsky e Seymour Papert, demonstrou as limitações fundamentais dos perceptrons de camada única, provando matematicamente que eles não podiam resolver problemas não-lineares simples, como o problema XOR. Essa demonstração foi um "golpe mortal" para a pesquisa em redes neurais na época. Outros fatores, como a limitada capacidade computacional e o "gargalo na aquisição de conhecimento" em sistemas especialistas, também contribuíram para esse ceticismo.

O renascimento das redes neurais ocorreu nos anos 1980, impulsionado pela redescoberta e popularização do algoritmo de **Backpropagation**. Embora o algoritmo já existisse, sua importância não foi totalmente reconhecida até a publicação de um artigo seminal em 1986 por David Rumelhart, Geoffrey Hinton e Ronald Williams. Este trabalho demonstrou que o backpropagation funcionava de forma significativamente mais rápida do que as abordagens anteriores, tornando possível treinar redes multicamadas e resolver problemas antes considerados insolúveis.

O verdadeiro ponto de virada, que culminou na revolução do Deep Learning, ocorreu em 2012 com a **AlexNet** no desafio ImageNet. A AlexNet não apenas venceu a competição, mas o fez com uma margem de erro significativamente menor do que os concorrentes, estabelecendo o Deep Learning como o paradigma dominante em visão computacional. Esse evento demonstrou a aplicação prática e o poder das redes neurais profundas em larga escala, marcando o fim do "Inverno da IA" e o início da era atual do Deep Learning.

### Marcos Históricos em Visão Computacional (15min)

A visão computacional foi um dos campos mais transformados pela ascensão do Deep Learning, com marcos arquitetônicos que definiram o estado da arte:

- **ImageNet: O Dataset que Mudou Tudo**: A ImageNet é um banco de dados de imagens em larga escala, iniciado por Fei-Fei Li em 2007, que se tornou um catalisador para a revolução do Deep Learning. Com mais de 14 milhões de imagens anotadas em milhares de categorias, a ImageNet forneceu o volume e a diversidade de dados necessários para treinar modelos de Deep Learning complexos. O ImageNet Large Scale Visual Recognition Challenge (ILSVRC), uma competição anual baseada nesse conjunto de dados, foi instrumental no avanço da pesquisa, impulsionando o desenvolvimento de modelos de ponta.

- **LeNet: A Pioneira do Reconhecimento de Dígitos**: A LeNet, desenvolvida por Yann LeCun e sua equipe entre 1988 e 1998, foi uma das primeiras Redes Neurais Convolucionais (CNNs). A LeNet-5 foi projetada para reconhecimento de dígitos manuscritos e introduziu conceitos fundamentais como camadas convolucionais e de pooling, campos receptivos locais e pesos compartilhados, que são padrão em CNNs modernas. Seu sucesso em aplicações práticas, como a leitura de cheques em caixas eletrônicos, validou o potencial das CNNs.

- **AlexNet: A Revolução de 2012**: Como mencionado, a AlexNet foi a CNN que dominou o ImageNet Challenge de 2012. Suas inovações incluíram o uso de ReLU (Rectified Linear Units) como função de ativação para acelerar o treinamento, a aplicação de Dropout para prevenir o overfitting, e o aproveitamento do poder de processamento paralelo das GPUs para treinar em grandes conjuntos de dados. A AlexNet demonstrou que redes mais profundas e complexas eram viáveis e podiam alcançar resultados sem precedentes.

- **VGGNet: A Profundidade Importa**: A VGGNet, desenvolvida em 2014, demonstrou que a profundidade da rede era um fator crucial para o desempenho. Ao empilhar consistentemente filtros convolucionais pequenos (3x3) em múltiplas camadas, a VGGNet alcançou alta precisão em tarefas de classificação de imagens, solidificando a ideia de que redes mais profundas podiam aprender representações mais ricas e complexas.

- **ResNet: Resolvendo Gradientes Evanescentes**: A arquitetura ResNet (Residual Neural Network), introduzida em 2015, revolucionou o Deep Learning ao resolver o problema do gradiente desvanecente em redes muito profundas. Ela introduziu as **conexões residuais** ou **skip connections**, que permitem que o gradiente flua diretamente das camadas mais profundas para as mais rasas durante a retropropagação, possibilitando o treinamento de redes com mais de 100 camadas.

- **EfficientNet: Otimizando a Eficiência Computacional**: A EfficientNet, lançada em 2019, focou em otimizar o equilíbrio entre eficiência computacional e desempenho do modelo. Ela introduziu a técnica de "compound scaling" (escala composta), que escala sistematicamente a largura, profundidade e resolução da rede de forma balanceada, permitindo modelos de alto desempenho com menos parâmetros e recursos computacionais.

- **Estado Atual: SAM e Modelos Multimodais**: O campo continua a evoluir com modelos como o Segment Anything Model (SAM), que é um sistema de segmentação promptable com generalização zero-shot para objetos e imagens desconhecidas, sem a necessidade de treinamento adicional. Além disso, a tendência crescente de **modelos multimodais** (como PaliGemma, GPT-4o, CLIP), que integram informações de diferentes modalidades (e.g., imagem e texto), está abrindo novas fronteiras na compreensão visual e linguística.

### Deep Learning no Sensoriamento Remoto (10min)

A aplicação do Deep Learning ao Sensoriamento Remoto representa uma transição histórica e um avanço significativo na análise de dados geoespaciais.

Tradicionalmente, os métodos de sensoriamento remoto eram predominantemente **pixel-based**, focando na classificação e análise de pixels individuais com base em seus valores espectrais. Embora eficazes para certas tarefas, essas abordagens frequentemente ignoravam o contexto espacial e tinham limitações significativas ao lidar com imagens de muito alta resolução (VHR), resultando em saídas ruidosas e o "efeito sal e pimenta".

A transição para abordagens baseadas em **CNNs** no sensoriamento remoto foi impulsionada pela capacidade dessas redes de aprender automaticamente características hierárquicas e contextuais a partir de dados complexos. As CNNs, especialmente as **3D-CNNs**, demonstraram ser potentes para processar dados espaço-temporais, capturando características tanto espaciais quanto temporais simultaneamente, o que é crucial para tarefas como detecção de mudanças.

No entanto, essa transição não foi isenta de desafios:

- **Dados Multiespectrais e Diferentes Resoluções**: O sensoriamento remoto lida com dados de múltiplas bandas espectrais e diversas resoluções (espacial, temporal, radiométrica), o que exige modelos capazes de processar informações espectrais complexas e adaptar-se a variações ambientais e sazonais. A complexidade computacional de processar múltiplas arquiteturas CNN para dados multiespectrais é uma preocupação.

- **Necessidade de Grandes Áreas de Treinamento**: A coleta de dados de treinamento de alta qualidade e com anotações precisas é um gargalo significativo, especialmente para classificação de uso e cobertura do solo em grandes áreas. Os dados de treinamento devem ter resolução espacial superior à dos dados de satélite a serem classificados.

Apesar desses desafios, as **tendências atuais** no Deep Learning para sensoriamento remoto são promissoras:

- **Modelos de Fundação (Foundation Models)**: Modelos grandes, pré-treinados em vastos conjuntos de dados não rotulados, que podem ser ajustados para diversas tarefas downstream com poucos dados específicos. Eles prometem democratizar a análise geoespacial avançada, simplificando os requisitos técnicos.

- **Técnicas de Self-Supervised Learning (SSL)**: O SSL permite aprender representações profundas a partir de dados não rotulados, criando "tarefas pretexto" onde os rótulos são gerados automaticamente. Isso é crucial para superar a limitação de dados anotados, que é um gargalo no sensoriamento remoto.

- **Crescente Importância de Dados de Alta Resolução Temporal**: A fusão espaço-temporal (STF) de sensoriamento remoto, impulsionada pelo Deep Learning, aborda a compensação entre resolução temporal e espacial, combinando imagens de diferentes resoluções para monitorar processos dinâmicos na superfície terrestre.

### Visão do Curso e Objetivos (20min)

Este curso de Deep Learning Aplicado ao Sensoriamento Remoto foi cuidadosamente estruturado para oferecer uma experiência didática **hands-on**, enfatizando uma progressão lógica de conceitos básicos para aplicações avançadas. Nosso objetivo é capacitar os alunos com as competências essenciais para atuar na vanguarda da análise geoespacial moderna.

Ao final deste curso, os alunos desenvolverão as seguintes competências:

- **Processamento de Dados Geoespaciais**: Habilidade para manipular e preparar diversos tipos de dados de sensoriamento remoto, incluindo imagens multiespectrais, SAR e LiDAR, para análise com Deep Learning.

- **Construção de Pipelines de Deep Learning para Sensoriamento Remoto**: Capacidade de projetar, implementar e treinar modelos de Deep Learning (especialmente CNNs) adaptados às peculiaridades dos dados geoespaciais.

- **Avaliação de Modelos com Métricas Adequadas**: Conhecimento e aplicação de métricas de avaliação específicas para tarefas de sensoriamento remoto, garantindo a robustez e a confiabilidade dos modelos.

- **Preparação de Dados Reais para Produção**: Experiência prática na curadoria, pré-processamento e aumento de dados de sensoriamento remoto para cenários de aplicação em larga escala.

Ao longo do curso, os alunos implementarão projetos práticos que abrangem desde a classificação de uso e cobertura do solo até a detecção de mudanças e a segmentação de objetos em imagens de satélite. Esses projetos serão construídos com base em dados reais, proporcionando uma experiência imersiva e relevante para o mercado de trabalho. Nossa meta é criar uma expectativa e motivação contínuas para o aprendizado, mostrando como o Deep Learning pode ser uma ferramenta transformadora para resolver desafios ambientais e geoespaciais complexos.

## 1.2 Setup do Ambiente (20min)

### Google Colab - Setup Rápido (10min)

O Google Colaboratory (Colab) é um serviço de Jupyter Notebook hospedado que não requer configuração e oferece acesso gratuito a recursos computacionais, incluindo GPUs e TPUs. É uma ferramenta ideal para machine learning, ciência de dados e educação, facilitando o compartilhamento de notebooks. O código é executado em máquinas virtuais nos servidores da nuvem do Google, permitindo alavancar o poder de hardware avançado independentemente da capacidade da máquina local do usuário.

Para começar a usar o Google Colab:

1. **Acesso e Criação do Primeiro Notebook**: Acesse [colab.research.google.com](https://colab.research.google.com) e faça login com sua conta Google. Você pode criar um novo notebook clicando em "File > New notebook".

2. **Ativar e Verificar a Disponibilidade de GPU**: Para utilizar uma GPU, você deve alterar o tipo de ambiente de execução. Vá em Runtime > Change runtime type no menu e selecione GPU como acelerador de hardware. A disponibilidade de GPUs pode variar na versão gratuita do Colab. Para verificar qual GPU foi atribuída e seu status, execute o seguinte comando em uma célula de código:

```bash
!nvidia-smi
```

Este comando exibirá informações detalhadas sobre a GPU, incluindo uso de memória e processos em execução. A utilização de GPUs é crucial para o Deep Learning devido à sua capacidade de processamento paralelo, que acelera significativamente o treinamento de redes neurais.

3. **Montar o Google Drive para Persistência de Dados**: Para acessar arquivos armazenados no seu Google Drive e garantir a persistência de dados entre as sessões do Colab, você pode montar o Drive no ambiente do Colab. Execute o seguinte snippet de código em uma célula:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Após a execução, você será solicitado a autorizar o acesso à sua conta do Google. Uma vez autenticado, seus arquivos do Drive estarão acessíveis através de caminhos como `/content/drive/My Drive/`. Para otimizar o desempenho, é recomendável copiar grandes volumes de dados do Drive para o sistema de arquivos local da VM do Colab, evitando muitas leituras e escritas pequenas diretamente do Drive.

4. **Verificações Básicas de Hardware e Monitoramento de Uso**: Você pode monitorar o uso de RAM e GPU durante seus experimentos. O comando `!nvidia-smi` já fornece informações sobre a GPU. Para a RAM, você pode usar a biblioteca `psutil`:

```python
import psutil
ram_gb = psutil.virtual_memory().total / 1e9
print(f'Sua runtime tem {ram_gb:.1f} gigabytes de RAM disponível\n')
```

É importante estar ciente dos limites de recursos dinâmicos do Colab, que podem flutuar, e desconectar-se do ambiente de execução quando não estiver em uso para evitar atingir esses limites.

### Instalação de Bibliotecas (10min)

Para garantir que todos os alunos tenham o ambiente de desenvolvimento configurado de forma consistente, é fundamental apresentar um script automatizado para a instalação das bibliotecas necessárias ao curso.

1. **Script Automatizado de Instalação**: É uma boa prática criar um script que instale todas as dependências de uma vez. Você pode usar o pip para isso. Um exemplo de script que pode ser executado em uma célula do Colab:

```python
# Instalação das bibliotecas essenciais para o curso
!pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python rasterio geopandas
!pip install --upgrade tensorflow # Garante a versão mais recente do TensorFlow
!pip install --upgrade scikit-image # Garante a versão mais recente do scikit-image
```

Este script instala bibliotecas comuns para manipulação de dados (`numpy`, `pandas`), visualização (`matplotlib`), machine learning (`scikit-learn`), deep learning (`tensorflow`), processamento de imagens (`opencv-python`, `scikit-image`), e dados geoespaciais (`rasterio`, `geopandas`).

2. **Importância de Verificar Versões**: A reprodutibilidade é crucial em projetos de Deep Learning. Diferentes versões de bibliotecas podem levar a comportamentos inesperados ou erros. É importante verificar as versões das bibliotecas instaladas:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import rasterio
import geopandas as gpd

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Rasterio version: {rasterio.__version__}")
print(f"Geopandas version: {gpd.__version__}")
```

3. **Template de Notebook com Imports Essenciais**: Criar um template de notebook com todos os imports essenciais organizados por categoria ajuda a manter o código limpo e legível desde o início.

```python
# Imports essenciais para o curso de Deep Learning Aplicado ao Sensoriamento Remoto

# 1. Manipulação de Dados e Computação Numérica
import numpy as np
import pandas as pd

# 2. Deep Learning Frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 3. Processamento de Imagens e Visão Computacional
import cv2
from skimage import io, transform

# 4. Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# 5. Geoespacial e Sensoriamento Remoto
import rasterio
import geopandas as gpd
from rasterio.plot import show

# 6. Utilitários
import os
import sys
import time
```

4. **Boas Práticas de Organização de Código**:
   - **Comentários**: Adicione comentários explicativos ao código.
   - **Nomenclatura**: Use nomes de variáveis e funções descritivos.
   - **Modularização**: Divida o código em funções e classes para maior organização.
   - **Consistência**: Mantenha um estilo de codificação consistente em todo o projeto.

Estabelecer essas boas práticas desde o início é fundamental para o desenvolvimento de projetos de Deep Learning robustos e de fácil manutenção.

## 1.3 Revisão Matemática com NumPy (3h)

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

#### Broadcasting e vetorização eficiente (20min)

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

A ênfase na vetorização e no broadcasting não é apenas uma questão de "código mais limpo", mas um princípio de design fundamental para a eficiência computacional em Deep Learning. Modelos de Deep Learning envolvem milhões ou bilhões de parâmetros e operam em conjuntos de dados massivos. Sem essas operações otimizadas, os tempos de treinamento seriam proibitivos.

#### Prática no Colab: Implementação de operações básicas (15min)

Nesta seção, os alunos realizarão exercícios práticos no Google Colab para consolidar os conceitos de representação de dados e operações básicas com NumPy.

**Exercícios Sugeridos:**

1. **Criação de Matrizes e Tensores:**
   - Crie um vetor 1D com 5 elementos.
   - Crie uma matriz 2D (3x3) preenchida com zeros.
   - Crie um tensor 3D que simule uma imagem colorida (por exemplo, 64x64 pixels com 3 canais).
   - Crie um tensor 4D que simule um lote de 16 imagens em escala de cinza (por exemplo, 32x32 pixels).

2. **Operações Matriciais:**
   - Crie duas matrizes 2x2 e realize a multiplicação de matrizes entre elas.
   - Crie duas matrizes 3x3 e realize o produto elemento a elemento (Hadamard).
   - Crie uma matriz 2x4 e obtenha sua transposta.
   - Realize operações de soma, subtração e escalonamento com matrizes e escalares.

3. **Broadcasting em Cenários Reais:**
   - Crie uma matriz de ativações (por exemplo, 5x10) e adicione um vetor de viés (1x10) usando broadcasting.
   - Simule um conjunto de dados (por exemplo, 100 amostras, 5 características) e normalize cada característica subtraindo sua média e dividindo pelo desvio padrão, utilizando broadcasting.
   - Crie um tensor que represente uma imagem (por exemplo, 100x100x3) e aplique um fator de brilho diferente para cada canal de cor usando broadcasting.

4. **Medição de Performance (Opcional, para demonstração):**
   - Compare o tempo de execução de uma operação (por exemplo, soma de dois arrays grandes) usando um loop for explícito em Python versus a operação vetorizada do NumPy. Isso demonstrará a importância da vetorização para eficiência.

Esses exercícios práticos ajudarão a solidificar a compreensão dos conceitos e a familiaridade com a sintaxe do NumPy, preparando os alunos para implementações mais complexas em redes neurais.

### Cálculo para Deep Learning (1h10min)

O cálculo diferencial é a espinha dorsal da otimização em Deep Learning, fornecendo as ferramentas para que as redes neurais aprendam e melhorem seu desempenho.

#### Derivadas e gradientes - intuição visual (20min)

- **Derivada**: A derivada de uma função mede a taxa de variação instantânea de uma função em relação à sua entrada. Em termos simples, ela indica o quanto a função muda para uma pequena variação na entrada, e seu sinal indica a direção dessa mudança (se a função está aumentando ou diminuindo).

Matematicamente, a derivada de uma função f(x) em um ponto x é definida como:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Intuição Visual**: Imagine uma função f(x) plotada em um gráfico. A derivada em um ponto específico é a inclinação da linha tangente à curva nesse ponto.
- Se a derivada é positiva, a função está subindo (aumentando).
- Se a derivada é negativa, a função está descendo (diminuindo).
- Se a derivada é zero, a função está em um ponto de máximo, mínimo ou sela (plano).

- **Gradiente**: O gradiente é a generalização da derivada para funções com múltiplas variáveis de entrada (funções multivariadas). O gradiente é um vetor que contém as derivadas parciais da função em relação a cada uma de suas variáveis de entrada.

Para uma função f(x₁,x₂,...,xₙ), o gradiente é denotado por ∇f e é definido como:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$

**Intuição Visual**: Imagine uma paisagem montanhosa (uma função com múltiplas entradas e uma saída, como uma função de custo). O gradiente em um ponto específico dessa paisagem aponta na direção da maior inclinação, ou seja, a direção de maior crescimento da função. Se você estivesse em um ponto da montanha e quisesse subir o mais rápido possível, o gradiente indicaria a direção a seguir.

No Deep Learning, o objetivo é minimizar uma função de custo (ou perda), que quantifica o erro do modelo. Portanto, o interesse recai sobre o negativo do gradiente (−∇f), que aponta na direção da maior descida, permitindo que o modelo se mova em direção a um erro menor. O gradiente funciona como uma "bússola" no espaço de alta dimensionalidade dos parâmetros do modelo, guiando o processo de otimização para o "vale" da função de perda.

#### Regra da cadeia - conceito fundamental (25min)

A **Regra da Cadeia** é uma ferramenta matemática fundamental para calcular derivadas de funções compostas. Em redes neurais, onde temos múltiplas camadas de transformações compostas (a saída de uma camada é a entrada da próxima), a regra da cadeia é essencial para o algoritmo de Backpropagation.

- **Conceito Básico**: Se temos uma função y=f(u) e u=g(x), então a derivada de y em relação a x é dada por:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

- **Generalização para Múltiplas Variáveis**: Se uma função z depende de variáveis u e v, e u e v por sua vez dependem de x, a derivada parcial de z em relação a x é:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$$

- **Relevância para Redes Neurais e Backpropagation**: Em uma rede neural, a função de custo (C) depende da saída da rede, que por sua vez depende das ativações das camadas anteriores, e estas dependem dos pesos e vieses. O Backpropagation é essencialmente uma aplicação sistemática da regra da cadeia para computar o gradiente da função de custo em relação a cada peso e viés na rede.

**Intuição Visual da Propagação**: Imagine uma rede neural como uma série de operações encadeadas. Quando calculamos o erro na saída da rede, queremos saber como esse erro é influenciado por cada peso e viés nas camadas anteriores. A regra da cadeia nos permite "propagar" essa informação de erro (gradiente) para trás através da rede, camada por camada.

Por exemplo, para calcular o erro em um neurônio na camada l (δⱼˡ), o Backpropagation usa a regra da cadeia para relacionar esse erro ao erro na camada seguinte (l+1):

$$\delta_j^l = \sum_k w_{kj}^{l+1} \delta_k^{l+1} \sigma'(z_j^l)$$

Onde δⱼˡ é o erro do j-ésimo neurônio na camada l, wₖⱼˡ⁺¹ é o peso da conexão do j-ésimo neurônio da camada l para o k-ésimo neurônio da camada l+1, δₖˡ⁺¹ é o erro do k-ésimo neurônio na camada l+1, e σ'(zⱼˡ) é a derivada da função de ativação do j-ésimo neurônio na camada l em relação à sua entrada ponderada.

Essa capacidade de propagar eficientemente os gradientes para trás através de múltiplas camadas é o que torna o treinamento de redes neurais profundas viável, permitindo que os pesos e vieses sejam ajustados de forma a minimizar a função de custo.

#### Otimização e gradiente descendente (15min)

O **Gradiente Descendente** é o algoritmo de otimização fundamental empregado para minimizar uma função de custo (ou perda) em modelos de Deep Learning, ajustando iterativamente os parâmetros do modelo (pesos e vieses).

O processo do Gradiente Descendente segue uma sequência iterativa:

1. **Cálculo da Perda**: A função de custo é avaliada com base nos parâmetros atuais do modelo, quantificando o erro das previsões em relação aos valores reais.

2. **Cálculo do Gradiente**: O gradiente da função de custo em relação a cada parâmetro é determinado. Este vetor gradiente indica a direção e a magnitude da inclinação da função de custo no ponto atual do espaço de parâmetros.

3. **Atualização dos Parâmetros**: Os parâmetros do modelo são ajustados em uma pequena quantidade na direção oposta ao gradiente, ou seja, descendo a "inclinação" da função de custo. A fórmula de atualização é tipicamente:

$$\theta_{new} = \theta_{old} - \eta \nabla J(\theta_{old})$$

Onde:
- θₙₑw são os novos valores dos parâmetros (pesos e vieses).
- θₒₗd são os valores atuais dos parâmetros.
- η (eta) é a taxa de aprendizagem (learning rate).
- ∇J(θₒₗd) é o gradiente da função de custo J em relação aos parâmetros θₒₗd.

4. **Iteração**: Os passos são repetidos em ciclos contínuos até que a função de custo não possa ser reduzida significativamente, indicando que o modelo alcançou a convergência.

- **Taxa de Aprendizagem (Learning Rate, η)**: Este é um hiperparâmetro crucial que determina o tamanho do passo dado na direção do gradiente negativo. Uma taxa de aprendizagem muito alta pode levar a oscilações excessivas ou a ultrapassar o mínimo da função de custo, resultando em volatilidade ou divergência do processo de treinamento. Por outro lado, uma taxa muito baixa pode tornar o treinamento extremamente lento e potencialmente fazer com que o algoritmo fique preso em mínimos locais, sem conseguir alcançar o mínimo global.

- **Convergência**: Refere-se ao ponto no processo de treinamento em que iterações adicionais do Gradiente Descendente não resultam em uma redução significativa da perda. Isso indica que o algoritmo encontrou um conjunto de parâmetros que minimiza a função de custo, ou pelo menos um mínimo local aceitável. Em funções de custo convexas, o Gradiente Descendente garante a convergência para o mínimo global se a taxa de aprendizagem for apropriadamente escolhida.

A intuição visual para esses conceitos pode ser imaginada como estar em uma montanha em um dia de neblina e desejar chegar ao vale mais baixo. O gradiente indica a direção mais íngreme para baixo. A taxa de aprendizagem representa o tamanho do seu passo. Se os passos forem muito grandes, há o risco de pular o vale ou de oscilar descontroladamente. Se forem muito pequenos, levará uma quantidade excessiva de tempo para chegar ao fundo.

#### Prática no Colab: Visualização de gradiente descendente (10min)

Nesta seção, os alunos implementarão o algoritmo de Gradiente Descendente para uma função simples e visualizarão seu comportamento no Google Colab.

**Exercício Sugerido:**

1. **Implementação de Gradiente Descendente para uma Função Simples:**
   - Escolha uma função de custo convexa simples, como J(x)=x²+2x+1.
   - Implemente o algoritmo de Gradiente Descendente para encontrar o mínimo dessa função.
   - Armazene os valores de x e J(x) em cada iteração.

2. **Visualização Animada da Convergência:**
   - Utilize matplotlib para criar um gráfico da função de custo.
   - Adicione pontos que representem a posição do algoritmo em cada iteração, criando uma animação que mostre como ele se move em direção ao mínimo.

3. **Experimentação com Taxas de Aprendizagem:**
   - Execute o algoritmo com diferentes valores de taxa de aprendizagem (por exemplo, 0.01, 0.1, 0.5, 1.0).
   - Observe e discuta o impacto da taxa de aprendizagem na velocidade de convergência e na estabilidade do algoritmo (convergência lenta, rápida, oscilação, divergência).

Esta prática concreta consolidará a intuição matemática por trás do Gradiente Descendente e preparará os alunos para compreender otimizadores mais complexos em redes neurais.

### Probabilidade e Estatística (30min)

Distribuições de probabilidade desempenham um papel fundamental no machine learning para modelar a incerteza dos dados e das informações, aplicar processos de otimização estocásticos e realizar processos de inferência.

#### Distribuições e amostragem (10min)

- **Distribuições de Probabilidade (com foco na Normal)**: A Distribuição Normal (Gaussiana) é uma das distribuições de probabilidade mais importantes, caracterizada por sua forma simétrica, em sino, onde os valores próximos à média são os mais comuns. É uma distribuição fundamental na estatística e no machine learning, frequentemente utilizada para modelar ruído, incerteza e equilíbrio em diversos sistemas. Seus parâmetros principais são a média (μ), que define o centro da curva, e o desvio padrão (σ), que determina a dispersão ou largura da curva.

A função densidade de probabilidade (PDF) da distribuição normal é dada por:

$$f(x|\mu,\sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Onde:
- x é a variável aleatória.
- μ é a média (valor esperado).
- σ² é a variância.
- σ é o desvio padrão.

As aplicações da Distribuição Normal em Machine Learning são vastas:
- **Inicialização de Pesos**: Na inicialização de pesos em redes neurais profundas, distribuições normais (ou Gaussianas) são amplamente utilizadas. Essa prática ajuda a manter as ativações e os gradientes dentro de uma faixa razoável, promovendo um treinamento mais estável e eficiente da rede.
- **Modelos Probabilísticos**: Em modelos avançados como os Variational Autoencoders (VAEs), o espaço latente é modelado usando uma distribuição normal, permitindo amostragem suave e diferenciável e a geração de novas amostras.
- **Detecção de Anomalias**: Em sistemas de alto volume, o comportamento normal frequentemente segue uma curva de sino. Os outliers, ou anomalias, residem nas "caudas" da distribuição, tornando a distribuição normal uma ferramenta ideal para detecção de fraudes, intrusões em redes ou defeitos de fabricação.

- **Amostragem Aleatória e sua Importância**: A amostragem aleatória é crucial em Machine Learning, especialmente para a criação de mini-batches e técnicas de regularização como Dropout.
  - **Mini-batches**: Em vez de alimentar todos os dados de treinamento de uma vez (abordagem de batch completo) ou um item por vez (abordagem de gradiente estocástico puro), os dados são processados em pequenos subconjuntos chamados "mini-batches". Esta abordagem torna o treinamento mais estável e rápido, e é crucial para o Gradiente Descendente Estocástico (SGD), uma variante amplamente utilizada do Gradiente Descendente. O uso de mini-batches introduz uma estocasticidade no processo de treinamento, o que ajuda a evitar mínimos locais e acelera a convergência.
  - **Dropout**: É uma técnica de regularização poderosa que consiste em desativar ("desligar" ou "dropar") aleatoriamente um subconjunto de neurônios durante o treinamento de uma rede neural.
    - **Propósito**: O principal objetivo do dropout é prevenir o overfitting (sobreajuste). Ao forçar a rede a aprender representações mais robustas e redundantes, o dropout torna o modelo menos sensível aos pesos específicos de neurônios individuais. Isso é análogo a treinar várias sub-redes diferentes em cada iteração.
    - **Visão Probabilística**: O dropout pode ser interpretado como uma forma de otimização estocástica, onde cada iteração treina uma sub-rede diferente. Durante a inferência (teste), quando o dropout é desativado, o comportamento é similar à média das previsões de múltiplos modelos (ensemble learning). Mais profundamente, o dropout pode ser visto como uma aproximação Bayesiana de um Processo Gaussiano, permitindo a estimativa da incerteza do modelo.
    - **Taxa de Dropout**: É um hiperparâmetro que define a probabilidade de um neurônio ser desativado durante o treinamento, tipicamente variando entre 0.2 e 0.5.
  - **Generative Adversarial Networks (GANs)**: As GANs são um exemplo notável de como as distribuições de probabilidade são usadas para gerar novos dados. Uma GAN consiste em duas redes neurais que competem entre si: um Gerador e um Discriminador. O Gerador aprende a criar dados (por exemplo, imagens) que se assemelham aos dados reais, enquanto o Discriminador aprende a distinguir entre dados reais e dados gerados. O Gerador tenta "enganar" o Discriminador, e o Discriminador tenta "pegar" o Gerador. Esse processo de jogo de soma zero leva o Gerador a aprender a mapear um vetor de ruído aleatório (frequentemente amostrado de uma distribuição normal ou uniforme) para uma distribuição de dados complexa, como a de imagens reais. As GANs são amplamente utilizadas para geração de imagens sintéticas, aumento de dados e até mesmo para super-resolução em sensoriamento remoto.

#### Média, variância e sua importância (10min)

- **Média**: A média é uma medida de tendência central que representa o valor típico de um conjunto de dados.

- **Variância**: A variância é uma medida de dispersão que quantifica o quão espalhados os dados estão em relação à média. O desvio padrão (σ) é a raiz quadrada da variância.

A conexão desses conceitos com a **normalização de dados** é crucial para o treinamento de redes neurais. A normalização de dados, como a padronização (transformar dados para ter média zero e variância unitária), é uma etapa de pré-processamento comum.

- **Por que Normalizar?** Dados com média zero e variância unitária facilitam o treinamento de redes neurais por várias razões:
  - **Estabilidade Numérica**: Evita problemas de overflow (valores muito grandes) ou underflow (valores muito pequenos) que podem ocorrer durante o cálculo de gradientes em redes profundas.
  - **Convergência Mais Rápida**: Ajuda os algoritmos de otimização, como o Gradiente Descendente, a convergir mais rapidamente, pois a superfície de perda se torna mais "bem comportada".
  - **Tratamento de Escalas Diferentes**: Garante que nenhuma característica domine o processo de aprendizado apenas por ter uma escala de valores maior.

A **estabilidade numérica** é um conceito fundamental em Deep Learning. Redes neurais profundas envolvem muitas operações matemáticas, e se os valores de entrada ou os pesos não forem bem comportados (por exemplo, muito grandes ou muito pequenos), isso pode levar a gradientes desvanecentes ou explosivos, dificultando ou impossibilitando o treinamento. Estatísticas bem comportadas, como média zero e variância unitária, ajudam a mitigar esses problemas.

#### Prática no Colab: Visualizações estatísticas (10min)

Nesta seção, os alunos realizarão exercícios práticos no Google Colab para visualizar distribuições de probabilidade e o impacto da normalização de dados.

**Exercícios Sugeridos:**

1. **Criação e Visualização de Distribuições:**
   - Gere um conjunto de dados aleatórios que sigam uma distribuição normal (Gaussiana) usando np.random.normal().
   - Crie um histograma para visualizar a distribuição dos dados usando matplotlib.pyplot.hist().
   - Experimente gerar dados com diferentes médias e desvios padrão e observe as mudanças nos histogramas.

2. **Normalização de Dados e Impacto Visual:**
   - Crie um conjunto de dados de exemplo com uma média e variância arbitrárias.
   - Aplique a transformação de padronização (subtrair a média e dividir pelo desvio padrão) para normalizar os dados.
   - Crie histogramas dos dados antes e depois da normalização para observar o impacto visual na distribuição (centralização em zero e escala unitária).

3. **Cálculo de Estatísticas Descritivas:**
   - Carregue um pequeno conjunto de dados de exemplo (pode ser um array NumPy simulando pixels de uma imagem ou características de um dataset).
   - Calcule a média, variância e desvio padrão para cada "canal" ou "característica" dos dados.
   - Discuta como essas estatísticas podem informar a necessidade de normalização.

Essas práticas concretas ajudarão os alunos a construir uma intuição sobre a importância das distribuições de probabilidade e da normalização de dados para o treinamento eficaz de modelos de Deep Learning.

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