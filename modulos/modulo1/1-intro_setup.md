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
# Exemplo de instação de biblioteca para o curso
!pip install rasterio
```

Este script instala bibliotecas comuns para manipulação de dados (`numpy`, `pandas`), visualização (`matplotlib`), machine learning (`scikit-learn`), deep learning (`torch`), processamento de imagens (`opencv-python`, `scikit-image`), e dados geoespaciais (`rasterio`, `geopandas`).

2. **Importância de Verificar Versões**: A reprodutibilidade é crucial em projetos de Deep Learning. Diferentes versões de bibliotecas podem levar a comportamentos inesperados ou erros. É importante verificar as versões das bibliotecas instaladas:

```python
import numpy as np
import pandas as pd
import torch
import cv2
import rasterio
import geopandas as gpd

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"PyTorch version: {torch.__version__}")
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
import torch

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

