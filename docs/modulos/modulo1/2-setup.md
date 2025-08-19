---
prev_page: "/modulos/modulo1/1-intro/"
next_page: "/modulos/modulo1/3-revisao_matematica/"
---

## Setup do Ambiente (20min)

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
