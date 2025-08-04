# Plano de Curso: Deep Learning Aplicado ao Sensoriamento Remoto
Visão Geral

Professor: Cap Philipe Borba 


Carga Horária Total: 60 horas 




Modalidade: Híbrido, com 20 horas EAD e 40 horas presenciais 



## 1. Módulos e Conteúdo Programático
### Semana 1: Modalidade EAD (20 horas) 




#### Módulo 1: Introdução e Fundamentos (5h) 


- Evolução da IA: Das origens ao Deep Learning, o Perceptron de Rosenblatt, o "Inverno da IA" e o renascimento com o backpropagation. A revolução da AlexNet em 2012 e a era dos Transformers a partir de 2017.
- Marcos em Visão Computacional: O dataset ImageNet e a evolução de arquiteturas como LeNet, AlexNet, VGG, ResNet e EfficientNet.
- Setup do Ambiente: Configuração do Google Colab e instalação de bibliotecas como PyTorch e NumPy.
- Revisão Matemática: Álgebra linear, cálculo (derivadas e gradientes) e noções de probabilidade e estatística aplicadas a Deep Learning, com práticas usando NumPy.

#### Módulo 2: Processamento de Imagens e Fundamentos (4h30min) 


- Fundamentos de Imagens: Representação de imagens como matrizes NumPy e operações básicas.
- Processamento Clássico: Filtros espaciais (média, Gaussiano), convolução 2D "from scratch" e técnicas de segmentação clássica (thresholding, k-means).
- Redes Neurais e PyTorch: O modelo matemático do neurônio, funções de ativação e os conceitos de forward e backward propagation. Introdução ao PyTorch (tensors, autograd, nn.Module) e implementação de uma MLP completa.

#### Módulo 3: CNNs e Técnicas de Regularização (6h30min) 


- Regularização: Diagnóstico de overfitting e underfitting. Teoria e prática de técnicas como Dropout, Weight Decay, Early Stopping e Batch Normalization.
- CNNs: Detalhes da operação de convolução (filtros, padding, stride), pooling e a evolução das arquiteturas convolucionais (LeNet, AlexNet, VGG, ResNet).
- Preparação de Dados: Criação de máscaras a partir de dados vetoriais, organização de datasets e estratégias teóricas para lidar com o balanceamento de classes.

#### Módulo 4: Aplicações em Sensoriamento Remoto (4h) 

- Processamento de Dados de SR: Características específicas de dados de sensoriamento remoto, como múltiplas bandas e o cálculo de índices espectrais (NDVI, NDWI). Pré-processamento específico para Deep Learning.
- Ferramentas Geoespaciais: Uso de bibliotecas como Rasterio e GeoPandas para leitura e manipulação de dados geoespaciais. Visualização e validação de dados no QGIS.
- Segmentação Semântica: Conceitos e métricas (IoU, Dice), além de uma introdução às principais arquiteturas de segmentação, como FCN e U-Net.


### Semana 2: Modalidade Presencial (40 horas) 

#### Dia 1: Implementação de Redes Neurais e CNNs (8h) 

- Manhã: Implementação prática de uma MLP completa com todas as técnicas de regularização, incluindo Batch Normalization. Análise comparativa e debugging com TensorBoard.
- Tarde: Implementação da arquitetura LeNet com o dataset MNIST. Desenvolvimento de uma CNN completa para classificação de cenas com o dataset WHU-RS19.

#### Dia 2: Arquiteturas CNN Avançadas (8h) 

- Manhã: Implementação da VGG-16 e aplicação no dataset WHU-RS19. Introdução e prática de Transfer Learning com modelos pré-treinados, adaptando-os para sensoriamento remoto.
- Tarde: Implementação da ResNet, focando nas skip connections e blocos residuais. Comparativo de performance e trade-offs entre LeNet, VGG e ResNet.

#### Dia 3: Preparação Profissional de Dados (8h) 

- Manhã: Criação de máscaras a partir de dados vetoriais reais usando GeoPandas e rasterização multiclasse. Implementação de um pipeline de tiling para imagens de alta resolução.
- Tarde: Implementação de um Dataset customizado no PyTorch para dados de SR. Abordagem prática para balanceamento de classes com weighted random sampling e class weights.

#### Dia 4: Segmentação Semântica (8h) 

- Manhã: Implementação da arquitetura U-Net "from scratch" para o dataset ISPRS Potsdam. Implementação de diferentes funções de perda para segmentação (Dice loss, Focal loss) e um training loop otimizado.
- Tarde: Uso da biblioteca Albumentations para Data Augmentation específica para sensoriamento remoto (ex: ruído de sensor e simulação de nuvens). Otimização do treinamento com mixed precision e estratégias de economia de memória.

#### Dia 5: Modelos Avançados e Projeto Final (8h) 

- Manhã: Exploração de bibliotecas como Segmentation Models PyTorch e uso de encoders pré-treinados. Estudo aprofundado da arquitetura DeepLabV3+, focando em atrous convolution e ASPP.
- Tarde: Período dedicado a práticas, refinamento de projetos e tirada de dúvidas.

# 2. Materiais e Recursos

Notebooks Jupyter/Colab: Conjunto de notebooks estruturados para cada módulo.


Datasets: MNIST, WHU-RS19, ISPRS Potsdam e um Dataset Brasileiro.


Software e Bibliotecas: PyTorch, NumPy, Rasterio, GeoPandas, QGIS (instalado localmente), TorchGeo, Albumentations, Segmentation Models PyTorch e TensorBoard.
