# Plano de Curso: Deep Learning Aplicado ao Sensoriamento Remoto

## Visão Geral

**Professor:** Cap Philipe Borba  
**Carga Horária Total:** 60 horas  
**Modalidade:** Híbrido, com 20 horas EAD e 40 horas presenciais

## 1. Módulos e Conteúdo Programático

### Semana 1: Modalidade EAD (20 horas)

#### Módulo 1: Fundamentos e Contexto (4h30min)

**1.1 Introdução ao Curso e Evolução da IA (1h10min)**
- **Contextualização e Evolução da IA:** Das origens ao Deep Learning, o Perceptron de Rosenblatt (1957), o "Inverno da IA" nas décadas de 1970-80, renascimento com backpropagation, revolução da AlexNet (2012) e era dos Transformers (2017-presente)
- **Marcos Históricos em Visão Computacional:** O dataset ImageNet e evolução arquitetural: LeNet → AlexNet → VGG → ResNet → EfficientNet
- **Deep Learning no Sensoriamento Remoto:** Transição de métodos pixel-based para CNNs, peculiaridades e tendências atuais

**1.2 Setup do Ambiente (20min)**
- Configuração do Google Colab com ativação de GPU
- Instalação automatizada de bibliotecas e template de notebooks

**1.3 Revisão Matemática com NumPy (3h)**
- **Álgebra Linear Aplicada:** Vetores e matrizes como representação de dados, operações matriciais essenciais, broadcasting e vetorização
- **Cálculo para Deep Learning:** Derivadas e gradientes com intuição visual, regra da cadeia, otimização e gradiente descendente
- **Probabilidade e Estatística:** Distribuições, amostragem, média e variância aplicadas ao contexto de Deep Learning
- **Práticas no Colab:** Implementação hands-on de operações básicas e visualizações

#### Módulo 2: Processamento de Imagens Clássico (2h)

**2.1 Fundamentos de Processamento de Imagens (2h)**
- **Conceitos Básicos:** Imagens como matrizes NumPy (grayscale, RGB, multiespectral), tipos de dados e conversões, operações básicas
- **Filtros e Convolução:** Filtros espaciais clássicos (média, Gaussiano, Sobel, Laplaciano), implementação de convolução 2D from scratch, conceitos de sliding window e padding
- **Segmentação Clássica:** Abordagens pixel-based vs region-based, técnicas tradicionais (thresholding, k-means, watershed)

#### Módulo 3: Sensoriamento Remoto e Normalização (2h30min)

**3.1 Características dos Dados de Sensoriamento Remoto (1h)**
- **Múltiplas Bandas e Resoluções:** Bandas espectrais e interpretações físicas, diferentes resoluções (espacial, espectral, temporal)
- **Índices Espectrais:** NDVI, NDWI e suas aplicações práticas

**3.2 Normalização: Ponte para Deep Learning (1h30min)**
- **Problemas Específicos:** Diferentes escalas de bandas espectrais e impacto na convergência
- **Técnicas de Normalização:** Normalização por canal vs global, cálculo incremental para datasets grandes
- **Implementação Prática:** Comparação visual e validação com dados simulados

#### Módulo 4: Redes Neurais - Teoria e Implementação (8h30min)

**4.1 Fundamentos de Redes Neurais (1h30min)**
- **O Perceptron:** Modelo matemático, implementação prática (AND/OR), limitações fundamentais (problema XOR)
- **Funções de Ativação:** ReLU, Sigmoid, Tanh - propriedades matemáticas e importância das não-linearidades
- **Arquitetura de MLPs:** Estrutura básica, representação matricial e notação

**4.2 Teoria do Treinamento (2h45min)**
- **Forward Propagation:** Matemática detalhada das transformações lineares e não-lineares, implementação manual
- **Funções de Perda:** MSE para regressão, Cross-entropy para classificação, intuição geométrica
- **Hiperparâmetros Fundamentais:** Conceitos de batch/batch size, epochs, learning rate como controlador crítico
- **Backpropagation:** Regra da cadeia aplicada, cálculo de gradientes, computational graphs, demonstração passo a passo

**4.3 Implementação com PyTorch (2h30min)**
- **PyTorch Fundamentals:** Tensors vs NumPy, criação e manipulação, conversão de operações
- **Autograd:** requires_grad, método .backward(), acúmulo de gradientes, torch.no_grad()
- **Estrutura nn.Module:** Construção de MLPs, funções de perda, otimizadores (SGD vs Adam)

**4.4 Training Loop Completo (1h45min)**
- **Anatomia do Training Loop:** Forward, loss, backward, step integrados com hiperparâmetros
- **Dataset e DataLoader:** MNIST como caso prático, modo train vs eval
- **Monitoramento:** Interpretação de loss curves, TensorBoard básico, debugging comum

#### Módulo 5: Introdução às CNNs (2h30min)

**5.1 Da Convolução Clássica às CNNs (1h15min)**
- **Limitações de MLPs:** Perda de informação espacial, explosão de parâmetros
- **Operação de Convolução Aprendida:** Conexão explícita com filtros clássicos do Módulo 2, conceitos de padding, stride, receptive field
- **Arquitetura CNN Básica:** Camadas convolucionais, pooling, fully connected, hierarquia de características

**5.2 LeNet no MNIST (1h15min)**
- **Implementação LeNet-5:** Arquitetura detalhada, implementação guiada, adaptação do training loop
- **Análise Comparativa:** MLP vs CNN, visualização de feature maps e filtros aprendidos
- **Preparação Conceitual:** Por que CNNs são ideais para sensoriamento remoto

### Semana 2: Modalidade Presencial (40 horas)

#### Dia 1: Consolidação, Regularização e Ferramentas (8h)

**Manhã - Síntese e Técnicas de Regularização (4h)**
- **Momento de Síntese:** Conexão de conceitos da primeira semana, Q&A intensiva
- **Overfitting e Diagnóstico:** Demonstração prática, curvas de aprendizado
- **Técnicas de Regularização:** Dropout, Weight Decay, Batch Normalization, Early Stopping com implementações práticas

**Tarde - Ferramentas Geoespaciais (4h)**
- **QGIS Hands-on:** Visualização de dados vetoriais e raster, alinhamento, styling
- **Rasterio e GeoPandas:** Pipeline programático para dados geoespaciais, conversões de formato

#### Dia 2: CNNs Avançadas e Aplicações em SR (8h)

**Manhã - Arquiteturas CNN Clássicas (4h)**
- **VGG Implementation:** Filosofia de redes profundas, blocos convolucionais modulares
- **ResNet e Skip Connections:** Solução para gradientes evanescentes, blocos residuais
- **Transfer Learning:** Modelos pré-treinados, adaptação para dados multiespectrais

**Tarde - Classificação de Cenas WHU-RS19 (4h)**
- **Dataset WHU-RS19:** 19 categorias de cenas, preprocessamento específico para SR
- **CNN Especializada:** Adaptações arquiteturais, pipeline completo de treinamento
- **Análise Comparativa:** Diferentes arquiteturas, matrizes de confusão, visualização de características

#### Dia 3: Preparação Profissional de Dados (8h)

**Manhã - Pipeline Vetor → Raster (4h)**
- **Dataset ISPRS Potsdam:** Segmentação urbana, 6 classes de cobertura do solo
- **Criação de Máscaras:** GeoPandas, rasterização multiclasse, validação QGIS
- **Estratégias de Tiling:** Sliding window, overlap, filtragem de tiles informativos

**Tarde - Dataset PyTorch e Balanceamento (4h)**
- **Dataset Customizado:** Lazy loading, caching, pipeline flexível para SR
- **Balanceamento de Classes:** Weighted random sampling, class weights, análise estatística
- **Focal Loss:** Solução para classes raras, implementação e comparação

#### Dia 4: Segmentação Semântica (8h)

**Manhã - U-Net Implementation (4h)**
- **Arquitetura U-Net:** Encoder-decoder, skip connections, implementação completa
- **Loss Functions:** Cross-entropy weighted, Dice loss, losses combinadas
- **Training Pipeline:** IoU tracking, validação especializada, TensorBoard

**Tarde - Data Augmentation e Otimização (4h)**
- **Albumentations:** Pipeline para SR, preservação mask-imagem, augmentations específicas
- **Técnicas Avançadas:** Mixed precision training, gradient accumulation, learning rate scheduling

#### Dia 5: Estado da Arte e Projeto Final (8h)

**Manhã - Arquiteturas Especializadas (4h)**
- **Segmentation Models PyTorch:** Encoders pré-treinados, comparação sistemática
- **DeepLab v3+:** Atrous convolution, ASPP, refinement decoder

**Tarde - Projeto Integrador (4h)**
- **Dataset Brasileiro:** Aplicação end-to-end, workflow completo
- **Apresentação e Síntese:** Resultados, lições aprendidas, próximos passos

## 2. Materiais e Recursos

### Notebooks Jupyter/Colab Estruturados
1. **00_Setup_Environment.ipynb** - Configuração completa
2. **01_Math_Foundations.ipynb** - Revisão matemática
3. **02_Image_Processing.ipynb** - Processamento com NumPy
4. **03_Neural_Networks_PyTorch.ipynb** - MLPs e training loops
5. **04_Remote_Sensing_Normalization.ipynb** - Dados SR e normalização
6. **05_CNN_Fundamentals.ipynb** - LeNet e conceitos básicos
7. **06_Advanced_CNNs.ipynb** - VGG, ResNet e transfer learning
8. **07_Geospatial_Tools.ipynb** - GeoPandas, Rasterio, QGIS
9. **08_Dataset_Creation.ipynb** - Pipeline completo de dados
10. **09_Segmentation_UNet.ipynb** - U-Net e segmentação semântica
11. **10_Complete_Pipeline.ipynb** - Projeto integrador

### Datasets Utilizados
- **MNIST** - Aquecimento com LeNet
- **WHU-RS19** - Classificação de cenas (19 classes)
- **ISPRS Potsdam** - Segmentação urbana (6 classes)
- **Dataset Brasileiro** - Projeto final

### Software e Bibliotecas
- **PyTorch** >= 2.0
- **NumPy** >= 1.24
- **Rasterio** >= 1.3
- **GeoPandas** >= 0.13
- **QGIS** >= 3.40 (instalado localmente)
- **Albumentations** >= 1.3
- **Segmentation Models PyTorch** >= 0.3
- **TensorBoard** >= 2.12

### Metodologia de Ensino
- **Abordagem Hands-on:** Teoria integrada com prática
- **Progressão Scaffolded:** Conceitos básicos → aplicações avançadas
- **Projetos Reais:** Dados brasileiros de sensoriamento remoto
- **Debugging Integrado:** Problemas comuns e soluções práticas
- **Visualização Contínua:** TensorBoard e QGIS para validação

### Competências Desenvolvidas
- Processamento de dados geoespaciais para Deep Learning
- Construção de pipelines completos de treinamento
- Implementação de arquiteturas CNN especializadas
- Avaliação com métricas adequadas para sensoriamento remoto
- Preparação de dados reais para produção
- Debugging e otimização de modelos
