# Plano de Curso: Deep Learning Aplicado ao Sensoriamento Remoto

## Visão Geral

**Professor:** Cap Philipe Borba  
**Carga Horária Total:** 56 horas  
**Modalidade:** Híbrido, com 16 horas EAD e 40 horas presenciais

## Semana 1: Modalidade EAD (16 horas)

### Módulo 1: Fundamentos e Contexto (4h)

**1.1 Introdução ao Curso e Evolução da IA no Sensoriamento Remoto (40 min)**
- **Apresentação do curso e da metodologia:** interação com os alunos para colher as expectativas e apresentar o conteúdo programático.
- **Contextualização e Evolução da IA** 
- **Aplicações de DL ao Sensoriamento Remoto:** detecção de objetos, detecção de mudanças, superresolução, transferência de estilo, classificação de cena;

**1.2 Setup do Ambiente (10min)**
- Configuração do Google Colab com ativação de GPU
- Instalação automatizada de bibliotecas e template de notebooks

**Intervalo (10 min)**

**1.3 Revisão Matemática com NumPy (50 min)**
- **Revisão de numpy, matplotlib e opencv**
- **Práticas no Colab:** Implementação hands-on de operações básicas e visualizações

**Intervalo (10 min)**

**1.4 Fundamentos de Processamento de Imagens (2h)**
- **Conceitos Básicos:** Imagens como matrizes NumPy (grayscale, RGB, multiespectral), tipos de dados e conversões, operações básicas
- **Filtros e Convolução:** Filtros espaciais clássicos (média, Gaussiano, Sobel, Laplaciano), implementação de convolução 2D, conceitos de sliding window e padding

### Módulo 2: Redes Neurais - Teoria e Primeiras Aplicações Práticas (4h)

**2.1 Fundamentos de Redes Neurais (2h)**
- **PyTorch Fundamentals:** Tensors vs NumPy, criação e manipulação, conversão de operações
- **Revisão de cálculo:** derivadas, gradiente, regra da cadeia
- **Gradiente descendente:** mostrar como se faz a minimização de uma função y = f(x) utilizando o gradiente descendente (exemplo J(x) = x²+2x+1)
- **O Perceptron:** Modelo matemático, implementação prática no pytorch, limitações fundamentais (problema XOR), regressão linear, Teoria de Aproximação Universal
- **Funções de Ativação:** ReLU, Sigmoid, Tanh - propriedades matemáticas e importância das não-linearidades
- **Arquitetura de MLPs:** Estrutura básica, representação matricial e notação
- **O processo de treinamento:** Explicar em termos gerais como uma rede neural aprende.
- **Estrutura nn.Module:** Construção de MLPs
- **Dataloader e Dataset:** MNIST como caso prático
- **Exemplo prático:** executar no colab o treinamento de uma rede já implementada

**2.2 Teoria do Treinamento e implementação com PyTorch (2h)**
- **Forward Propagation:** Matemática detalhada das transformações lineares e não-lineares, implementação manual
- **Funções de Perda:** MSE para regressão, Cross-entropy para classificação, intuição geométrica
- **Hiperparâmetros Fundamentais:** Conceitos de batch/batch size, epochs, learning rate como controlador crítico
- **Cálculo para Deep Learning:** Derivadas e gradientes com intuição visual, regra da cadeia, otimização e gradiente descendente
- **Backpropagation:** Regra da cadeia aplicada, cálculo de gradientes, computational graphs, demonstração passo a passo
- **Autograd:** requires_grad, método .backward(), acúmulo de gradientes, torch.no_grad()
- **Otimizadores:** o que são e como usar

### Módulo 3: Treinamento de Redes Neurais (4h)

**3.1 Training Loop Completo e Preparação de Dados (1h15min)**
- **Anatomia do Training Loop:** Estrutura fundamental (forward, loss, backward, step)
- **Dataset e DataLoader para MNIST:** Carregamento eficiente com torchvision
- **Modo train vs eval:** Quando e por que alternar entre modos
- **Dataset e DataLoader Avançado:** 
  - Otimização de performance (num_workers, prefetch factor, pin memory)
  - Lidando com desbalanceamento de classes
  - WeightedRandomSampler para balanceamento de batches
- **Implementação Guiada:** Primeiro training loop funcional completo

**Intervalo (10 min)**

**3.2 Funções de Perda e Otimização (1h15min)**
- **Funções de Perda Fundamentais:**
  - MSE e MAE para regressão
  - Binary Cross-Entropy e Categorical Cross-Entropy
  - Importância do formato de entrada (logits vs probabilidades)
- **Funções de Perda Avançadas:**
  - Dice Loss e IoU Loss para segmentação
  - Focal Loss para desequilíbrio de classes
- **Learning Rate Schedulers:**
  - StepLR, MultiStepLR, ExponentialLR
  - ReduceLROnPlateau e OneCycleLR
  - Ordem crítica: optimizer.step() vs scheduler.step()

**Intervalo (10 min)**

**3.3 Avaliação, Diagnóstico e Regularização (1h10min)**
- **Métricas de Avaliação Essenciais:**
  - Classificação: Accuracy, Precision, Recall, F1-Score, ROC AUC
  - Regressão: RMSE, MAE
  - Detecção/Segmentação: IoU (Intersection over Union)
- **Underfitting e Overfitting:**
  - Identificação através de curvas de perda (learning curves)
  - Early Stopping: implementação e parametrização
- **Técnicas de Regularização:**
  - Dropout: prevenção de overfitting
  - Weight Decay (L2 regularization)
  - Batch Normalization: estabilização do treinamento
  - Gradient Clipping e Gradient Accumulation

**3.4 Checkpointing, Monitoramento e Integração (1h10min)**
- **Checkpointing:**
  - Salvar e restaurar state_dict do modelo e otimizador
  - Restauração para retomar treinamento vs inferência
  - Inferência eficiente: model.eval() e torch.no_grad()
- **Monitoramento com TensorBoard:**
  - Configuração básica e visualizações
  - Matriz de confusão e visualização de métricas
  - Interpretação de loss curves e debugging comum
- **Exercício Prático Integrado:**
  - Sistema completo integrando todas as técnicas
  - Análise comparativa: com e sem balanceamento de classes
  - Visualizações avançadas e insights

### Módulo 4: Introdução às CNNs (4h)

**4.1 Da Convolução Clássica às CNNs (2h)**
- **Limitações de MLPs:** Perda de informação espacial, explosão de parâmetros, maldição da dimensionalidade (curse of dimensionality)
- **Operação de Convolução Aprendida:** Conexão explícita com filtros clássicos do Módulo 2, conceitos de padding, stride, receptive field
- **Arquitetura CNN Básica:** Camadas convolucionais, pooling, fully connected, hierarquia de características

**4.2 LeNet no MNIST (2h)**
- **Implementação LeNet-5:** Arquitetura detalhada, implementação guiada, adaptação do training loop
- **Análise Comparativa:** MLP vs CNN, visualização de feature maps e filtros aprendidos
- **Preparação Conceitual:** Por que CNNs são ideais para sensoriamento remoto

## Semana 2: Modalidade Presencial (40 horas)

### Dia 1: Consolidação, Regularização e Ferramentas (8h)

**Manhã - Síntese e Técnicas de Regularização (4h)**
- **Momento de Síntese:** Conexão de conceitos da primeira semana, revisão

**Tarde - Ferramentas Geoespaciais (4h)**
- **QGIS Hands-on:** Visualização de dados vetoriais e raster, alinhamento, styling
- **Rasterio e GeoPandas:** Pipeline programático para dados geoespaciais, conversões de formato

### Dia 2: CNNs Avançadas e Aplicações em SR (8h)

**Manhã - Arquiteturas CNN Clássicas (4h)**
- **VGG Implementation:** Filosofia de redes profundas, blocos convolucionais modulares
- **ResNet e Skip Connections:** Solução para vanishing gradients, blocos residuais
- **Transfer Learning:** Modelos pré-treinados, adaptação para dados multiespectrais

**Tarde - Classificação de Cenas RESIC-45 (4h)**
- **Dataset RESIC-45:** 45 categorias de cenas, preprocessamento específico para SR
- **CNN Especializada:** Adaptações arquiteturais, pipeline completo de treinamento
- **Análise Comparativa:** Diferentes arquiteturas, matrizes de confusão, visualização de características

### Dia 3: Segmentação Semântica (8h)

**Manhã - U-Net Implementation (4h)**
- **Arquitetura U-Net:** Encoder-decoder, skip connections, implementação completa
- **Loss Functions:** Cross-entropy weighted, Dice loss, losses combinadas
- **Training Pipeline:** IoU tracking, validação especializada, TensorBoard
- **Treinamento usando o Dataset ISPRS Potsdam:** Segmentação urbana, 6 classes de cobertura do solo

**Tarde - Data Augmentation e Otimização (4h)**
- **Albumentations:** Pipeline para SR, preservação mask-imagem, augmentations específicas
- **PyTorch Lightning:** Conversão de códigos para o PyTorch Lightning, vantagens de uso (Treinamento distribuído)
- **Técnicas Avançadas:** Mixed precision training, gradient accumulation, learning rate scheduling

### Dia 4: Preparação Profissional de Dados (8h)

**Manhã - Pipeline Vetor → Raster (4h)**
- **Dataset Custom:** Passo a passo de criação de um dataset utilizando dados da DSG
- **Criação de Máscaras:** GeoPandas, rasterização multiclasse, validação QGIS
- **Estratégias de Tiling:** Sliding window, overlap, filtragem de tiles informativos

**Tarde - Dataset PyTorch e Balanceamento (4h)**
- **Dataset Customizado:** Lazy loading, caching, pipeline flexível para SR
- **Balanceamento de Classes:** Weighted random sampling, class weights, análise estatística
- **Focal Loss:** Solução para classes raras, implementação e comparação
- **Treinamento:**  Utilizar um trecho do dataset de edificações da DSG para treinar

### Dia 5: Estado da Arte e Projeto Final (8h)

**Manhã - Arquiteturas Especializadas (4h)**
- **Segmentation Models PyTorch:** Encoders pré-treinados, comparação sistemática
- **DeepLab v3+:** Atrous convolution, ASPP, refinement decoder
- **Treinamento utilizando o SMP e dados da DSG:** utilizar os dados do dataset construído e as implementações anteriores para realizar o treinamento

**Tarde - Projeto Integrador (4h)**
- **Continuação do treinamento anterior:** Aplicação end-to-end, workflow completo
- **Apresentação e Síntese:** Resultados, lições aprendidas e conclusão do curso

## Datasets Utilizados
- **MNIST** - Aquecimento com LeNet
- **WHU-RS19** - Classificação de cenas (19 classes)
- **ISPRS Potsdam** - Segmentação urbana (6 classes)
- **Dataset Custom** - Projeto final

## Software e Bibliotecas
- **PyTorch** >= 2.0
- **NumPy** >= 1.24
- **Rasterio** >= 1.3
- **GeoPandas** >= 0.13
- **QGIS** >= 3.40 (instalado localmente)
- **Albumentations** >= 1.3
- **Segmentation Models PyTorch** >= 0.3
- **TensorBoard** >= 2.12

## Competências Desenvolvidas
- Processamento de dados geoespaciais para Deep Learning
- Construção de pipelines completos de treinamento
- Implementação de arquiteturas CNN especializadas
- Avaliação com métricas adequadas para sensoriamento remoto
- Preparação de dados reais para produção
