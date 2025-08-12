# Estrutura Refinada com Detalhamento Conceitual Completo

## PRIMEIRA SEMANA - MODALIDADE EAD (20 horas)

### Módulo 1: Fundamentos e Contexto (4h30min)

#### **1.1 Introdução ao Curso e Evolução da IA (1h10min)**

##### **Contextualização e Evolução da IA** (50min)

###### **Das Origens ao Deep Learning** (25min)
**Conceitos a abordar:** Apresentar a linha temporal desde o Perceptron de Rosenblatt (1957) como primeira tentativa de simular neurônios artificiais. Explicar o conceito de "Inverno da IA" nas décadas de 1970-80, quando as limitações dos perceptrons simples e a falta de poder computacional levaram ao ceticismo sobre redes neurais. Discutir o renascimento com o algoritmo de backpropagation nos anos 1980 e como isso permitiu treinar redes multicamadas. Culminar com a revolução de 2012 com AlexNet no ImageNet, estabelecendo o deep learning como paradigma dominante em visão computacional.

###### **Marcos Históricos em Visão Computacional** (15min)
**Conceitos a abordar:** Contextualizar o ImageNet como o dataset que mudou tudo, explicando por que ter milhões de imagens rotuladas foi crucial para o sucesso do deep learning. Apresentar a evolução arquitetural: LeNet como pioneira para reconhecimento de dígitos, AlexNet introduzindo ReLU e dropout, VGG demonstrando que profundidade importa, ResNet resolvendo gradientes evanescentes com skip connections, e EfficientNet otimizando eficiência computacional. Mencionar brevemente o estado atual com modelos como SAM (Segment Anything Model) e modelos multimodais.

###### **Deep Learning no Sensoriamento Remoto** (10min)
**Conceitos a abordar:** Explicar a transição histórica dos métodos tradicionais pixel-based (classificação baseada apenas em valores espectrais) para abordagens baseadas em CNNs que consideram contexto espacial. Destacar as peculiaridades que tornaram essa transição desafiadora: dados multiespectrais, diferentes resoluções, necessidade de grandes áreas de treinamento. Apresentar as tendências atuais: modelos foundation para sensoriamento remoto, técnicas de self-supervised learning, e a crescente importância de dados de alta resolução temporal.

##### **Visão do Curso e Objetivos** (20min)
**Conceitos a abordar:** Apresentar a estrutura didática hands-on do curso, enfatizando a progressão de conceitos básicos para aplicações avançadas. Definir claramente as competências que os alunos desenvolverão: processamento de dados geoespaciais, construção de pipelines de deep learning para sensoriamento remoto, avaliação de modelos com métricas adequadas, e preparação de dados reais para produção. Mostrar exemplos dos projetos práticos que implementarão, criando expectativa e motivação para o aprendizado.

#### **1.2 Setup do Ambiente (20min)**

##### **Google Colab - Setup Rápido** (10min)
**Conceitos a abordar:** Guiar os alunos através do acesso ao Google Colab e criação do primeiro notebook. Demonstrar como ativar e verificar a disponibilidade de GPU, explicando por que isso é crucial para deep learning. Mostrar como montar o Google Drive para persistência de dados entre sessões. Ensinar verificações básicas de hardware disponível e como monitorar uso de RAM e GPU durante experimentos.

##### **Instalação de Bibliotecas** (10min)
**Conceitos a abordar:** Apresentar um script automatizado para instalação de todas as bibliotecas necessárias ao curso. Explicar a importância de verificar versões de bibliotecas para reprodutibilidade. Criar um template de notebook com todos os imports essenciais organizados por categoria: manipulação de dados, deep learning, visualização, e geoespacial. Estabelecer boas práticas de organização de código desde o início.

#### **1.3 Revisão Matemática com NumPy (3h)**

##### **Álgebra Linear Aplicada** (1h20min)

###### **Vetores e matrizes como representação de dados** (20min)
**Conceitos a abordar:** Explicar como dados em machine learning são naturalmente representados como vetores (uma observação) e matrizes (múltiplas observações). Demonstrar que imagens são tensors 3D (altura × largura × canais) e como isso se estende para batches (batch × altura × largura × canais). Mostrar a conexão entre representação matemática e implementação computacional, estabelecendo que compreender álgebra linear é fundamental para compreender deep learning.

###### **Operações matriciais essenciais** (25min)
**Conceitos a abordar:** Cobrir multiplicação de matrizes como operação fundamental em redes neurais, explicando como cada camada realiza transformação linear através de multiplicação matriz-vetor. Demonstrar operações elemento-wise (produto de Hadamard) e quando são usadas. Explicar transposição e sua importância em backpropagation. Mostrar como operações como soma, subtração e escalonamento se comportam em tensors multidimensionais.

###### **Broadcasting e vetorização eficiente** (20min)
**Conceitos a abordar:** Explicar o conceito de broadcasting do NumPy, que permite operações entre arrays de diferentes dimensões seguindo regras específicas. Demonstrar como isso elimina a necessidade de loops explícitos, tornando o código mais eficiente e legível. Mostrar exemplos práticos: adicionar bias a uma matriz de ativações, normalizar dados por canal, aplicar transformações elemento-wise. Enfatizar que compreender broadcasting é essencial para implementações eficientes.

###### **Prática no Colab: Implementação de operações básicas** (15min)
**Conceitos a abordar:** Exercícios práticos para consolidar os conceitos: criar matrizes de diferentes dimensões, realizar multiplicações matriciais, aplicar broadcasting em cenários reais, medir diferenças de performance entre implementações vetorizadas e com loops. Criar exemplos que simulam operações de redes neurais (multiplicação de pesos, adição de bias, aplicação de funções de ativação).

##### **Cálculo para Deep Learning** (1h10min)

###### **Derivadas e gradientes - intuição visual** (20min)
**Conceitos a abordar:** Revisar o conceito de derivada como taxa de mudança instantânea, usando visualizações gráficas para construir intuição. Explicar gradientes como generalizações de derivadas para funções multivariáveis, representando a direção de maior crescimento. Demonstrar através de gráficos de contorno como gradientes apontam "morro acima" e como isso se conecta com otimização em machine learning, onde queremos ir "vale abaixo" para minimizar loss functions.

###### **Regra da cadeia - conceito fundamental** (25min)
**Conceitos a abordar:** Apresentar a regra da cadeia como ferramenta matemática para calcular derivadas de funções compostas. Usar exemplos simples inicialmente: f(g(x)) = f'(g(x)) × g'(x). Explicar por que isso é fundamental para redes neurais, onde temos múltiplas camadas de transformações compostas. Mostrar visualmente como a regra da cadeia "propaga" derivadas através de uma cadeia de operações, preparando conceitualmente para o backpropagation que virá no Módulo 4.

###### **Otimização e gradiente descendente** (15min)
**Conceitos a abordar:** Introduzir o problema fundamental de otimização em machine learning: encontrar parâmetros que minimizam uma função de loss. Explicar o algoritmo de gradiente descendente intuitivamente: começar em um ponto aleatório, calcular a direção de maior crescimento (gradiente), mover-se na direção oposta. Demonstrar através de visualizações como o learning rate controla o tamanho dos passos e por que escolhê-lo adequadamente é crucial.

###### **Prática no Colab: Visualização de gradiente descendente** (10min)
**Conceitos a abordar:** Implementar gradiente descendente para uma função simples (por exemplo, y = x² + 2x + 1), criando visualizações animadas mostrando como o algoritmo converge para o mínimo. Experimentar com diferentes learning rates para demonstrar convergência lenta, rápida, e instabilidade. Esta prática concreta consolidará a intuição matemática e preparará para compreender otimizadores em redes neurais.

##### **Probabilidade e Estatística** (30min)

###### **Distribuições e amostragem** (10min)
**Conceitos a abordar:** Revisar conceitos fundamentais de distribuições de probabilidade, focando especialmente na distribuição normal (Gaussiana) por sua relevância em inicialização de pesos e análise de dados. Explicar amostragem aleatória e sua importância em machine learning, especialmente para criação de minibatches e técnicas de regularização como dropout.

###### **Média, variância e sua importância** (10min)
**Conceitos a abordar:** Explicar média como medida de tendência central e variância como medida de dispersão. Conectar esses conceitos com normalização de dados, explicando por que dados com média zero e variância unitária facilitam o treinamento de redes neurais. Introduzir o conceito de estabilidade numérica e como estatísticas bem comportadas evitam problemas de overflow/underflow.

###### **Prática no Colab: Visualizações estatísticas** (10min)
**Conceitos a abordar:** Criar histogramas e gráficos para visualizar diferentes distribuições. Gerar dados com diferentes médias e variâncias, aplicar transformações de normalização, e observar o impacto visual. Calcular estatísticas descritivas de datasets reais (por exemplo, pixels de imagens) para construir intuição sobre a importância da normalização.

### Módulo 2: Processamento de Imagens Clássico (2h)

#### **2.1 Fundamentos de Processamento de Imagens (2h)**

##### **Conceitos Básicos e Representação** (40min)

###### **Imagem como matriz NumPy (grayscale, RGB, multiespectral)** (15min)
**Conceitos a abordar:** Explicar como imagens digitais são arrays multidimensionais de valores numéricos. Demonstrar imagens grayscale como matrizes 2D (altura × largura) onde cada valor representa intensidade de pixel. Mostrar imagens RGB como arrays 3D (altura × largura × 3 canais) e como cada canal representa uma cor primária. Introduzir o conceito de imagens multiespectrais como extensão natural (altura × largura × N canais), preparando para dados de sensoriamento remoto. Explicar convenções de indexação e diferenças entre sistemas de coordenadas.

###### **Tipos de dados e conversões** (10min)
**Conceitos a abordar:** Apresentar diferentes tipos de dados para pixels: uint8 (0-255) mais comum, float32/float64 para processamento matemático, e tipos específicos de sensoriamento remoto (uint16 para dados de 16 bits). Explicar quando e como converter entre tipos, demonstrando problemas de overflow/underflow. Mostrar como diferentes tipos afetam operações matemáticas e visualização.

###### **Operações básicas: slicing, reshape, normalização básica** (15min)
**Conceitos a abordar:** Demonstrar slicing para extrair regiões de interesse, canais específicos, ou subamostrar imagens. Explicar reshape para diferentes organizações de dados (por exemplo, de 2D para 1D para MLPs). Introduzir normalização básica min-max e padronização z-score, explicando quando usar cada uma. Mostrar como essas operações são building blocks para processamento mais avançado.

##### **Filtros e Convolução** (45min)

###### **Filtros espaciais clássicos (média, Gaussiano, Sobel, Laplaciano)** (20min)
**Conceitos a abordar:** Explicar cada filtro com sua interpretação física e matemática. Filtro de média como suavização simples que remove ruído mas borra detalhes. Filtro Gaussiano como suavização mais sofisticada que preserva melhor as bordas, explicando o parâmetro sigma. Operador Sobel como detector de bordas que calcula gradientes espaciais. Filtro Laplaciano como detector de mudanças rápidas de intensidade. Mostrar kernels 3×3 típicos para cada filtro e explicar como cada um transforma a informação da imagem.

###### **Implementação de convolução 2D from scratch** (15min)
**Conceitos a abordar:** Implementar operação de convolução passo a passo usando loops duplos e slicing do NumPy. Explicar como o kernel "desliza" sobre a imagem, calculando produtos elemento-wise e soma. Demonstrar como diferentes kernels produzem diferentes efeitos. Esta implementação manual é crucial para compreender o que acontece "por baixo dos panos" nas CNNs posteriormente.

###### **Sliding window e padding** (10min)
**Conceitos a abordar:** Explicar o conceito de janela deslizante como mecanismo fundamental da convolução. Demonstrar o problema de redução de dimensões nas bordas e como padding resolve isso. Mostrar diferentes tipos de padding (zeros, reflect, replicate) e quando usar cada um. Explicar como padding 'valid' vs 'same' afeta o tamanho da saída, preparando conceitos para CNNs.

##### **Segmentação Clássica** (35min)

###### **Conceitos: pixel-based vs region-based** (10min)
**Conceitos a abordar:** Distinguir abordagens que tomam decisões pixel por pixel (thresholding, classificação por cor) daquelas que agrupam pixels similares em regiões (region growing, watershed). Explicar vantagens e desvantagens: métodos pixel-based são simples mas podem resultar em fragmentação; métodos region-based produzem segmentos coerentes mas são mais complexos. Conectar com limitações que motivam deep learning para segmentação.

###### **Técnicas tradicionais (thresholding, k-means, watershed)** (15min)
**Conceitos a abordar:** Thresholding como técnica mais simples, separando objetos por intensidade, incluindo métodos adaptativos como Otsu. K-means como clustering de pixels no espaço de cores, útil para quantização e segmentação básica. Algoritmo Watershed como técnica mais sofisticada baseada em morfologia matemática, explicando a analogia com bacias hidrográficas e o problema de super-segmentação.

###### **Prática NumPy: Aplicação completa em imagens simples** (10min)
**Conceitos a abordar:** Implementar e aplicar cada técnica em imagens de exemplo, comparando resultados visualmente. Usar imagens sintéticas simples inicialmente, depois imagens reais. Demonstrar limitações de cada método e situações onde falham, criando motivação para técnicas mais avançadas de deep learning.

### Módulo 3: Sensoriamento Remoto e Normalização (2h30min)

#### **3.1 Características dos Dados de Sensoriamento Remoto (1h)**

##### **Múltiplas Bandas e Resoluções** (35min)

###### **Bandas espectrais e suas interpretações físicas** (15min)
**Conceitos a abordar:** Explicar o espectro eletromagnético e como diferentes materiais refletem diferentes comprimentos de onda. Apresentar bandas típicas de satélites (azul, verde, vermelho, infravermelho próximo, infravermelho de ondas curtas) e suas aplicações: vegetação aparece brilhante no infravermelho próximo, água absorve fortemente essa radiação, solos têm assinaturas espectrais características. Mostrar como isso difere fundamentalmente de imagens RGB convencionais e por que mais informação espectral permite análises mais sofisticadas.

###### **Diferentes resoluções espaciais, espectrais e temporais** (10min)
**Conceitos a abordar:** Definir resolução espacial (tamanho do pixel no terreno), espectral (número e largura das bandas), temporal (frequência de revisita), e radiométrica (precisão dos valores digitais). Explicar trade-offs: alta resolução espacial geralmente significa menor cobertura e revisita menos frequente. Mostrar exemplos de diferentes sensores (Landsat, Sentinel, Planet) e suas características. Conectar com implicações para deep learning: mais dados espectrais oferecem mais informação, mas também aumentam complexidade computacional.

###### **Comparação com imagens convencionais** (10min)
**Conceitos a abordar:** Contrastar imagens de sensoriamento remoto com fotografia digital convencional em aspectos fundamentais: número de bandas espectrais, faixas de valores (não limitados a 0-255), calibração radiométrica, e informação geoespacial (sistemas de coordenadas, projeções). Explicar como essas diferenças impactam processamento e requerem técnicas específicas de normalização e preparação de dados.

##### **Índices Espectrais** (25min)

###### **NDVI, NDWI e suas aplicações** (15min)
**Conceitos a abordar:** Apresentar NDVI (Normalized Difference Vegetation Index) como medida de vigor vegetativo, explicando a fórmula (NIR - Red)/(NIR + Red) e sua interpretação física: vegetação saudável tem alta reflectância no infravermelho próximo e baixa no vermelho. Introduzir NDWI (Normalized Difference Water Index) para detecção de corpos d'água. Explicar como esses índices transformam informação multiespectral em medidas físicas interpretáveis, e por que são frequentemente usados como features adicionais em modelos de machine learning.

###### **Prática no Colab: Calcular índices com dados simulados** (10min)
**Conceitos a abordar:** Implementar cálculo de NDVI e NDWI usando arrays NumPy simulando dados multiespectrais. Criar visualizações coloridas dos índices (mapas de calor) para interpretação intuitiva. Demonstrar como diferentes tipos de cobertura do solo (vegetação, água, solo nu, áreas urbanas) produzem valores característicos desses índices.

#### **3.2 Normalização: Ponte para Deep Learning (1h30min)**

##### **Por que Normalizar Dados de Sensoriamento Remoto** (30min)

###### **Problemas específicos: diferentes escalas de bandas** (15min)
**Conceitos a abordar:** Explicar como diferentes bandas espectrais têm faixas de valores naturalmente diferentes devido às propriedades físicas da radiação e resposta dos sensores. Por exemplo, valores típicos podem variar de centenas para bandas visíveis a milhares para infravermelhas. Demonstrar como isso pode fazer com que algumas bandas dominem o aprendizado de modelos, mascarando informações importantes de outras bandas. Usar exemplos concretos de histogramas de diferentes bandas para visualizar o problema.

###### **Impacto na convergência de redes neurais** (15min)
**Conceitos a abordar:** Explicar como dados não normalizados podem causar problemas de otimização: gradientes muito grandes ou muito pequenos, convergência lenta, instabilidade numérica. Mostrar através de analogias visuais (paisagem de otimização) como normalização "suaviza" a superfície de loss, facilitando o trabalho dos otimizadores. Conectar com conceitos matemáticos vistos no Módulo 1 sobre gradiente descendente.

##### **Técnicas de Normalização** (45min)

###### **Normalização por canal vs global** (20min)
**Conceitos a abordar:** Apresentar a abordagem de normalização por canal como método preferido para dados multiespectrais: calcular média e desvio padrão independentemente para cada banda espectral, aplicando z-score normalização. Contrastar com normalização global que usa estatísticas de todas as bandas juntas. Explicar por que normalização por canal preserva as características espectrais relativas entre bandas, enquanto global pode mascarar diferenças importantes. Demonstrar matematicamente: (pixel - média_canal) / desvpad_canal.

###### **Cálculo incremental de estatísticas para datasets grandes** (15min)
**Conceitos a abordar:** Abordar o problema prático de calcular estatísticas quando o dataset não cabe na memória. Apresentar algoritmo de cálculo incremental: iterar sobre o dataset acumulando soma e soma dos quadrados, depois calcular média e variância usando fórmulas estatísticas. Explicar a importância de usar apenas dados de treinamento para calcular estatísticas (evitar data leakage), aplicando as mesmas estatísticas em validação e teste.

###### **Prática intensiva: Implementação completa** (10min)
**Conceitos a abordar:** Implementar ambos algoritmos de normalização (por canal e global) com dados simulados multiespectrais. Calcular estatísticas incrementalmente simulando um dataset grande. Comparar distribuições antes e depois da normalização usando histogramas. Validar que dados normalizados têm média ≈ 0 e desvio padrão ≈ 1.

##### **Implementação Prática** (15min)

###### **Comparação visual dos resultados**
**Conceitos a abordar:** Criar visualizações lado-a-lado mostrando o efeito da normalização em imagens multiespectrais. Usar diferentes composições coloridas (true color, false color, índices espectrais) para demonstrar que normalização preserva informação relativa enquanto padroniza escalas. Mostrar histogramas before/after para cada banda.

###### **Validação com dados de satélite simulados**
**Conceitos a abordar:** Usar dados simulados que replicam características estatísticas de dados reais de sensoriamento remoto (diferentes médias e variâncias por banda, correlações entre bandas). Aplicar pipeline completo de normalização e validar resultados numericamente e visualmente.

### Módulo 4: Redes Neurais - Teoria e Implementação (8h30min)

#### **4.1 Fundamentos de Redes Neurais (1h30min)**

##### **O Perceptron e suas Limitações** (35min)

###### **Modelo matemático do neurônio** (15min)
**Conceitos a abordar:** Apresentar o perceptron como modelo simplificado de neurônio biológico, explicando componentes: entradas (xi), pesos (wi), bias (b), função de ativação degrau. Formalizar matematicamente: saída = f(Σ(wi × xi) + b) onde f é função degrau. Explicar interpretação geométrica: o perceptron define um hiperplano que separa espaço de entrada em duas regiões. Demonstrar como diferentes valores de pesos e bias movem e rotacionam este hiperplano.

###### **Prática: Implementação para problema AND/OR** (10min)
**Conceitos a abordar:** Implementar perceptron from scratch para resolver portas lógicas AND e OR. Mostrar como encontrar pesos manualmente que resolvem cada problema. Demonstrar que estes problemas são linearmente separáveis, visualizando dados de entrada e linha de separação no espaço 2D. Esta prática concreta consolidará a compreensão teórica.

###### **Limitações fundamentais: o problema XOR** (10min)
**Conceitos a abordar:** Apresentar o problema XOR como exemplo clássico de padrão não-linearmente separável. Mostrar visualmente que não existe linha reta que separe corretamente as quatro combinações de entrada. Tentar resolver XOR com perceptron simples para demonstrar empiricamente a falha. Esta limitação histórica motivará a necessidade de redes multicamadas e não-linearidades.

##### **Funções de Ativação** (30min)

###### **ReLU, Sigmoid, Tanh: propriedades matemáticas** (20min)
**Conceitos a abordar:** Apresentar cada função com sua fórmula matemática e gráfico. ReLU: f(x) = max(0,x) - simplicidade e eficiência computacional. Sigmoid: f(x) = 1/(1+e^(-x)) - saída entre 0 e 1, interpretação probabilística. Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - saída entre -1 e 1, centrada em zero. Explicar propriedades de cada uma: domínio, imagem, monotonia, continuidade.

###### **Por que as não-linearidades são essenciais** (10min)
**Conceitos a abordar:** Explicar matematicamente que composição de funções lineares resulta em função linear: se f(x) = ax + b e g(x) = cx + d, então f(g(x)) = a(cx + d) + b = acx + ad + b, que ainda é linear. Demonstrar que sem não-linearidades, uma rede neural com N camadas é equivalente a uma única transformação linear. Usar analogia visual: não-linearidades permitem "dobrar" e "torcer" o espaço, permitindo separar padrões complexos.

##### **Arquitetura de Redes Multicamadas** (25min)

###### **Estrutura básica de MLPs** (15min)
**Conceitos a abordar:** Definir Multi-Layer Perceptron como rede feedforward com uma ou mais camadas ocultas. Explicar terminologia: camada de entrada (não conta como "camada" da rede), camadas ocultas, camada de saída. Mostrar como neurônios de uma camada se conectam com todos os neurônios da camada seguinte (fully connected). Discutir como profundidade e largura das camadas afetam capacidade expressiva da rede.

###### **Representação matricial e notação** (10min)
**Conceitos a abordar:** Introduzir notação matemática padrão: W^(l) para matriz de pesos da camada l, b^(l) para vetor de bias, a^(l) para ativações. Mostrar como operação de uma camada pode ser expressa como: z^(l) = W^(l)a^(l-1) + b^(l), seguido por a^(l) = f(z^(l)) onde f é função de ativação. Esta notação será fundamental para compreender backpropagation posteriormente.

#### **4.2 Teoria do Treinamento: Como as Redes Aprendem (2h45min)**

##### **Forward Propagation: Fluxo de Dados** (50min)

###### **Matemática detalhada: transformações lineares e não-lineares** (20min)
**Conceitos a abordar:** Explicar cada etapa do forward pass matematicamente. Transformação linear: z = Wx + b onde z é entrada para ativação, W matriz de pesos, x entrada da camada anterior, b bias. Transformação não-linear: a = f(z) onde f é função de ativação. Mostrar como isso se repete camada por camada. Explicar dimensionalidade: se entrada tem n neurônios e saída tem m, então W é m×n, b é m×1, x é n×1, resultando em z de dimensão m×1.

###### **Propagação através de múltiplas camadas** (15min)
**Conceitos a abordar:** Demonstrar forward pass completo através de exemplo concreto: rede 3-2-1 (3 entradas, 2 neurônios na camada oculta, 1 saída). Rastrear valores numéricos através de cada transformação, mostrando como entrada inicial se transforma progressivamente até produzir saída final. Enfatizar que cada camada extrai representações progressivamente mais abstratas dos dados.

###### **Implementação manual: Forward pass from scratch** (15min)
**Conceitos a abordar:** Implementar forward pass usando apenas NumPy, sem frameworks de deep learning. Começar com exemplo simples (2-2-1) usando valores numéricos específicos. Calcular cada etapa manualmente, verificando dimensões matriciais. Esta implementação concreta consolida compreensão teórica e prepara para compreender o que PyTorch faz automaticamente.

##### **Funções de Perda: Quantificando o Erro** (30min)

###### **Mean Squared Error para regressão** (10min)
**Conceitos a abordar:** Apresentar MSE como L = (1/2n)Σ(ŷi - yi)² onde ŷi é predição, yi é valor verdadeiro, n é número de amostras. Explicar por que quadrado: penaliza erros grandes mais severamente, resulta em derivadas convenientes. Mostrar interpretação geométrica: distância Euclidiana no espaço de saída. Discutir quando usar MSE: problemas de regressão, saídas contínuas.

###### **Cross-entropy para classificação** (15min)
**Conceitos a abordar:** Apresentar cross-entropy como L = -Σ yi log(ŷi) para classificação multiclasse. Explicar conexão com teoria da informação: cross-entropy mede "surpresa" de predições incorretas. Para classificação binária: L = -[y log(ŷ) + (1-y) log(1-ŷ)]. Demonstrar por que cross-entropy é preferível a MSE para classificação: gradientes mais informativos, interpretação probabilística natural.

###### **Intuição geométrica da otimização** (5min)
**Conceitos a abordar:** Visualizar função de perda como paisagem multidimensional onde cada dimensão representa um parâmetro da rede. Objetivo do treinamento é encontrar ponto mais baixo (mínimo global) nesta paisagem. Mostrar como diferentes loss functions criam paisagens com características diferentes: convexas vs não-convexas, múltiplos mínimos locais.

##### **Hiperparâmetros Fundamentais: Os Controladores do Treinamento** (45min)

###### **Conceito de Batch e Batch Size** (15min)
**Conceitos a abordar:** Explicar que um batch é um subconjunto de dados processado simultaneamente pela rede. Definir batch size como número de amostras em cada batch. Mostrar por que usamos batches: limitações de memória GPU, estabilidade dos gradientes, eficiência computacional através de paralelização. Discutir trade-offs: batches grandes fornecem gradientes mais estáveis mas requerem mais memória; batches pequenos são mais ruidosos mas podem escapar de mínimos locais. Apresentar valores típicos (32, 64, 128) e como escolher adequadamente.

###### **Epochs: Ciclos Completos pelos Dados** (10min)
**Conceitos a abordar:** Definir epoch como uma passagem completa através de todo o dataset de treinamento. Explicar que se temos 1000 amostras e batch size 100, uma epoch consiste em 10 batches. Discutir por que múltiplas epochs são necessárias: o modelo precisa "ver" os dados muitas vezes para aprender padrões complexos. Apresentar conceitos de underfitting (poucas epochs) vs overfitting (muitas epochs), estabelecendo que número ideal de epochs varia por problema.

###### **Learning Rate: O Hiperparâmetro Mais Crítico** (20min)
**Conceitos a abordar:** Definir learning rate como multiplicador que controla o tamanho dos passos na direção do gradiente. Usar analogia visual: descendo uma montanha, learning rate determina o tamanho dos passos. Learning rate muito alto: passos grandes podem "pular" sobre o mínimo, causando oscilações ou divergência. Learning rate muito baixo: passos pequenos levam a convergência extremamente lenta, podendo travar em plateaus. Mostrar matematicamente: θ_novo = θ_antigo - α × ∇L, onde α é learning rate. Apresentar valores típicos (0.001 a 0.1) e estratégias de ajuste.

##### **Backpropagation: A Matemática dos Gradientes** (1h10min)

###### **Regra da cadeia aplicada a redes neurais** (25min)
**Conceitos a abordar:** Revisar regra da cadeia do cálculo: d/dx[f(g(x))] = f'(g(x)) × g'(x). Expandir para caso multivariável e múltiplas composições. Aplicar a redes neurais: para calcular ∂L/∂W^(l), precisamos usar cadeia: ∂L/∂W^(l) = ∂L/∂a^(L) × ∂a^(L)/∂z^(L) × ... × ∂z^(l+1)/∂a^(l) × ∂a^(l)/∂z^(l) × ∂z^(l)/∂W^(l). Cada termo representa derivada de uma operação específica na rede.

###### **Cálculo de gradientes camada por camada** (20min)
**Conceitos a abordar:** Demonstrar cálculo específico para cada tipo de operação. Para camada linear z = Wx + b: ∂z/∂W = x^T, ∂z/∂b = 1, ∂z/∂x = W^T. Para função de ativação a = f(z): ∂a/∂z = f'(z). Mostrar como gradientes "fluem" backwards: começando com ∂L/∂a^(L) da loss function, calculamos sucessivamente gradientes de cada camada usando regra da cadeia.

###### **Computational graphs e visualização do fluxo** (15min)
**Conceitos a abordar:** Introduzir computational graphs como representação visual das operações matemáticas. Cada nó representa operação ou variável, arestas representam dependências. Mostrar como forward pass segue direção das arestas, backward pass vai na direção oposta. Criar exemplos visuais simples: grafo para operação z = x × y + b, mostrando como gradientes se propagam backwards.

###### **Demonstração passo a passo: Backprop em rede simples** (15min)
**Conceitos a abordar:** Trabalhar exemplo numérico completo com rede pequena (2-2-1), valores específicos, e cálculos manuais. Começar com forward pass para estabelecer valores das ativações. Calcular perda e iniciar backward pass, calculando cada gradiente numericamente. Verificar resultados usando diferenciação numérica (aproximação por diferenças finitas). Esta demonstração concreta é crucial para solidificar compreensão.

###### **Intuição: como os gradientes "fluem" de volta** (5min)
**Conceitos a abordar:** Desenvolver intuição sobre o significado dos gradientes: indicam quanto cada parâmetro deve mudar para reduzir a perda. Gradientes grandes significam que pequenas mudanças no parâmetro causam grandes mudanças na perda. Explicar por que gradientes "fluem" backwards: começamos sabendo como perda muda com saída, usamos regra da cadeia para descobrir como muda com parâmetros internos.

#### **4.3 Implementação com PyTorch (2h30min)**

##### **PyTorch Fundamentals** (45min)

###### **Tensors vs NumPy: diferenças essenciais** (15min)
**Conceitos a abordar:** Comparar tensors PyTorch com arrays NumPy: ambos são containers multidimensionais, mas tensors têm capacidades adicionais cruciais para deep learning. Tensors podem rastrear gradientes (autograd), são otimizados para GPUs, integram-se nativamente com redes neurais. Mostrar conversões entre tensors e NumPy arrays, enfatizando quando usar cada um.

###### **Criação e manipulação básica de tensors** (15min)
**Conceitos a abordar:** Demonstrar criação de tensors: torch.tensor(), torch.zeros(), torch.ones(), torch.randn(). Mostrar operações básicas: indexing, slicing, reshape, transpose. Explicar tipos de dados (dtype) e dispositivos (device). Introduzir conceito de in-place vs out-of-place operations. Mostrar como verificar propriedades: .shape, .dtype, .device.

###### **Prática: Conversão de operações NumPy para PyTorch** (15min)
**Conceitos a abordar:** Refazer exercícios do Módulo 1 usando PyTorch tensors ao invés de NumPy arrays. Implementar operações matriciais, broadcasting, e manipulações básicas. Comparar sintaxe e resultados, destacando similaridades e diferenças. Esta prática facilita transição mental de NumPy para PyTorch.

##### **Autograd: Diferenciação Automática Prática** (50min)

###### **O que é requires_grad e quando usar** (15min)
**Conceitos a abordar:** Explicar que requires_grad=True marca tensors para rastreamento de gradientes. Mostrar que apenas tensors que precisam ser otimizados (parâmetros do modelo) devem ter requires_grad=True. Demonstrar que operações entre tensors com requires_grad=True resultam em tensor que também rastreia gradientes. Explicar implicações de performance: rastreamento consome memória e tempo computacional.

###### **Método .backward() e acúmulo de gradientes** (15min)
**Conceitos a abordar:** Demonstrar que .backward() calcula gradientes de saída escalar em relação a todos os tensors com requires_grad=True. Explicar que PyTorch acumula gradientes por padrão: múltiplas chamadas de .backward() somam gradientes. Mostrar problema prático: gradientes de batches anteriores contaminam cálculos atuais. Introduzir necessidade de zerar gradientes explicitamente.

###### **Prática intensiva: Gradientes em funções simples** (15min)
**Conceitos a abordar:** Implementar exemplos progressivamente mais complexos: função quadrática simples, função com múltiplas variáveis, composição de funções. Para cada exemplo, calcular gradientes manualmente e comparar com PyTorch autograd. Experimentar com acúmulo de gradientes intencionalmente para compreender comportamento.

###### **torch.no_grad() para validação** (5min)
**Conceitos a abordar:** Explicar que durante validação/teste não precisamos calcular gradientes, pois não vamos atualizar parâmetros. torch.no_grad() desativa rastreamento, economizando memória e acelerando computação. Demonstrar diferenças de uso de memória com/sem torch.no_grad() em exemplos práticos.

##### **Estrutura nn.Module e Otimizadores** (55min)

###### **Construção de MLPs com PyTorch** (20min)
**Conceitos a abordar:** Apresentar nn.Module como classe base para todos os modelos PyTorch. Explicar estrutura: herdar de nn.Module, definir layers em __init__, implementar forward pass em forward(). Mostrar nn.Linear como implementação de camada fully connected. Demonstrar como PyTorch automaticamente rastreia parâmetros definidos como atributos da classe.

###### **Funções de Perda (Loss Functions)** (15min)
**Conceitos a abordar:** Apresentar nn.CrossEntropyLoss para classificação multiclasse, explicando que combina softmax e negative log-likelihood. Mostrar nn.MSELoss para regressão. Explicar que loss functions esperam shapes específicos: CrossEntropyLoss espera logits (scores brutos) e labels como índices inteiros. Demonstrar uso prático: instanciar loss function, calcular perda entre predições e targets.

###### **Otimizadores: SGD e Adam** (20min)
**Conceitos a abordar:** Apresentar torch.optim.SGD como implementação do Stochastic Gradient Descent, explicando parâmetros: lr (learning rate), momentum (acelera convergência), weight_decay (regularização L2). Introduzir torch.optim.Adam como algoritmo adaptativo que ajusta learning rate para cada parâmetro, explicando que geralmente funciona bem com configurações padrão. Mostrar como instanciar otimizadores passando model.parameters() e como usar optimizer.step() para atualizar pesos, optimizer.zero_grad() para zerar gradientes.

#### **4.4 Training Loop Completo: Integrando Tudo (1h45min)**

##### **Anatomia do Training Loop** (1h5min)

###### **Estrutura fundamental: forward, loss, backward, step** (25min)
**Conceitos a abordar:** Reforçar os quatro passos essenciais do treinamento, agora conectando com conceitos de hiperparâmetros: 1) Forward pass: propagar batch de dados através da rede, 2) Loss calculation: comparar predições com verdades usando função de perda apropriada, 3) Backward pass: calcular gradientes via backpropagation usando .backward(), 4) Optimizer step: atualizar parâmetros usando gradientes com learning rate definido. Enfatizar que esta sequência se repete para cada batch dentro de cada epoch.

###### **Dataset e DataLoader para MNIST** (15min)
**Conceitos a abordar:** Introduzir MNIST como dataset clássico: 70.000 imagens 28×28 de dígitos manuscritos. Mostrar como carregar usando torchvision.datasets. Apresentar DataLoader como wrapper que implementa conceito de batching: divide dataset em mini-batches do tamanho especificado, fornece shuffling para randomizar ordem das amostras, permite loading paralelo. Demonstrar iteração através do DataLoader para acessar batches, mostrando shapes dos tensors.

###### **Modo train vs eval: quando e por que** (10min)
**Conceitos a abordar:** Explicar que model.train() e model.eval() controlam comportamento de certas camadas durante treinamento vs inferência. Embora ainda não tenham visto Dropout ou Batch Normalization, explicar conceitualmente que algumas operações se comportam diferentemente. Estabelecer boa prática: sempre model.train() antes de training loops, model.eval() antes de validação.

###### **Implementação guiada: Primeiro training loop funcional** (15min)
**Conceitos a abordar:** Implementar training loop completo para MNIST usando MLP simples, reforçando conceitos de hiperparâmetros na prática. Mostrar como escolher batch_size adequado para memória disponível, como definir número de epochs baseado na observação de convergência, como ajustar learning rate observando comportamento da loss. Incluir progress tracking básico: print loss a cada N batches. Ver convergência acontecendo em tempo real, consolidando conexão entre teoria e prática.

##### **Monitoramento e Debugging** (40min)

###### **Interpretação de loss curves e hiperparâmetros** (15min)
**Conceitos a abordar:** Mostrar como diferentes hiperparâmetros afetam formato das loss curves. Learning rate muito alto: oscilações erráticas ou divergência. Learning rate muito baixo: convergência extremamente lenta. Batch size pequeno: curvas mais ruidosas. Epochs insuficientes: underfitting (loss ainda decrescendo). Epochs excessivas: overfitting (gap entre train e validation loss). Desenvolver habilidade de "ler" loss curves para diagnosticar problemas de hiperparâmetros.

###### **TensorBoard básico para visualização** (15min)
**Conceitos a abordar:** Introduzir TensorBoard como ferramenta de visualização para experimentos ML. Mostrar setup básico: SummaryWriter, logging scalar values (loss, accuracy). Demonstrar interface web básica: scalar plots, comparação entre runs com diferentes hiperparâmetros. Explicar benefits: track experimentos, compare hyperparameters, share results. Setup para uso throughout remainder of course.

###### **Problemas comuns e como resolvê-los** (10min)
**Conceitos a abordar:** Catalogar problemas típicos que beginners encounter: gradientes não sendo calculados (esquecer requires_grad), gradientes explodindo (learning rate muito alto), gradientes desaparecendo (rede muito profunda sem técnicas adequadas), loss não mudando (learning rate muito baixo ou bug no código), memory errors (batch_size muito grande). Para cada problema, mostrar symptoms e soluções. Esta seção prática economiza muito tempo de debugging later.

### Módulo 5: Introdução às CNNs (2h30min)

#### **5.1 Da Convolução Clássica às CNNs (1h15min)**

##### **Limitações de MLPs para Imagens** (25min)

###### **Perda de informação espacial ao "achatar" imagens** (10min)
**Conceitos a abordar:** Demonstrar que MLPs tratam cada pixel independentemente, ignorando relações espaciais entre pixels vizinhos. Mostrar como flatten() operation destrói estrutura 2D da imagem. Usar analogia: é como tentar reconhecer faces olhando apenas lista de valores de pixels sem saber posições. Explicar que informação espacial é crucial para visão: bordas, texturas, formas são padrões espaciais.

###### **Explosão de parâmetros com imagens grandes** (15min)
**Conceitos a abordar:** Calcular número de parâmetros para MLP com imagens realistas. Exemplo: imagem 224×224×3 = 150.528 pixels. Primeira camada oculta com 1000 neurônios teria 150.528.000 parâmetros apenas na primeira conexão! Mostrar que isso é computacionalmente intratável e propenso a overfitting. Estabelecer necessidade de arquiteturas mais eficientes que exploram estrutura espacial.

##### **Operação de Convolução Aprendida** (30min)

###### **Conexão explícita com filtros clássicos do Módulo 2** (20min)
**Conceitos a abordar:** Fazer conexão direta e explícita com os filtros implementados no Módulo 2. Relembrar que implementamos filtros de Sobel para detecção de bordas, filtros Gaussianos para suavização, filtros Laplacianos para detecção de mudanças. Explicar insight revolucionário: ao invés de projetar filtros manualmente (como fizemos com Sobel), deixamos a rede aprender os filtros ideais para a tarefa específica. Mostrar que uma camada convolucional é essencialmente um banco de filtros aprendidos automaticamente. Demonstrar que durante o treinamento, alguns filtros podem naturalmente aprender a detectar bordas (similar ao Sobel), outros podem aprender texturas, formas, padrões mais complexos.

###### **Conceitos fundamentais: padding, stride, receptive field** (10min)
**Conceitos a abordar:** Revisar padding e stride do Módulo 2, mas agora no contexto de layers aprendidos. Introduzir receptive field como conceito crucial: região da entrada que influencia uma ativação específica. Mostrar como receptive field cresce com profundidade da rede, permitindo que layers mais profundas "vejam" porções maiores da imagem. Explicar que isso cria detecção hierárquica de características: layers iniciais detectam características simples, layers profundas combinam em padrões complexos.

##### **Arquitetura CNN Básica** (20min)

###### **Camadas convolucionais, pooling e fully connected** (15min)
**Conceitos a abordar:** Apresentar building blocks típicos de CNNs. Convolutional layers: extraem características espaciais usando filtros aprendidos (conectar com filtros clássicos). Pooling layers (max, average): fazem downsampling dos feature maps, fornecem invariância a translações, reduzem custo computacional. Fully connected layers: classificação final baseada em características extraídas. Mostrar padrão arquitetural típico: CONV → POOL → CONV → POOL → FC → FC → OUTPUT.

###### **Feature maps e hierarquia de características** (5min)
**Conceitos a abordar:** Explicar que cada filtro produz um feature map destacando onde aquela característica ocorre na imagem. Layers iniciais: características simples (bordas, cores) similar aos filtros clássicos. Layers intermediárias: formas, texturas. Layers profundas: objetos complexos, cenas. Esta representação hierárquica é insight-chave que torna CNNs poderosas para tarefas de visão.

#### **5.2 LeNet no MNIST: Primeira CNN Prática (1h15min)**

##### **Implementação LeNet-5** (45min)

###### **Arquitetura detalhada e motivação histórica** (15min)
**Conceitos a abordar:** Apresentar LeNet-5 como arquitetura CNN pioneira (LeCun, 1998). Detalhar arquitetura específica: 32×32 input → C1 (6 filtros 5×5) → S2 (2×2 avg pool) → C3 (16 filtros 5×5) → S4 (2×2 avg pool) → C5 (120 filtros 5×5) → F6 (84 neurônios) → OUTPUT (10 classes). Explicar escolhas de design e como elas endereçaram desafios específicos de reconhecimento de dígitos.

###### **Implementação guiada: LeNet from scratch** (20min)
**Conceitos a abordar:** Implementar LeNet usando nn.Conv2d, nn.AvgPool2d, nn.Linear. Explicar parâmetros de cada layer: in_channels, out_channels, kernel_size, stride, padding. Mostrar como calcular dimensões de saída após cada operação. Guiar através da implementação do método forward(), conectando cada operação ao diagrama arquitetural.

###### **Adaptação do training loop para CNNs** (10min)
**Conceitos a abordar:** Mostrar que training loop fundamentals permanecem os mesmos, mas dados de entrada precisam preprocessamento diferente. Imagens MNIST são 28×28, mas LeNet espera 32×32 - mostrar padding. Demonstrar processamento em batch através de CNN: input shape muda de (batch, 1, 28, 28) através de várias layers. Rastrear tensor shapes em cada estágio para construir intuição sobre fluxo de dados em CNN.

##### **Análise Comparativa** (30min)

###### **MLP vs CNN no mesmo dataset MNIST** (15min)
**Conceitos a abordar:** Treinar tanto MLP quanto LeNet no MNIST usando setup de treinamento idêntico. Comparar tempo de treinamento, acurácia final, contagem de parâmetros, uso de memória. Demonstrar superioridade da CNN empiricamente. Analisar matrizes de confusão para ver se CNN comete diferentes tipos de erros. Esta comparação concretiza vantagens teóricas discutidas anteriormente.

###### **Visualização de feature maps e filtros aprendidos** (10min)
**Conceitos a abordar:** Extrair e visualizar filtros aprendidos da primeira camada convolucional. Mostrar feature maps produzidos por diferentes filtros ao processar imagens de amostra. Demonstrar que a rede automaticamente aprendeu detectores de borda, detectores de blob, etc. similares aos filtros clássicos mas otimizados para tarefa de reconhecimento de dígitos. Esta visualização conecta representações aprendidas de volta aos conceitos de processamento clássico de imagens.

###### **Preparação conceitual: Por que CNNs são ideais para sensoriamento remoto** (5min)
**Conceitos a abordar:** Conectar insights do experimento MNIST a aplicações de sensoriamento remoto. Imagens de satélite têm estrutura espacial similar a imagens naturais: estradas são características lineares, campos têm texturas consistentes, áreas urbanas têm padrões característicos. CNNs podem aprender estes padrões espaciais automaticamente, tornando-as ideais para classificação de cobertura do solo, detecção de mudanças, reconhecimento de objetos em imagens de satélite. Esta conexão prepara estudantes mentalmente para aplicações da segunda semana.

## SEGUNDA SEMANA - MODALIDADE PRESENCIAL (40 horas)

### Dia 1: Consolidação, Regularização e Ferramentas (8h)

#### **Manhã - Síntese e Técnicas de Regularização (4h)**

##### **Momento de Síntese e Revisão** (45min)
**Conceitos a abordar:** Facilitar discussão ativa conectando todos os conceitos da primeira semana. Revisar progressão: processamento clássico → dados de sensoriamento remoto → normalização → teoria de redes neurais → implementação PyTorch → CNNs básicas. Identificar e resolver gaps de compreensão através de Q&A intensiva. Abordar problemas comuns encontrados durante primeira semana e esclarecer mal-entendidos.

##### **Overfitting e Diagnóstico** (45min)
**Conceitos a abordar:** Demonstrar overfitting empiricamente modificando experimento MNIST: usar apenas pequeno subset dos dados de treinamento para forçar overfitting. Plotar curvas de loss de treinamento vs validação mostrando padrão característico: loss de treinamento continua decrescendo enquanto loss de validação começa a aumentar. Explicar causa subjacente: modelo memoriza exemplos de treinamento ao invés de aprender padrões generalizáveis. Introduzir conjunto de validação como ferramenta crucial para detectar overfitting durante treinamento.

##### **Técnicas de Regularização** (2h30min)

###### **Dropout** (45min)
**Conceitos a abordar:** Explicar dropout como regularização estocástica que randomicamente "desliga" neurônios durante treinamento. Mostrar formulação matemática e implementar from scratch para construir intuição. Demonstrar comportamento diferente durante modos train vs eval: durante treinamento, randomicamente zera neurônios e escala os restantes; durante inferência, usa todos os neurônios. Implementar dropout em exemplos anteriores MLP/CNN e observar impacto no overfitting através de curvas de loss.

###### **Weight Decay** (30min)
**Conceitos a abordar:** Apresentar weight decay (regularização L2) como técnica para penalizar pesos grandes. Explicar formulação matemática: adicionar λ||W||² à função de perda. Mostrar como isso encoraja pesos menores, levando a fronteiras de decisão mais suaves e melhor generalização. Implementar através do parâmetro weight_decay do otimizador PyTorch. Comparar regularização L1 vs L2 conceitualmente e demonstrar empiricamente.

###### **Batch Normalization** (45min)
**Conceitos a abordar:** Introduzir conceito de internal covariate shift: distribuições das ativações mudam durante treinamento conforme parâmetros se atualizam. Explicar solução batch normalization: normalizar ativações para ter média zero e variância unitária dentro de cada batch. Mostrar formulação matemática com parâmetros de scale/shift aprendíveis. Implementar BatchNorm em exemplo CNN e observar impacto na estabilidade de treinamento e velocidade de convergência. Discutir posicionamento: antes vs depois das funções de ativação.

###### **Early Stopping** (30min)
**Conceitos a abordar:** Apresentar early stopping como técnica de regularização simples mas efetiva. Explicar conceito de patience: parar treinamento quando loss de validação não melhora por número especificado de epochs. Implementar lógica de early stopping com model checkpointing para salvar melhor modelo. Demonstrar na prática usando exemplos anteriores, mostrando como previne overfitting mantendo boa performance.

#### **Tarde - Ferramentas Geoespaciais (4h)**

##### **QGIS Hands-on** (2h)
**Conceitos a abordar:** Guiar estudantes através de sessão hands-on QGIS para familiarizar com visualização de dados geoespaciais. Carregar dados vetoriais (shapefiles) representando diferentes classes de cobertura do solo. Carregar dados raster correspondentes (imagens de satélite) e garantir alinhamento adequado. Explorar tabelas de atributos, sistemas de referência de coordenadas, e relações espaciais. Praticar styling tanto de layers vetoriais quanto raster para visualização efetiva. Esta sessão prática constrói intuição espacial crucial para restante do curso.

##### **Rasterio e GeoPandas** (2h)
**Conceitos a abordar:** Introduzir manipulação programática de dados geoespaciais usando bibliotecas Python. Usar Rasterio para ler imagens de satélite: compreender bandas, resolução espacial, sistemas de coordenadas, valores nodata. Usar GeoPandas para manipulação de dados vetoriais: carregar shapefiles, examinar geometrias, realizar queries espaciais. Demonstrar conversão entre sistemas de referência de coordenadas. Criar pipeline de dados geoespaciais brutos para arrays NumPy adequados para deep learning, estabelecendo fundação para preparação de datasets.

### Dia 2: CNNs Avançadas e Aplicações em SR (8h)

#### **Manhã - Arquiteturas CNN Clássicas (4h)**

##### **VGG Implementation** (1h30min)
**Conceitos a abordar:** Apresentar filosofia da arquitetura VGG: redes mais profundas através de filtros pequenos (3×3). Explicar princípios arquiteturais: empilhar filtros pequenos fornece mesmo receptive field que filtros grandes mas com mais não-linearidade e menos parâmetros. Implementar arquitetura tipo VGG adaptada para MNIST, demonstrando design modular com blocos repetidos. Comparar performance contra LeNet, observando benefícios da profundidade aumentada. Discutir trade-offs computacionais: acurácia vs velocidade/memória.

##### **ResNet e Skip Connections** (1h30min)
**Conceitos a abordar:** Introduzir conexões residuais como solução para problema de gradientes evanescentes em redes profundas. Explicar formulação matemática: y = F(x) + x onde F(x) é mapeamento residual aprendido. Mostrar como skip connections permitem treinar redes muito mais profundas permitindo que gradientes fluam diretamente através de atalhos. Implementar bloco ResNet básico e demonstrar na prática. Comparar comportamento de convergência de redes profundas com/sem skip connections.

##### **Transfer Learning** (1h)
**Conceitos a abordar:** Introduzir conceito de transfer learning: aproveitar modelos pré-treinados treinados em datasets grandes (ImageNet) para novas tarefas. Explicar duas abordagens principais: extração de características (congelar layers pré-treinados, treinar apenas classificador) vs fine-tuning (ajustar todas as layers com learning rate menor). Demonstrar usando modelos pré-treinados do torchvision. Abordar desafios de adaptação para dados multiespectrais de sensoriamento remoto: modelos pré-treinados RGB vs entradas multi-band. Mostrar estratégias para lidar com incompatibilidade de canais.

#### **Tarde - Classificação de Cenas WHU-RS19 (4h)**

##### **Dataset WHU-RS19** (1h)
**Conceitos a abordar:** Introduzir WHU-RS19 como dataset benchmark de sensoriamento remoto contendo 19 categorias de cenas (agrícola, comercial, residencial, etc.). Explorar estrutura do dataset, distribuição de classes, e características das imagens. Carregar e visualizar amostras representativas de cada classe. Criar data loaders com preprocessamento apropriado: normalização específica para dados de sensoriamento remoto, data augmentation adequada para imagens de satélite. Analisar balanceamento de classes e discutir implicações para estratégias de treinamento.

##### **CNN Especializada para SR** (2h)
**Conceitos a abordar:** Adaptar arquiteturas CNN especificamente para classificação de cenas de sensoriamento remoto. Discutir modificações necessárias: normalização de entrada adaptada para características espectrais, escolhas arquiteturais considerando resolução espacial e variações de escala em dados de sensoriamento remoto. Implementar pipeline completo de treinamento incluindo funções de perda apropriadas, métricas (acurácia, F1-score por classe), e protocolos de avaliação. Abordar desbalanceamento de classes através de funções de perda ponderadas ou estratégias de sampling.

##### **Análise e Comparação** (1h)
**Conceitos a abordar:** Conduzir comparação sistemática de diferentes arquiteturas CNN no dataset WHU-RS19. Gerar matrizes de confusão para analisar performance por classe e identificar pares de classes desafiadores. Visualizar características aprendidas através de análise de feature maps e plots t-SNE de representações aprendidas. Discutir resultados no contexto de desafios de sensoriamento remoto: similaridade espectral entre classes, variações de escala espacial, condições de imageamento. Extrair insights relevantes para aplicações de sensoriamento remoto do mundo real.

### Dia 3: Preparação Profissional de Dados (8h)

#### **Manhã - Pipeline Vetor → Raster (4h)**

##### **Dataset ISPRS Potsdam** (1h)
**Conceitos a abordar:** Introduzir ISPRS Potsdam como dataset benchmark de segmentação semântica urbana. Explorar estrutura do dataset: imagens aéreas de alta resolução (5cm GSD) com 6 classes de cobertura do solo (edificação, estrada, vegetação, árvore, carro, clutter). Analisar características espaciais: complexidade urbana, distribuições de classes, qualidade de anotações. Compreender divisões do dataset e protocolos de avaliação usados na comunidade de sensoriamento remoto.

##### **Criação de Máscaras** (2h)
**Conceitos a abordar:** Implementar pipeline completo para converter anotações vetoriais em máscaras raster adequadas para segmentação semântica. Usar GeoPandas para carregar e limpar geometrias vetoriais, lidando com erros de topologia e polígonos sobrepostos. Implementar processo de rasterização usando Rasterio/GDAL, garantindo alinhamento adequado entre imagens e máscaras. Abordar desafios práticos: efeitos de borda, pixels mistos, prioridades de classes durante resolução de sobreposição. Validar qualidade das máscaras através de visualização QGIS.

##### **Estratégias de Tiling** (1h)
**Conceitos a abordar:** Projetar estratégia de tiling para lidar com imagens grandes que excedem limitações de memória GPU. Implementar abordagem de sliding window com overlap configurável para preservar contexto através de fronteiras de tiles. Desenvolver critérios de filtragem para selecionar tiles informativos: excluir tiles com nodata excessivo, garantir representação mínima de classes, evitar tiles com qualidade de imagem ruim. Criar estrutura de metadados para rastrear proveniência de tiles e permitir reconstrução de predições em escala completa.

#### **Tarde - Dataset PyTorch e Balanceamento (4h)**

##### **Dataset Customizado para SR** (1h30min)
**Conceitos a abordar:** Implementar classe Dataset PyTorch projetada especificamente para aplicações de sensoriamento remoto. Abordar desafios únicos: tamanhos grandes de imagens requerendo lazy loading, imagens multi-band com diferentes tipos de dados, manipulação de sistemas de referência de coordenadas, gerenciamento de valores nodata. Implementar estratégias eficientes de caching para tiles acessados frequentemente. Criar pipeline flexível de preprocessamento suportando várias transformações e esquemas de normalização específicos para dados espectrais.

##### **Balanceamento de Classes** (1h30min)
**Conceitos a abordar:** Analisar distribuição de classes no dataset ISPRS, quantificando severidade do desbalanceamento usando métricas como Imbalance Ratio. Implementar estratégia de weighted random sampling para garantir representação balanceada durante batches de treinamento. Calcular class weights apropriados usando weighted por frequência inversa ou abordagens baseadas em effective number. Demonstrar impacto através de experimentos de treinamento comparando estratégias de sampling balanceadas vs desbalanceadas na performance de métricas do modelo.

##### **Focal Loss para Classes Raras** (1h)
**Conceitos a abordar:** Introduzir Focal Loss como solução avançada para desbalanceamento extremo de classes comum em aplicações de sensoriamento remoto. Explicar formulação matemática: FL(pt) = -α(1-pt)^γ log(pt), mostrando como parâmetro de focusing γ down-weights exemplos fáceis. Implementar Focal Loss em PyTorch e integrar no pipeline de treinamento de segmentação. Comparar performance contra cross-entropy padrão e cross-entropy ponderada através de experimentos controlados em subsets desbalanceados do ISPRS.

### Dia 4: Segmentação Semântica (8h)

#### **Manhã - U-Net Implementation (4h)**

##### **Arquitetura U-Net Completa** (2h)
**Conceitos a abordar:** Implementar arquitetura U-Net completa from scratch, explicando estrutura encoder-decoder com skip connections. Detalhar cada componente: contracting path (encoder) para capturar contexto, expanding path (decoder) para localização precisa, skip connections para combinar características de alta resolução com saída upsampled. Abordar detalhes de implementação: estratégias de upsampling (transpose convolution vs interpolação bilinear), concatenação de skip connections, design da camada final para saída multiclasse. Adaptar arquitetura para 6 classes do dataset ISPRS.

##### **Loss Functions para Segmentação** (1h)
**Conceitos a abordar:** Implementar funções de perda especializadas para tarefas de segmentação semântica. Cross-entropy ponderada por frequência de classe para lidar com desbalanceamento. Dice loss baseada em coeficiente de overlap, particularmente efetiva para tarefas de segmentação com classes positivas esparsas. Implementar losses combinadas (e.g., soma ponderada de cross-entropy e Dice) para aproveitar benefícios de ambas abordagens. Explicar quando cada função de perda é mais apropriada baseada em distribuição de classes e requisitos da tarefa.

##### **Training Pipeline Especializado** (1h)
**Conceitos a abordar:** Desenvolver pipeline completo de treinamento adaptado para tarefas de segmentação. Implementar cálculo de IoU (Intersection over Union) durante treinamento para monitorar qualidade de segmentação. Criar loop de validação com reporting de IoU por classe e cálculo de mean IoU. Implementar estratégias de learning rate scheduling apropriadas para tarefas de segmentação. Setup TensorBoard logging para rastrear métricas, curvas de loss, e predições de amostra durante treinamento.

#### **Tarde - Data Augmentation e Otimização (4h)**

##### **Albumentations para Sensoriamento Remoto** (2h)
**Conceitos a abordar:** Implementar pipeline abrangente de data augmentation usando biblioteca Albumentations, garantindo preservação de correspondência mask-imagem. Projetar estratégias de augmentation específicas para sensoriamento remoto: transformações geométricas (rotação, flipping, scaling) para simular diferentes viewpoints, augmentations fotométricas (brightness, contrast, noise) para simular diferentes condições atmosféricas/sensor, augmentations especializadas como CoarseDropout para simular cobertura de nuvens. Validar pipeline de augmentation através de inspeção visual e garantir que estratégias de augmentation não violem restrições físicas.

##### **Técnicas de Otimização Avançadas** (2h)
**Conceitos a abordar:** Implementar mixed precision training usando automatic mixed precision (AMP) do PyTorch para reduzir uso de memória e acelerar treinamento. Demonstrar técnicas de gradient accumulation para efetivamente aumentar batch size além das limitações de memória GPU. Implementar estratégias de learning rate scheduling: step decay, cosine annealing, warm-up schedules. Monitorar eficiência de treinamento: uso de memória, utilização de GPU, velocidade de treinamento. Comparar eficiência de treinamento e performance final através de diferentes estratégias de otimização.

### Dia 5: Estado da Arte e Projeto Final (8h)

#### **Manhã - Arquiteturas Especializadas (4h)**

##### **Segmentation Models PyTorch** (2h)
**Conceitos a abordar:** Explorar biblioteca segmentation_models_pytorch fornecendo arquiteturas de segmentação estado-da-arte com encoders pré-treinados. Comparar diferentes encoder backbones (ResNet, EfficientNet, VGG) e arquiteturas de decoder (U-Net, FPN, PSPNet, DeepLabV3+). Conduzir avaliação sistemática no dataset ISPRS, analisando trade-offs entre acurácia, velocidade de inferência, e requisitos de memória. Demonstrar efetividade de transfer learning usando encoders pré-treinados ImageNet para tarefas de sensoriamento remoto.

##### **DeepLab v3+ Detalhado** (2h)
**Conceitos a abordar:** Deep dive na arquitetura DeepLabV3+, focando em inovações cruciais para tarefas de predição densa. Explicar atrous (dilated) convolutions para aumentar receptive field sem perder resolução. Detalhar módulo Atrous Spatial Pyramid Pooling (ASPP) para capturar informação contextual multi-escala. Analisar mecanismo de refinement do decoder para recuperar detalhes espaciais finos. Implementar componentes-chave e comparar performance contra baseline U-Net em métricas de segmentação.

#### **Tarde - Projeto Integrador (4h)**

##### **Aplicação em Dataset Brasileiro** (2h)
**Conceitos a abordar:** Aplicar pipeline completo de deep learning a dataset brasileiro de sensoriamento remoto, implementando workflow end-to-end de dados brutos a resultados finais. Abordar desafios específicos do dataset: diferentes características espectrais, tipos únicos de cobertura do solo, escalas espaciais variadas. Implementar preprocessamento apropriado, seleção de modelo, estratégias de treinamento, e protocolos de avaliação. Gerar insights acionáveis relevantes para aplicações brasileiras de monitoramento ambiental.

##### **Apresentação e Síntese Final** (2h)
**Conceitos a abordar:** Estudantes apresentam resultados de seus projetos, demonstrando mastery de pipeline completo desde preparação de dados até avaliação de modelo. Facilitar peer review e discussão de diferentes abordagens tomadas por diferentes grupos. Sintetizar lições-chave aprendidas ao longo do curso: importância de qualidade de dados, métricas de avaliação apropriadas, considerações computacionais, integração de expertise de domínio. Fornecer guidance para próximos passos: arquiteturas avançadas, aplicações especializadas, considerações de deployment, direções de pesquisa em deep learning para sensoriamento remoto.
