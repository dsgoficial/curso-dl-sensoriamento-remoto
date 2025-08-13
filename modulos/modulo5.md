# Módulo 5: Introdução às CNNs

Este módulo aprofunda a compreensão das Redes Neurais Convolucionais (CNNs), uma arquitetura fundamental no campo do Deep Learning, especialmente para tarefas de visão computacional e sensoriamento remoto. Serão exploradas as limitações das Redes Neurais Multicamadas (MLPs) para o processamento de imagens, a operação revolucionária da convolução aprendida, a arquitetura básica das CNNs e uma implementação prática da pioneira LeNet-5 no conjunto de dados MNIST.

## 5.1 Da Convolução Clássica às CNNs

### Limitações de MLPs para Imagens

As Redes Neurais Multicamadas (MLPs), também conhecidas como redes totalmente conectadas, foram um marco inicial no Deep Learning. No entanto, sua aplicação direta a problemas de visão computacional, especialmente com imagens, revela desafios significativos que as tornam ineficientes e, em muitos casos, impraticáveis.

#### Perda de informação espacial ao "achatar" imagens

MLPs foram concebidas para processar dados vetoriais unidimensionais. Ao lidar com imagens, que possuem uma estrutura espacial bidimensional (ou tridimensional, para imagens coloridas), uma etapa de pré-processamento indispensável para MLPs é o "achatamento" (flattening). Esta operação transforma a grade de pixels 2D em um vetor 1D. Por exemplo, uma imagem em tons de cinza de 28x28 pixels, ao ser achatada, torna-se um vetor de 784 elementos.

Embora os valores dos pixels sejam preservados, suas posições relativas — que definem padrões locais como bordas, texturas e formas — são completamente descartadas. Uma MLP trata cada pixel como uma característica independente, ignorando as relações espaciais entre pixels vizinhos. O fato de uma MLP não utilizar informações espaciais significa que a ordem em que os pixels são alimentados não importa, desde que a mesma ordem seja mantida para todas as entradas. Isso ressalta a invariância de permutação das MLPs em relação às características de entrada.

Para ilustrar essa perda, pode-se usar a analogia de tentar reconhecer um rosto humano. Um ser humano reconhece um rosto pela disposição espacial de características como olhos, nariz e boca. Se fossem fornecidos apenas os valores de pixels de uma imagem de rosto, sem qualquer informação sobre suas posições 2D originais, seria como tentar reconhecer o rosto apenas conhecendo os valores de cor de pontos individuais, mas sem saber como eles formam uma estrutura coerente. As relações espaciais cruciais que definem as características faciais (por exemplo, a curva de uma sobrancelha, a linha de uma mandíbula) são perdidas. De forma análoga, para uma MLP, a entrada é apenas uma longa lista de números, e ela não possui um mecanismo inerente para entender que pixel[i] e pixel[i+1] eram adjacentes na imagem original, enquanto pixel[j] (distante no vetor achatado) poderia ter sido espacialmente próximo a pixel[i] na grade 2D original.

A ausência de uma premissa embutida de que pixels próximos são mais relacionados do que pixels distantes é uma limitação fundamental. Isso significa que a rede não possui um "viés indutivo" para dados espaciais. Sem esse viés, uma MLP precisa aprender essas relações espaciais do zero, o que é extremamente ineficiente e exige uma quantidade massiva de dados para ser eficaz. Essa restrição fundamental foi a principal motivação para a invenção das CNNs, que introduzem conectividade local e compartilhamento de parâmetros, mecanismos que explicitamente incorporam o viés indutivo para dados espaciais. Para imagens de sensoriamento remoto, que frequentemente contêm padrões espaciais repetitivos (por exemplo, campos agrícolas, redes urbanas, características lineares como rios e estradas), uma arquitetura que compreenda inerentemente as relações espaciais é de suma importância.

#### Explosão de parâmetros com imagens grandes

A natureza "totalmente conectada" das MLPs leva a um número massivo de parâmetros, especialmente ao lidar com imagens de alta resolução. Cada neurônio em uma determinada camada é conectado a todos os neurônios da camada anterior. Essa conectividade densa resulta em um rápido aumento no número de pesos à medida que as dimensões da imagem ou a profundidade da rede crescem.

Para ilustrar com um exemplo concreto, considere uma imagem colorida realista de tamanho 224x224x3 (Altura x Largura x Canais), um tamanho de entrada comum para muitas CNNs modernas (por exemplo, modelos pré-treinados no ImageNet). O número total de pixels (ou características de entrada) para tal imagem é 224×224×3=150.528. Se essa imagem achatada fosse alimentada em uma MLP, e a primeira camada oculta tivesse 1000 neurônios, o número de pesos conectando a camada de entrada a essa primeira camada oculta seria:

**Número de Parâmetros (Pesos) = (Número de Neurônios de Entrada) × (Número de Neurônios na Camada Oculta)**
= 150.528 × 1000 = 150.528.000 pesos

Além disso, cada um dos 1000 neurônios na camada oculta teria seu próprio termo de viés, adicionando mais 1000 parâmetros. Assim, apenas para a primeira conexão, são mais de 150 milhões de parâmetros. Essa contagem de parâmetros é um problema amplamente reconhecido para MLPs com imagens grandes.

Esse número imenso de parâmetros apresenta vários problemas críticos:

1. **Intratabilidade Computacional**: Treinar um modelo com tantos parâmetros exige recursos computacionais (memória de CPU/GPU e poder de processamento) e tempo enormes. Cada parâmetro precisa ser atualizado durante a retropropagação.

2. **Requisitos de Dados**: Um modelo com essa capacidade exigiria um conjunto de dados astronomicamente grande para aprender padrões significativos e generalizar bem. Um princípio geral sugere que uma rede neural precisa de significativamente mais pontos de dados do que parâmetros para treinar eficazmente e evitar o sobreajuste.

3. **Sobreajuste (Overfitting)**: Com um número tão vasto de parâmetros em relação aos dados de treinamento disponíveis, uma MLP é altamente propensa ao sobreajuste. O sobreajuste ocorre quando o modelo aprende os dados de treinamento tão bem que memoriza ruídos e exemplos específicos, em vez de generalizar os padrões subjacentes. Consequentemente, seu desempenho em dados novos e não vistos (conjuntos de validação ou teste) será ruim.

A literatura indica que "o processo de aprendizado desse tipo de rede geralmente requer grandes volumes de dados, comumente imagens de alta resolução, e o ajuste de um grande número de parâmetros. A falta de controle sobre o processo de aprendizado do modelo pode levar a vários problemas. Um deles é o sobreajuste, que leva a rede a uma situação em que perde a generalidade, fazendo previsões incorretas na presença de novos dados".

A explosão de parâmetros é uma consequência direta do design da MLP, onde cada característica de entrada se conecta a cada neurônio na próxima camada. Para imagens, a dimensionalidade de entrada (pixels) é inerentemente muito alta. Essa alta dimensionalidade, combinada com a conectividade total, leva a um crescimento exponencial nos parâmetros. Essa é uma manifestação da "maldição da dimensionalidade", onde o volume do espaço de entrada cresce tão rapidamente que qualquer ponto de dado se torna esparso, dificultando a generalização do modelo sem uma quantidade imensa de dados. Esse problema é particularmente agudo no sensoriamento remoto, onde as imagens podem ter resoluções muito altas (por exemplo, imagens de satélite de gigapixels) e frequentemente são multiespectrais (muitos canais além do RGB). Uma MLP seria simplesmente impraticável para tais dados. Essa limitação enfatiza a necessidade de arquiteturas que possam lidar com entradas de alta dimensionalidade de forma eficiente, impondo restrições estruturais, o que as CNNs alcançam através da conectividade local e do compartilhamento de parâmetros.

A grande quantidade de parâmetros confere à MLP uma imensa "capacidade" para aprender funções complexas. No entanto, se essa capacidade exceder em muito a complexidade dos padrões subjacentes nos dados e a quantidade de dados de treinamento disponíveis, o modelo recorrerá à memorização de exemplos de treinamento individuais, incluindo ruídos. Essa é a essência do sobreajuste. A cadeia de eventos é: Alta dimensionalidade da entrada (imagens) -> Achatamento -> Alta contagem de parâmetros em MLPs -> Alta capacidade do modelo -> Dados insuficientes para restringir todos os parâmetros -> Memorização dos dados de treinamento (sobreajuste) -> Baixa generalização para novos dados. Essa observação destaca um compromisso fundamental no design de redes neurais: capacidade do modelo versus generalização. Embora mais parâmetros permitam o aprendizado de funções mais complexas, eles também exigem mais dados e regularização cuidadosa para evitar o sobreajuste. As CNNs abordam isso reduzindo o número efetivo de parâmetros independentes através do compartilhamento de pesos, tornando-as mais adequadas para dados de imagem, mesmo com amostras rotuladas limitadas, um desafio comum no sensoriamento remoto.

### Operação de Convolução Aprendida

No Módulo 2, foram exploradas técnicas clássicas de processamento de imagens, incluindo a aplicação de filtros (ou kernels) predefinidos para extrair características específicas de imagens. Foram abordados filtros projetados manualmente, como:

- **Filtros de Sobel**: Utilizados para detecção de bordas, aproximando o gradiente da intensidade da imagem nas direções horizontal e vertical.
- **Filtros Gaussianos**: Aplicados para suavização de imagens e redução de ruído, realizando uma média ponderada das vizinhanças de pixels.
- **Filtros Laplacianos**: Empregados para detectar regiões de mudança rápida de intensidade, frequentemente realçando bordas ou transições nítidas, com base na segunda derivada.

Esses filtros clássicos são caracterizados por seus valores de peso fixos e projetados manualmente. Por exemplo, um filtro Sobel comum para bordas verticais pode ser representado pela seguinte matriz:

```
-1  -2  -1
 0   0   0
 1   2   1
```

A inovação revolucionária das Redes Neurais Convolucionais (CNNs) reside em transformar esse processo manual em um processo de aprendizado automático. Em vez de o designer humano especificar os valores exatos para esses filtros, a rede neural aprende os valores ideais do filtro diretamente dos dados durante o treinamento.

Uma camada convolucional em uma CNN é essencialmente uma coleção desses "filtros aprendidos" (também chamados de kernels). Cada filtro é uma pequena matriz de parâmetros que podem ser ajustados. Durante o passo forward, esse filtro desliza pela imagem de entrada (ou mapa de características de uma camada anterior), realizando um produto escalar (multiplicação elemento a elemento e soma) em cada posição para produzir um único valor em um mapa de características de saída.

A operação matemática para uma convolução 2D é definida como:

**S(i,j) = Σ Σ I(i+m,j+n) × K(m,n)**

Onde:
- **I(i+m,j+n)** é o valor do pixel na posição (i+m,j+n) na imagem de entrada (ou mapa de características).
- **K(m,n)** é o peso do kernel (filtro) na posição (m,n).
- **S(i,j)** é o mapa de características de saída na posição (i,j).
- **M×N** é a dimensão do kernel.

Durante a fase de retropropagação do treinamento, os pesos K(m,n) desses filtros são ajustados com base no sinal de erro, permitindo que a rede aprenda quais padrões são mais relevantes para a tarefa específica (por exemplo, classificação de imagens, detecção de objetos). Isso significa que, através do treinamento, alguns filtros na primeira camada convolucional podem naturalmente convergir para valores que se assemelham a detectores de bordas (semelhantes aos filtros Sobel), enquanto outros podem aprender a detectar texturas, cantos ou manchas de cor específicas. Camadas mais profundas então combinam essas características mais simples em padrões mais complexos e abstratos.

A transição de filtros projetados manualmente para filtros aprendidos é a essência do "deep learning" para a visão computacional. Em vez de um pipeline de várias etapas onde as características são extraídas manualmente e depois alimentadas a um classificador, as CNNs integram a extração de características e a classificação em um único sistema treinável de ponta a ponta. Isso permite que a rede otimize o processo de extração de características especificamente para a tarefa final, frequentemente resultando em desempenho superior. A natureza hierárquica significa que as camadas iniciais aprendem características genéricas de baixo nível (como bordas), enquanto as camadas mais profundas as combinam em características de alto nível mais abstratas, relevantes para objetos ou cenas complexas. Essa capacidade de aprendizado de ponta a ponta é fundamental para o sensoriamento remoto. Imagens de satélite frequentemente contêm padrões complexos e sutis que são difíceis de capturar com características projetadas manualmente. As CNNs podem descobrir automaticamente características ótimas para tarefas como classificação de cobertura do solo, detecção de mudanças ou reconhecimento de objetos em ambientes diversos, adaptando-se a variações de iluminação, tipos de sensores e regiões geográficas.

### Conceitos fundamentais: padding, stride, receptive field

Para controlar as dimensões espaciais dos mapas de características e a forma como os filtros interagem com a entrada, três conceitos fundamentais são cruciais nas camadas convolucionais: padding, stride e receptive field.

#### 1. Padding (Preenchimento):

**Conceito**: Padding envolve adicionar pixels extras, tipicamente com valores zero (zero-padding), ao redor da borda da imagem de entrada ou do mapa de características antes de aplicar a convolução.

**Propósito**:
- **Preservação das Dimensões Espaciais**: Sem padding, cada operação de convolução tipicamente reduz as dimensões espaciais (altura e largura) do mapa de características. Após várias camadas, o mapa de características poderia encolher significativamente, perdendo informações, especialmente nas fronteiras. O padding ajuda a manter as dimensões de entrada originais, ou pelo menos a controlar a taxa de redução.
- **Garantia do Uso da Informação de Borda**: Pixels nas bordas de uma imagem são "vistos" pelo filtro menos vezes do que os pixels no centro. O padding garante que os pixels de borda contribuam igualmente para a saída, evitando a perda de informação das fronteiras da imagem.

**Fórmula para o Tamanho da Saída com Padding**:

Para uma entrada de tamanho NH×NW, um kernel de tamanho KH×KW, e padding PH (linhas) e PW (colunas), as dimensões de saída (OH,OW) são:

**OH = NH - KH + 2PH + 1**

Comumente, o padding simétrico é usado (PH=PW=P). Para o padding "same", P=⌊(K-1)/2⌋ para tamanhos de kernel ímpares K, garantindo que o tamanho da saída seja o mesmo que o tamanho da entrada quando o stride é 1.

#### 2. Stride (Passo):

**Conceito**: Stride define o número de pixels que o filtro de convolução se desloca pela entrada a cada passo. Um stride de 1 significa que o filtro se move um pixel por vez. Um stride de 2 significa que ele pula um pixel a cada passo.

**Propósito**:
- **Redução de Dimensionalidade**: Um stride maior que 1 efetivamente subamostra o mapa de características, reduzindo suas dimensões espaciais. Isso diminui o custo computacional e o número de parâmetros em camadas subsequentes.
- **Eficiência Computacional**: Ao pular locais, o stride reduz o número de operações de convolução realizadas.

**Fórmula para o Tamanho da Saída com Stride e Padding**:

Para uma entrada de tamanho NH×NW, um kernel de tamanho KH×KW, padding PH e PW, e strides SH (vertical) e SW (horizontal), as dimensões de saída (OH,OW) são:

**OH = ⌊(NH - KH + 2PH)/SH⌋ + 1**
**OW = ⌊(NW - KW + 2PW)/SW⌋ + 1**

A função ⌊⋅⌋ (piso) indica que qualquer resultado fracionário é arredondado para baixo, pois as dimensões devem ser inteiros.

#### 3. Receptive Field (Campo Receptivo):

**Conceito**: O campo receptivo de um neurônio em uma camada convolucional é a região na imagem de entrada original que influencia a ativação desse neurônio. É a área da entrada "vista" por uma característica de saída específica.

**Crescimento com a Profundidade**: Um aspecto crucial das CNNs é como o campo receptivo cresce com a profundidade da rede.

- **Camadas Iniciais**: Neurônios na primeira camada convolucional têm um campo receptivo igual ao tamanho do seu filtro (por exemplo, um filtro 3x3 significa um campo receptivo 3x3 na entrada). Essas camadas detectam características pequenas e locais.
- **Camadas Mais Profundas**: À medida que mais camadas convolucionais e de pooling são empilhadas, o campo receptivo dos neurônios em camadas mais profundas se expande. Um neurônio em uma terceira camada convolucional, por exemplo, é influenciado por uma região muito maior da imagem de entrada original do que um neurônio na primeira camada. Isso ocorre porque sua entrada vem de um mapa de características, onde cada pixel já representa um resumo de uma região local da camada anterior.

**Detecção Hierárquica de Características**: Esse crescimento do campo receptivo permite a detecção hierárquica de características. As camadas iniciais detectam características simples e de granularidade fina (por exemplo, bordas, cantos). As camadas intermediárias as combinam em padrões mais complexos (por exemplo, texturas, partes de objetos). As camadas mais profundas, com seus grandes campos receptivos, podem reconhecer objetos inteiros ou elementos de cena complexos, integrando informações de amplas regiões da entrada.

**Cálculo**:

O tamanho do campo receptivo (RFL) na camada L pode ser calculado recursivamente da camada mais profunda de volta à entrada:

**RFl = RFl+1 + (Kl - 1) × ∏(Si)**

onde RFl é o tamanho do campo receptivo da camada atual, Sl é o stride da camada atual e Kl é o tamanho do kernel da camada atual. O campo receptivo da última camada é 1.

Padding e stride não são apenas parâmetros arbitrários; são escolhas de design críticas que influenciam diretamente as dimensões espaciais dos mapas de características em toda a rede. O padding ajuda a preservar informações, especialmente nas fronteiras da imagem, que de outra forma poderiam ser perdidas devido a convoluções sucessivas. O stride, por outro lado, é um mecanismo para redução controlada de dimensionalidade. A interação entre esses dois parâmetros permite que os arquitetos equilibrem o desejo de reter detalhes espaciais finos com a necessidade de reduzir a carga computacional e criar representações mais abstratas em camadas mais profundas. Este é um compromisso de engenharia fundamental no design de CNNs. No sensoriamento remoto, as imagens podem ser muito grandes, e características específicas (por exemplo, pequenos edifícios, estradas estreitas) podem ser críticas. A seleção cuidadosa de padding e stride é essencial para garantir que esses detalhes finos não sejam perdidos muito cedo na rede, ao mesmo tempo em que permite o processamento eficiente de grandes imagens de entrada. Isso é particularmente relevante para tarefas como mapeamento de cobertura do solo de alta resolução ou detecção de objetos pequenos.

O crescimento do campo receptivo com a profundidade da rede é uma consequência direta do empilhamento de operações locais. Isso não é apenas uma curiosidade matemática; é o mecanismo pelo qual as CNNs constroem uma compreensão hierárquica da informação visual. Um campo receptivo pequeno significa que um neurônio "vê" apenas um pequeno pedaço, ideal para detectar padrões simples como bordas. Um campo receptivo grande significa que um neurônio integra informações de uma ampla área, permitindo-lhe detectar objetos complexos ou até mesmo cenas inteiras. Assim, o tamanho do campo receptivo se correlaciona implicitamente com a complexidade e a abstração das características aprendidas por uma determinada camada. Para o sensoriamento remoto, entender o campo receptivo é vital. Por exemplo, para classificar um grande campo agrícola, um grande campo receptivo é necessário para capturar a textura e a forma geral. Para detectar um pequeno veículo, um campo receptivo menor e mais localizado pode ser suficiente nas camadas iniciais, que então alimentam campos receptivos maiores que reconhecem o objeto no contexto. Esse conceito orienta o design da profundidade da rede e dos tamanhos de kernel para diferentes tarefas de sensoriamento remoto.

## Arquitetura CNN Básica

A eficácia das CNNs deriva de sua arquitetura modular, construída principalmente a partir de três tipos distintos de camadas, cada uma com um propósito específico no processo hierárquico de extração de características e classificação.

### Camadas convolucionais, pooling e fully connected

#### 1. Camadas Convolucionais (Convolutional Layers):

**Função**: São os blocos de construção centrais de uma CNN, responsáveis por extrair características espaciais dos dados de entrada. Elas aplicam um conjunto de filtros (kernels) aprendíveis sobre a entrada.

**Características**:
- **Conectividade Local**: Cada neurônio em uma camada convolucional é conectado apenas a uma pequena região localizada (campo receptivo) da camada anterior. Isso contrasta com as MLPs, onde cada neurônio é conectado a cada entrada.
- **Compartilhamento de Parâmetros (Weight Sharing)**: O mesmo filtro (conjunto de pesos) é aplicado em toda a dimensão espacial da entrada. Isso significa que se um filtro aprende a detectar uma borda vertical em uma parte da imagem, ele pode detectar a mesma borda vertical em qualquer outro lugar da imagem. Isso reduz significativamente o número de parâmetros únicos, tornando a rede mais eficiente e robusta a translações (equivariância translacional).
- **Múltiplos Filtros**: Uma camada convolucional tipicamente consiste em múltiplos filtros, cada um aprendendo a detectar uma característica diferente (por exemplo, diferentes orientações de bordas, texturas, manchas de cor). Cada filtro produz um "mapa de características" separado.
- **Funções de Ativação**: Após a operação de convolução, uma função de ativação não linear (por exemplo, ReLU - Unidade Linear Retificada) é aplicada elemento a elemento ao mapa de características. Isso introduz não linearidade, permitindo que a rede aprenda relações complexas e não lineares nos dados.

#### 2. Camadas de Pooling (Pooling Layers):

**Função**: As camadas de pooling realizam o subamostragem dos mapas de características, reduzindo suas dimensões espaciais (altura e largura) enquanto retêm as informações mais importantes. Elas tipicamente seguem as camadas convolucionais.

**Características**:
- **Redução de Dimensionalidade**: Reduz o número de parâmetros e a complexidade computacional em camadas subsequentes, tornando a rede mais eficiente.
- **Invariância Translacional**: Introduz um grau de invariância translacional local. Por exemplo, no max pooling, se uma característica se desloca ligeiramente dentro da janela de pooling, o valor máximo de saída pode permanecer o mesmo, tornando a rede mais robusta a pequenas mudanças ou distorções na entrada.
- **Sem Parâmetros Aprendíveis**: As camadas de pooling não possuem pesos ou vieses aprendíveis. Elas aplicam uma função de agregação fixa.

**Tipos Comuns**:
- **Max Pooling**: Seleciona o valor máximo dentro de uma janela de pooling definida (por exemplo, 2x2 ou 3x3). Este é o tipo mais comum e tende a capturar as características mais salientes.
- **Average Pooling**: Calcula o valor médio dentro da janela de pooling. Menos comum que o max pooling, mas pode ser utilizado.

#### 3. Camadas Totalmente Conectadas (Fully Connected - FC Layers):

**Função**: Após várias camadas convolucionais e de pooling terem extraído características hierárquicas, as camadas FC são tipicamente usadas no final da rede para raciocínio de alto nível e classificação.

**Características**:
- **Achatamento**: Antes de se conectar às camadas FC, os mapas de características multidimensionais da última camada de pooling são achatados em um vetor 1D (semelhante a como as MLPs processam imagens, mas agora a entrada é uma rica representação de características, não pixels brutos).
- **Conectividade Densa**: Cada neurônio em uma camada FC é conectado a cada neurônio na camada anterior, assim como em uma MLP tradicional.
- **Classificação**: A camada FC final geralmente tem um número de neurônios igual ao número de classes de saída e usa uma função de ativação como Softmax (para classificação multiclasse) para produzir probabilidades de classe.

### Padrão Arquitetural Típico:

Um padrão arquitetural comum para CNNs é uma sequência de camadas convolucionais e de pooling, seguida por uma ou mais camadas totalmente conectadas:

**ENTRADA → CONV → POOL → CONV → POOL → FC → FC → SAÍDA**

Esse padrão permite que a rede extraia progressivamente características mais abstratas enquanto reduz as dimensões espaciais, culminando em uma decisão de classificação baseada nessas características aprendidas.

A organização sequencial de camadas convolucionais, de pooling e totalmente conectadas cria um poderoso extrator de características hierárquico. As camadas convolucionais aprendem quais características estão presentes (por exemplo, bordas, texturas), enquanto as camadas de pooling aprendem onde essas características estão aproximadamente localizadas (invariância translacional) e reduzem o volume de dados. As camadas FC então utilizam essas características abstratas de alto nível para a classificação final. A conectividade local e o compartilhamento de pesos nas camadas convolucionais atuam como uma forma de regularização embutida, reduzindo significativamente o número total de parâmetros em comparação com uma MLP e, assim, mitigando o sobreajuste, mesmo antes que técnicas de regularização explícitas como o Dropout sejam aplicadas. Essa regularização inerente e o aprendizado hierárquico tornam as CNNs excepcionalmente adequadas para dados de imagem complexos, como imagens de sensoriamento remoto. Elas podem aprender características robustas que são invariantes a pequenas mudanças ou distorções, que são comuns em imagens de satélite ou aéreas do mundo real (por exemplo, pequenos desalinhamentos, variações no ângulo de visão). Essa é uma razão fundamental para seu sucesso na classificação de cobertura do solo, onde o mesmo tipo de cobertura do solo (por exemplo, floresta) pode aparecer com pequenas variações em diferentes partes de uma imagem ou em diferentes momentos de aquisição.

### Feature maps e hierarquia de características

Cada filtro em uma camada convolucional produz um "mapa de características" (também conhecido como mapa de ativação ou característica convolucionada). Um mapa de características é uma matriz 2D que destaca onde uma característica aprendida específica (correspondente a esse filtro) está presente na imagem de entrada ou no mapa de características da camada anterior. Valores altos em um mapa de características indicam uma forte presença da característica detectada naquela localização espacial, enquanto valores baixos indicam sua ausência.

#### Hierarquia de Características:

O conceito de uma hierarquia de características é central para entender por que as CNNs são tão poderosas para tarefas de visão. À medida que os dados fluem através de camadas sucessivas de uma CNN profunda:

**Camadas Iniciais (Lower Layers)**: Essas camadas tipicamente aprendem e detectam características muito simples e de baixo nível, semelhantes aos filtros clássicos discutidos. Exemplos incluem:
- Bordas: Linhas horizontais, verticais, diagonais.
- Cantos: Interseções de bordas.
- Manchas de Cor: Regiões de cor uniforme.
- Esses são análogos aos blocos de construção básicos da informação visual.

**Camadas Intermediárias (Intermediate Layers)**: Com base nas características simples das camadas anteriores, essas camadas aprendem a combiná-las em padrões mais complexos, como:
- Texturas: Padrões visuais repetitivos (por exemplo, grama, tijolo).
- Formas Simples: Círculos, quadrados, triângulos ou partes de objetos (por exemplo, uma roda, um olho).
- Essas características são mais abstratas e semanticamente significativas do que as bordas brutas.

**Camadas Profundas (Deep Layers)**: As camadas mais profundas, com seus maiores campos receptivos, sintetizam as características das camadas intermediárias para reconhecer padrões altamente abstratos e complexos, como:
- Objetos Completos: Carros inteiros, rostos, edifícios, árvores.
- Cenas: Reconhecimento do contexto geral de uma imagem (por exemplo, uma floresta, uma área urbana, um corpo d'água).
- Essa representação hierárquica permite que a rede passe de dados de pixels brutos para uma compreensão rica e semântica do conteúdo da imagem.

Essa capacidade de aprender e representar automaticamente características em múltiplos níveis de abstração é o fator chave que torna as CNNs excepcionalmente eficazes para várias tarefas de visão computacional, incluindo classificação, detecção de objetos e segmentação.

Embora as redes neurais profundas sejam frequentemente consideradas "caixas pretas", os mapas de características oferecem um grau de interpretabilidade sobre o que a rede está aprendendo em cada estágio. Ao visualizar os mapas de características, é possível observar como os dados de pixel brutos são progressivamente transformados em representações mais abstratas e semanticamente significativas. Isso fornece evidências empíricas de que a rede está, de fato, aprendendo características hierárquicas, começando por primitivas visuais básicas e construindo até partes de objetos complexos e objetos inteiros. Essa capacidade de visualização é crucial para depurar, compreender e confiar nos modelos de deep learning. Para o sensoriamento remoto, onde especialistas de domínio frequentemente dependem de pistas visuais específicas (por exemplo, características lineares para estradas, padrões texturais para culturas, assinaturas espectrais para corpos d'água), a visualização de mapas de características pode ajudar a preencher a lacuna entre a interpretação humana e o aprendizado de máquina. Isso permite que pesquisadores confirmem se a rede está "vendo" os mesmos padrões que os analistas humanos usam, ou se está descobrindo características novas e igualmente eficazes. Isso pode levar a novas descobertas sobre as características dos dados de sensoriamento remoto.

## 5.2 LeNet no MNIST: Primeira CNN Prática

### Implementação LeNet-5

#### Arquitetura detalhada e motivação histórica

LeNet-5 é uma arquitetura pioneira de Rede Neural Convolucional proposta por Yann LeCun, Leon Bottou, Yoshua Bengio e Patrick Haffner em 1998. Foi especificamente projetada para reconhecimento de caracteres manuscritos e impressos, notavelmente utilizada para leitura de cheques em caixas eletrônicos. LeNet-5 é historicamente significativa como uma das primeiras demonstrações bem-sucedidas de CNNs para aplicações práticas e serve como uma excelente arquitetura fundamental para a compreensão dos princípios das CNNs.

#### Motivação e Escolhas de Design:

LeCun e sua equipe projetaram a LeNet-5 para abordar os desafios do reconhecimento de dígitos manuscritos, que envolvem variabilidade significativa em estilos de traço, espessura e inclinação. A arquitetura incorpora princípios chave das CNNs para alcançar robustez a essas variações:

- **Extração Hierárquica de Características**: A rede usa múltiplas camadas de convolução e pooling para extrair progressivamente características do simples ao complexo.
- **Campos Receptivos Locais**: Cada neurônio se conecta apenas a uma região local da camada anterior, imitando o córtex visual.
- **Pesos Compartilhados**: Os filtros são compartilhados em toda a imagem, permitindo a detecção da mesma característica independentemente de sua posição (equivariância translacional).
- **Subamostragem (Pooling)**: Usada para reduzir a dimensionalidade e introduzir um grau de invariância a pequenas distorções.

#### Arquitetura Detalhada da LeNet-5:

A LeNet-5 consiste em sete camadas, excluindo a entrada, com duas camadas convolucionais, duas camadas de pooling médio, uma camada convolucional atuando como camada totalmente conectada, e duas camadas totalmente conectadas. Ela tipicamente espera uma imagem em tons de cinza de 32x32 como entrada. As imagens MNIST são de 28x28, então o padding é geralmente aplicado para corresponder ao requisito de entrada da LeNet-5.

1. **Camada de Entrada**: Imagem em tons de cinza de 32x32x1.

2. **C1 (Camada Convolucional)**:
   - Filtros: 6 filtros de tamanho 5x5.
   - Stride: 1.
   - Saída: 6 mapas de características de 28x28x6.
   - Ativação: Tanh (LeNet-5 original, embora implementações modernas frequentemente usem ReLU).

3. **S2 (Camada de Pooling Médio/Subamostragem)**:
   - Janela de Pooling: 2x2.
   - Stride: 2 (não sobreposto).
   - Operação: Pooling médio.
   - Saída: 6 mapas de características de 14x14x6.

4. **C3 (Camada Convolucional)**:
   - Filtros: 16 filtros de tamanho 5x5.
   - Stride: 1.
   - Saída: 16 mapas de características de 10x10x16.
   - **Nota sobre Conectividade**: Na LeNet-5 original, nem todos os 16 mapas de características em C3 eram conectados a todos os 6 mapas de características de S2. Em vez disso, um esquema de conexão esparsa específico foi usado (por exemplo, 10 dos 16 mapas conectados a 3 ou 4 mapas anteriores, outros a 6). Isso foi feito para quebrar a simetria e limitar os parâmetros, uma forma de regularização. Para simplificar em implementações modernas, a conectividade total é frequentemente assumida, levando a uma contagem de parâmetros mais alta para esta camada.
   - Ativação: Tanh.

5. **S4 (Camada de Pooling Médio/Subamostragem)**:
   - Janela de Pooling: 2x2.
   - Stride: 2.
   - Operação: Pooling médio.
   - Saída: 16 mapas de características de 5x5x16.

6. **C5 (Camada Convolucional/Totalmente Conectada)**:
   - Filtros: 120 filtros de tamanho 5x5.
   - Saída: 120 mapas de características de 1x1x120. Esta camada é frequentemente descrita como uma camada convolucional onde o tamanho do kernel corresponde ao tamanho do mapa de características de entrada, resultando em uma saída 1x1 para cada filtro. Efetivamente, ela atua como uma camada totalmente conectada aos 400 nós da camada S4 anterior (5x5x16 = 400).
   - Ativação: Tanh.

7. **F6 (Camada Totalmente Conectada)**:
   - Neurônios: 84 neurônios.
   - Entrada: 120 características de C5.
   - Ativação: Tanh.

8. **Camada de Saída**:
   - Neurônios: 10 neurônios (um para cada dígito de 0 a 9).
   - Entrada: 84 características de F6.
   - Ativação: Softmax (para produzir probabilidades para cada classe).

Apesar de ter sido proposta em 1998, a LeNet-5 incorpora quase todos os princípios fundamentais que sustentam as arquiteturas de CNN modernas: camadas convolucionais e de pooling alternadas, conectividade local, compartilhamento de pesos e extração hierárquica de características. Suas escolhas de design, como o uso de pooling médio e ativações Tanh, refletem o estado da arte da época, mas o padrão arquitetural central permanece altamente relevante. A compreensão da LeNet-5 oferece um modelo histórico e conceitual para arquiteturas mais complexas como AlexNet, VGG e ResNet, que essencialmente escalam essas ideias fundamentais. Essa perspectiva histórica é crucial para estudantes de mestrado e doutorado, pois demonstra que as ideias centrais do deep learning para visão não são inteiramente novas, mas evoluíram a partir de um trabalho fundamental robusto. Também destaca a natureza iterativa da pesquisa, onde melhorias incrementais (por exemplo, ReLU sobre Tanh, Max Pooling sobre Average Pooling) podem levar a ganhos significativos de desempenho. Para o sensoriamento remoto, a compreensão dessa evolução ajuda a apreciar por que certas escolhas arquiteturais são feitas nos modelos de última geração para imagens de satélite.

### Implementação guiada: LeNet from scratch

A implementação da arquitetura LeNet-5 usando o módulo torch.nn do PyTorch é um exercício prático que solidifica a compreensão das camadas convolucionais, de pooling e totalmente conectadas.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a arquitetura LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # C1: Camada convolucional (entrada: 1 canal, saída: 6 canais, kernel: 5x5)
        # Tamanho da saída: (32 - 5 + 1) = 28. Então, 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # S2: Camada de Average Pooling (kernel: 2x2, stride: 2)
        # Tamanho da saída: (28 / 2) = 14. Então, 14x14x6
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C3: Camada convolucional (entrada: 6 canais, saída: 16 canais, kernel: 5x5)
        # Tamanho da saída: (14 - 5 + 1) = 10. Então, 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # S4: Camada de Average Pooling (kernel: 2x2, stride: 2)
        # Tamanho da saída: (10 / 2) = 5. Então, 5x5x16
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: Camada convolucional atuando como FC (entrada: 16 canais, saída: 120 canais, kernel: 5x5)
        # Tamanho da saída: (5 - 5 + 1) = 1. Então, 1x1x120. Isso achata para 120 características.
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        # F6: Camada Totalmente Conectada (entrada: 120 características, saída: 84 características)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        
        # Camada de Saída: Camada Totalmente Conectada (entrada: 84 características, saída: num_classes)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        # Forma da entrada: (Batch_size, 1, 32, 32)
        
        # C1: Aplica convolução, depois ativação Tanh (LeNet-5 original usava Tanh)
        x = F.tanh(self.conv1(x))  # Saída: (Batch_size, 6, 28, 28)
        
        # S2: Aplica pooling médio
        x = self.avgpool1(x)  # Saída: (Batch_size, 6, 14, 14)
        
        # C3: Aplica convolução, depois ativação Tanh
        x = F.tanh(self.conv2(x))  # Saída: (Batch_size, 16, 10, 10)
        
        # S4: Aplica pooling médio
        x = self.avgpool2(x)  # Saída: (Batch_size, 16, 5, 5)
        
        # C5: Aplica convolução, depois ativação Tanh. A saída desta camada é 1x1x120.
        x = F.tanh(self.conv3(x))  # Saída: (Batch_size, 120, 1, 1)
        
        # Achata a saída para as camadas totalmente conectadas
        # x.view(-1, 120) ou torch.flatten(x, 1)
        x = torch.flatten(x, 1)  # Saída: (Batch_size, 120)
        
        # F6: Aplica camada totalmente conectada, depois ativação Tanh
        x = F.tanh(self.fc1(x))  # Saída: (Batch_size, 84)
        
        # Camada de Saída: Aplica camada totalmente conectada (sem ativação aqui, Softmax será aplicada na função de perda)
        x = self.fc2(x)  # Saída: (Batch_size, num_classes)
        
        return x

# Exemplo de uso:
# model = LeNet5(num_classes=10)
# dummy_input = torch.randn(1, 1, 32, 32)  # Tamanho do batch 1, 1 canal, imagem 32x32
# output = model(dummy_input)
# print(output.shape)
```

#### Explicação dos Parâmetros e Cálculos de Dimensão de Saída de nn.Conv2d, nn.AvgPool2d, nn.Linear no PyTorch:

**1. nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)**:
- **in_channels**: Número de canais na imagem/mapa de características de entrada. Para uma imagem em tons de cinza, in_channels=1. Para uma imagem RGB, in_channels=3. Para camadas convolucionais subsequentes, é o out_channels da camada anterior.
- **out_channels**: Número de filtros (kernels) a serem aplicados. Cada filtro produz um mapa de características de saída. Isso determina a profundidade do volume de saída.
- **kernel_size**: Tamanho do kernel convolucional (por exemplo, 5 para 5x5, ou (5, 5)).
- **stride**: Quantos pixels o kernel desliza por vez. O padrão é 1. Se stride=2, ele se move 2 pixels.
- **padding**: Número de pixels preenchidos com zero adicionados a cada lado da entrada. O padrão é 0.
- **Cálculo da Forma de Saída**: Se as dimensões de entrada forem (N,Cin,Hin,Win), onde N é o tamanho do batch. As dimensões de saída (N,Cout,Hout,Wout) são calculadas como:
  - **Hout = ⌊(Hin−kernel_size+2×padding)/stride⌋+1**
  - **Cout é simplesmente out_channels**.

**2. nn.AvgPool2d(kernel_size, stride=None, padding=0)**:
- **kernel_size**: Tamanho da janela de pooling (por exemplo, 2 para 2x2, ou (2, 2)).
- **stride**: Quantos pixels a janela de pooling desliza por vez. Se None, o padrão é kernel_size (pooling não sobreposto).
- **padding**: Não é comumente usado para pooling, geralmente 0.
- **Cálculo da Forma de Saída**: Mesma fórmula que nn.Conv2d, mas out_channels permanece o mesmo que in_channels.
  - **Hout = ⌊(Hin−kernel_size+2×padding)/stride⌋+1**
  - **Cout = Cin**.

**3. nn.Linear(in_features, out_features, bias=True)**:
- **in_features**: Tamanho de cada amostra de entrada (ou seja, a dimensão achatada da camada anterior).
- **out_features**: Tamanho de cada amostra de saída (número de neurônios nesta camada).
- **bias**: Se True, um viés aditivo é aprendido. O padrão é True.
- **Cálculo da Forma de Saída**: Se a forma de entrada for (N,in_features), onde N é o tamanho do batch. A forma de saída será (N,out_features). A operação é y=xW^T+b, onde W é a matriz de pesos de forma (out_features, in_features) e b é o vetor de viés de forma (out_features).

A implementação da LeNet-5 "do zero" usando o nn.Module do PyTorch e suas camadas pré-construídas (nn.Conv2d, nn.AvgPool2d, nn.Linear) demonstra como frameworks de deep learning de alto nível abstraem as complexas operações matemáticas de baixo nível (como loops explícitos para convolução ou multiplicações de matrizes para camadas totalmente conectadas). Essa abstração permite que pesquisadores e desenvolvedores se concentrem no design arquitetural e na iteração experimental, em vez de se prenderem à estabilidade numérica ou a implementações eficientes em GPU. O método forward mapeia claramente o fluxo de dados através das camadas definidas. Essa eficiência na implementação é crucial para a prototipagem rápida e a experimentação no sensoriamento remoto, onde novos conjuntos de dados e modelos complexos estão constantemente sendo desenvolvidos. Ela permite que os pesquisadores testem rapidamente diferentes arquiteturas de CNN para tipos específicos de cobertura do solo ou desafios de detecção de objetos sem ter que reimplementar blocos de construção fundamentais.

### Adaptação do training loop para CNNs

O loop de treinamento fundamental para CNNs permanece semelhante ao das MLPs (passo forward, cálculo da perda, retropropagação, passo do otimizador). No entanto, o pré-processamento de dados e a forma como as formas de entrada são tratadas através da rede exigem adaptações específicas para dados de imagem.

#### 1. Pré-processamento de Dados para MNIST (Padding de 28x28 para 32x32):

O conjunto de dados original MNIST consiste em imagens em tons de cinza de 28x28 pixels. A LeNet-5, conforme projetada por LeCun, espera imagens de entrada de 32x32. É necessário preencher as imagens MNIST para atender a esse requisito. O módulo torchvision.transforms do PyTorch é ideal para isso.

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define as transformações: Redimensionar (o que inclui padding se o tamanho alvo for maior) e converter para Tensor
# Normalizar: Valores de pixel MNIST são 0-255. Normalizar para [-1, 1] ou [0,1].
# LeNet-5 originalmente usava Tanh, que espera entradas em torno de 0, então [-1, 1] é frequentemente preferido.
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensiona de 28x28 para 32x32
    transforms.ToTensor(),  # Converte PIL Image para Tensor
    transforms.Normalize((0.5,), (0.5,))  # (média=0.5, desvio padrão=0.5 mapeará 0-1 para -1-1)
])

# Carrega o conjunto de dados MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Cria DataLoaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Exemplo de verificação da forma da imagem após a transformação
# for images, labels in train_loader:
#     print(f"Forma da imagem do batch: {images.shape}")  # Esperado: torch.Size([batch_size, 1, 32, 32])
#     break
```

A operação transforms.Resize((32,32)) lida com o preenchimento redimensionando a imagem de 28x28 para 32x32. Para imagens, se o tamanho alvo for maior, ela geralmente preenche com pixels pretos (zero) ao redor da imagem.

#### 2. Processamento em Batch Através da CNN e Rastreamento das Formas dos Tensores:

Durante o passo forward, os tensores fluem pela CNN, e suas formas mudam em cada camada devido a convoluções, pooling e achatamento. Compreender essas transformações de forma é crucial para depurar e projetar redes.

Vamos rastrear as formas dos tensores para um batch de N imagens (por exemplo, batch_size = 64) através do modelo LeNet5:

**Tabela 5.2: Rastreamento de Dimensões de Tensores na LeNet-5 (Forward Pass)**

| Camada | Operação | Forma da Entrada (Batch, C, H, W) | Forma da Saída (Batch, C, H, W) | Parâmetros | Notas |
|--------|----------|----------------------------------|------------------------------|------------|--------|
| Input | Imagem Original | (N, 1, 28, 28) | - | - | Imagem MNIST |
| Pré-processamento | transforms.Resize + ToTensor + Normalize | (N, 1, 28, 28) | (N, 1, 32, 32) | - | Padding para LeNet-5 |
| C1 | nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0) | (N, 1, 32, 32) | (N, 6, 28, 28) | 6×(1×5×5+1) =156 | Aprende 6 características |
| S2 | nn.AvgPool2d(kernel_size=2, stride=2) | (N, 6, 28, 28) | (N, 6, 14, 14) | 0 | Subamostragem por 2x2 |
| C3 | nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) | (N, 6, 14, 14) | (N, 16, 10, 10) | 16×(6×5×5+1) =2416 | Aprende 16 características |
| S4 | nn.AvgPool2d(kernel_size=2, stride=2) | (N, 16, 10, 10) | (N, 16, 5, 5) | 0 | Subamostragem por 2x2 |
| C5 | nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0) | (N, 16, 5, 5) | (N, 120, 1, 1) | 120×(16×5×5+1)=48120 | Atua como FC, mapeia 16x5x5 para 120 características |
| Flatten | torch.flatten(x, 1) | (N, 120, 1, 1) | (N, 120) | 0 | Converte 4D para 2D para camadas FC |
| F6 | nn.Linear(120, 84) | (N, 120) | (N, 84) | 120×84+84=10164 | Primeira camada totalmente conectada |
| Output | nn.Linear(84, 10) | (N, 84) | (N, 10) | 84×10+10=850 | Camada de classificação final |
| Total de Parâmetros Treináveis | | | | ~51.556 | Soma dos parâmetros de todas as camadas |

A necessidade de preencher imagens MNIST de 28x28 para 32x32 para a LeNet-5 é um detalhe sutil, mas crítico. Isso destaca que as arquiteturas de rede frequentemente têm requisitos específicos de tamanho de entrada ditados por seus tamanhos de filtro internos e operações de pooling. Não se trata apenas de fazer as dimensões "encaixarem"; também pode ser uma escolha de design para garantir que certas características (como pontos finais de traços, conforme mencionado) sejam capturadas de forma eficaz. Ignorar tais etapas de pré-processamento pode levar a erros ou desempenho subótimo, enfatizando a importância de compreender as motivações de design originais de uma rede. No sensoriamento remoto, as imagens vêm em várias resoluções e tipos de sensores. Adaptar os dados (por exemplo, redimensionar, normalizar, preencher) para se adequar a modelos pré-treinados específicos ou arquiteturas personalizadas é uma etapa comum e crucial. Essa compreensão prepara os estudantes para desafios práticos onde os dados brutos de sensoriamento remoto raramente correspondem perfeitamente aos requisitos de entrada do modelo, exigindo engenharia de dados cuidadosa.

O rastreamento explícito das formas dos tensores em cada camada (conforme mostrado na Tabela 5.2 e nos comentários do código) é mais do que um mero exercício acadêmico. É uma técnica fundamental de depuração para modelos de deep learning. Dimensões incompatíveis são uma fonte comum de erros (RuntimeError: size mismatch). Além disso, entender como as dimensões evoluem ajuda no projeto de redes mais profundas, permitindo prever o tamanho de saída de uma camada antes de implementá-la e garantindo que a forma de saída final corresponda aos requisitos da tarefa (por exemplo, número de classes). Para modelos complexos de sensoriamento remoto, que podem envolver múltiplas modalidades de entrada (por exemplo, óptico, SAR, LiDAR) ou processamento multi-escala, o gerenciamento das formas dos tensores torna-se ainda mais crítico. Essa habilidade é transferível e essencial para a construção de pipelines de deep learning robustos e escaláveis para dados geoespaciais.

#### Exemplo de Adaptação do Loop de Treinamento:

```python
import torch.optim as optim
from tqdm import tqdm  # Para barra de progresso

# Instancia o modelo, a função de perda e o otimizador
model = LeNet5(num_classes=10)
criterion = nn.CrossEntropyLoss()  # Adequado para classificação multiclasse
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD com momentum como nos artigos originais da LeNet

num_epochs = 10

print("Iniciando o Treinamento da LeNet-5...")
for epoch in range(num_epochs):
    model.train()  # Define o modelo para o modo de treinamento
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}")):
        # Forma das imagens: (batch_size, 1, 32, 32)
        # Forma dos rótulos: (batch_size)
        
        # Passo forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward e otimização
        optimizer.zero_grad()  # Limpa os gradientes do passo anterior
        loss.backward()  # Calcula os gradientes
        optimizer.step()  # Atualiza os pesos
        
        running_loss += loss.item()
    
    print(f"Época [{epoch+1}/{num_epochs}], Perda: {running_loss / len(train_loader):.4f}")
    
    # Fase de avaliação (opcional, mas boa prática para rastrear o desempenho de validação)
    model.eval()  # Define o modelo para o modo de avaliação
    correct = 0
    total = 0
    with torch.no_grad():  # Desabilita o cálculo de gradiente durante a avaliação
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Acurácia no Teste: {accuracy:.2f}%")

print("Treinamento finalizado.")
```

## Análise Comparativa

### MLP vs CNN no mesmo dataset MNIST

Para demonstrar empiricamente as vantagens das CNNs sobre as MLPs, serão treinadas tanto uma MLP simples quanto a CNN LeNet-5 no conjunto de dados MNIST, utilizando configurações de treinamento idênticas (otimizador, taxa de aprendizado, número de épocas). Essa comparação concretiza as limitações teóricas das MLPs para dados de imagem discutidas anteriormente.

#### Esquema de Configuração de Treinamento:

1. **Conjunto de Dados**: MNIST (60.000 imagens de treinamento, 10.000 imagens de teste).

2. **Pré-processamento**:
   - **MLP**: Achatar imagens 28x28 para vetores de 784 elementos. Normalizar valores de pixel (por exemplo, para 0-1 ou -1-1).
   - **CNN (LeNet-5)**: Preencher imagens 28x28 para 32x32. Normalizar valores de pixel (por exemplo, para -1-1).

3. **Modelos**:
   - **MLP**: Uma MLP simples, por exemplo, nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).
   - **CNN**: O modelo LeNet5 implementado.

4. **Parâmetros de Treinamento**:
   - **Função de Perda**: nn.CrossEntropyLoss() para ambos.
   - **Otimizador**: optim.SGD() com taxa de aprendizado consistente (por exemplo, 0.01) e momentum (por exemplo, 0.9) para ambos.
   - **Épocas**: Um número razoável de épocas (por exemplo, 10-20) para permitir a convergência de ambos os modelos.
   - **Tamanho do Batch**: Tamanho do batch consistente (por exemplo, 64 ou 128) para ambos.

5. **Métricas para Comparação**:
   - **Tempo de Treinamento**: Medir o tempo total gasto no treinamento de cada modelo.
   - **Acurácia Final no Teste**: A acurácia de classificação no conjunto de teste não visto.
   - **Contagem de Parâmetros**: Calcular o número total de parâmetros treináveis para cada modelo.
   - **Uso de Memória**: (Opcional, mas esclarecedor) Monitorar o pico de uso de memória da GPU durante o treinamento.

#### Resultados Esperados e Análise:

- **Contagem de Parâmetros**: A MLP provavelmente terá um número de parâmetros maior ou comparável ao de uma CNN simples como a LeNet-5, mesmo para uma imagem pequena como MNIST (por exemplo, uma MLP com entrada de 784, 128 neurônios ocultos e 10 de saída tem aproximadamente 784×128+128×10+128+10≈100.000 parâmetros, enquanto a LeNet-5 tem ~51.556 parâmetros conforme calculado anteriormente). Uma MLP mais complexa superaria rapidamente a CNN.

- **Tempo de Treinamento**: Para contagens de parâmetros semelhantes, as CNNs podem levar um pouco mais de tempo por época devido às operações de convolução, mas sua convergência mais rápida devido ao melhor aprendizado de características frequentemente leva a tempos de treinamento gerais mais curtos para alcançar alta acurácia.

- **Acurácia**: A CNN (LeNet-5) deve alcançar uma acurácia significativamente maior no conjunto de teste MNIST em comparação com a MLP. Por exemplo, MLPs tipicamente alcançam cerca de 95-96% de acurácia no MNIST, enquanto a LeNet-5 pode atingir 98-99%.

- **Uso de Memória**: As CNNs, devido ao compartilhamento de parâmetros e ao pooling, são geralmente mais eficientes em termos de memória do que as MLPs para tarefas de imagem, especialmente à medida que o tamanho da imagem aumenta.

#### Análise de Matrizes de Confusão:

Após o treinamento, gerar matrizes de confusão para ambos os modelos no conjunto de teste. Analisar onde cada modelo comete erros.

- **Erros da MLP**: Podem mostrar erros mais difusos entre diferentes dígitos, indicando uma compreensão mais pobre dos padrões visuais. Pode confundir dígitos que são visualmente semelhantes, mas espacialmente distintos (por exemplo, 4 e 9, 3 e 8).
- **Erros da CNN**: Espera-se que sejam menos numerosos no geral. Os erros podem se concentrar em dígitos altamente ambíguos ou mal escritos, ou aqueles com estilos incomuns aos quais os filtros aprendidos não conseguiram generalizar.

Isso demonstra que a CNN aprendeu características mais robustas e invariantes à translação.

**Tabela 5.1: Comparativo de Desempenho MLP vs. CNN no MNIST**

| Característica | MLP (Exemplo Simples) | LeNet-5 (CNN) |
|----------------|----------------------|---------------|
| Arquitetura | Flatten -> Linear -> ReLU -> Linear | Conv -> Pool -> Conv -> Pool -> Conv (FC) -> FC -> FC |
| Tratamento da Imagem | Achata pixels, ignora relações espaciais | Preserva estrutura espacial, usa filtros aprendidos |
| Contagem de Parâmetros | ~100.000 (exemplo) | ~51.556 (LeNet-5 padrão) |
| Tempo de Treinamento (Exemplo) | Curto a Moderado | Moderado a Longo (por época), mas converge mais rápido para alta acurácia |
| Acurácia Final (MNIST) | ~95-96% | ~98-99% |
| Propensão a Overfitting | Alta para imagens grandes | Menor devido a compartilhamento de pesos e pooling |
| Invariância a Translações | Nenhuma inerente | Inerente (devido a pooling e compartilhamento de pesos) |

A diferença de desempenho entre MLP e CNN no MNIST não é apenas uma observação; é uma validação empírica poderosa da importância dos vieses indutivos arquiteturais. A CNN, ao incorporar conectividade local, compartilhamento de pesos e pooling, tem uma "vantagem" na compreensão de dados de imagem. Ela não precisa aprender do zero que pixels adjacentes estão relacionados ou que uma característica detectada em uma parte da imagem é a mesma em outra. Essa compreensão inerente leva a um aprendizado mais rápido, melhor generalização e maior acurácia com menos parâmetros, especialmente quando comparada a uma MLP que trata cada pixel de forma independente. Isso reforça a ideia de que o melhor modelo para uma tarefa é frequentemente aquele cuja arquitetura se alinha com a estrutura dos dados.

Para o sensoriamento remoto, isso significa que as CNNs não são apenas uma escolha da moda, mas uma ferramenta fundamentalmente mais apropriada para processar imagens geoespaciais devido às correlações espaciais e padrões repetitivos inerentes a esses dados.

### Visualização de feature maps e filtros aprendidos

A visualização dos filtros aprendidos e dos mapas de características que eles produzem é uma maneira poderosa de obter intuição sobre como as CNNs "veem" e processam imagens. Isso ajuda a desmistificar a natureza de "caixa preta" das redes neurais e se conecta à discussão sobre filtros clássicos.

#### 1. Extrair e Visualizar Filtros Aprendidos da Primeira Camada Convolucional:

Os filtros na primeira camada convolucional operam diretamente nos dados de pixels brutos. Espera-se que alguns deles se assemelhem a detectores de bordas básicos, filtros de desfoque ou detectores de cor.

```python
import matplotlib.pyplot as plt
import numpy as np

# Assumindo que 'model' é sua instância LeNet5 treinada
# Acessa os pesos da primeira camada convolucional (conv1)
# Os pesos são tipicamente armazenados como (out_channels, in_channels, kernel_height, kernel_width)
filters = model.conv1.weight.detach().cpu().numpy()

# Normaliza os valores do filtro para visualização (por exemplo, para o intervalo 0-1 ou -1-1)
# Para melhor contraste, geralmente escala para o min/max do filtro
filters_normalized = (filters - filters.min()) / (filters.max() - filters.min())

# Plota os 6 filtros aprendidos da primeira camada
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i, ax in enumerate(axes):
    # Para entrada em tons de cinza, in_channels é 1, então filters[i, 0] fornece o kernel 2D
    ax.imshow(filters_normalized[i, 0], cmap='gray')
    ax.set_title(f'Filtro {i+1}')
    ax.axis('off')

plt.suptitle('Filtros Aprendidos da Camada C1 (LeNet-5)')
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()
```

**Observação Esperada**: Deve-se observar filtros que se parecem com detectores de bordas horizontais, verticais ou diagonais, detectores de blobs ou detectores de textura simples. Isso fornece evidências concretas de que a rede aprendeu automaticamente características semelhantes às que foram projetadas manualmente no processamento de imagens clássico.

#### 2. Mostrar Feature Maps Produzidos por Diferentes Filtros:

É possível passar uma imagem de amostra pela rede (ou apenas pelas primeiras camadas) e visualizar os mapas de características de saída. Cada mapa de características mostra onde seu filtro correspondente "ativou" na imagem.

```python
# Assumindo que 'model' é sua instância LeNet5 treinada
# Assumindo que 'test_loader' é seu DataLoader para o conjunto de teste MNIST
# Obtém uma imagem de amostra
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Seleciona uma imagem do batch
sample_image = images[0:1]  # Mantém a dimensão do batch para a entrada do modelo

# Passa a imagem de amostra pela primeira camada convolucional
with torch.no_grad():
    conv1_output = model.conv1(sample_image)

# Converte para numpy para visualização
conv1_output_np = conv1_output.squeeze(0).detach().cpu().numpy()  # Remove a dimensão do batch

# Plota a imagem original
plt.figure(figsize=(10, 5))
plt.subplot(2, 4, 1)
plt.imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
plt.title(f'Imagem Original (Rótulo: {labels[0].item()})')
plt.axis('off')

# Plota alguns mapas de características de C1
for i in range(min(6, conv1_output_np.shape[0])):  # Plota até 6 mapas de características
    plt.subplot(2, 4, i + 2)
    plt.imshow(conv1_output_np[i], cmap='viridis')  # Usa 'viridis' ou 'gray'
    plt.title(f'Mapa de Características {i+1}')
    plt.axis('off')

plt.suptitle('Mapas de Características da Camada C1 (LeNet-5)')
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()
```

**Observação Esperada**: Cada mapa de características destacará diferentes aspectos da imagem de entrada. Por exemplo, se um filtro aprendeu a detectar bordas verticais, seu mapa de características mostrará ativações brilhantes onde as bordas verticais estão presentes na imagem de entrada. Isso demonstra visualmente o processo de extração de características.

A capacidade de visualizar filtros aprendidos e mapas de características é crucial para a interpretabilidade. Ela transforma o conceito abstrato de "características aprendidas" em padrões visuais concretos. Quando se observa um filtro que se assemelha a um detector de bordas Sobel, isso confirma que a rede, sem programação explícita, descobriu uma operação fundamental de processamento de imagem relevante para a tarefa. Isso ajuda a construir confiança em modelos de deep learning e fornece uma ligação tangível ao conhecimento clássico de processamento de imagens. Para aplicações de alto risco no sensoriamento remoto (por exemplo, avaliação de desastres, monitoramento ambiental, defesa), a interpretabilidade do modelo é frequentemente primordial. Ser capaz de visualizar quais características uma CNN está aprendendo para tipos de cobertura do solo ou classes de objetos pode fornecer informações valiosas para especialistas de domínio, validar o comportamento do modelo e até mesmo informar novas hipóteses sobre as características dos dados subjacentes. Isso move o deep learning de uma ferramenta puramente preditiva para uma mais analítica.

### Preparação conceitual: Por que CNNs são ideais para sensoriamento remoto

As informações obtidas com o experimento LeNet-5 no MNIST se traduzem diretamente nas vantagens das CNNs para aplicações de sensoriamento remoto. Imagens de sensoriamento remoto, sejam de satélites, plataformas aéreas ou drones, compartilham características fundamentais com imagens naturais que tornam as CNNs unicamente adequadas para sua análise.

1. **Estrutura Espacial Intrínseca**: Imagens de sensoriamento remoto são inerentemente espaciais. Elas contêm estradas como características lineares, campos agrícolas com texturas consistentes, áreas urbanas com padrões característicos de grade ou irregulares, e corpos d'água com propriedades espectrais e texturais distintas. Todos esses são padrões espaciais que as CNNs são projetadas para aprender e explorar.

2. **Invariância a Translações**: Objetos ou características de interesse (por exemplo, um edifício específico, um tipo de cultura, um veículo) podem aparecer em qualquer lugar dentro de uma grande imagem de satélite. A propriedade de compartilhamento de pesos das CNNs permite que elas detectem a mesma característica independentemente de sua posição, proporcionando equivariância translacional. Isso é crucial para tarefas como detecção de objetos ou classificação de cobertura do solo em vastas áreas geográficas.

3. **Hierarquia de Características**: Assim como a LeNet-5 aprendeu bordas simples e as combinou em dígitos, as CNNs para sensoriamento remoto podem aprender uma hierarquia de características:
   - **Camadas Iniciais**: Podem detectar elementos básicos como bordas de estradas, limites de campos ou variações espectrais (por exemplo, identificando vegetação versus solo nu).
   - **Camadas Intermediárias**: Podem combinar esses elementos em padrões mais complexos, como formas específicas de edifícios, texturas de diferentes tipos de culturas ou padrões urbanos distintos.
   - **Camadas Profundas**: Podem reconhecer classes inteiras de cobertura do solo (por exemplo, "floresta densa", "área urbana", "corpo d'água") ou objetos complexos (por exemplo, "navio", "aeronave") integrando diversas características de baixo nível. Isso é especialmente poderoso para imagens de sensoriamento remoto de alta resolução, onde detalhes finos contribuem para a compreensão geral da cena.

4. **Processamento de Dados de Alta Dimensionalidade**: Imagens de sensoriamento remoto frequentemente possuem múltiplas bandas espectrais (por exemplo, Sentinel-2 tem 13 bandas) além do RGB visível. As CNNs podem lidar naturalmente com essas entradas multi-canal, estendendo o conceito de in_channels em camadas convolucionais, permitindo-lhes aprender características que combinam informações espectrais e espaciais.

5. **Extração Automática de Características**: O sensoriamento remoto tradicional frequentemente depende de características projetadas manualmente ou índices espectrais. As CNNs automatizam esse processo de engenharia de características, aprendendo as características mais discriminativas diretamente dos dados, o que pode levar a um desempenho superior e reduzir o esforço humano.

Essas capacidades inerentes tornam as CNNs o padrão de fato para uma ampla gama de tarefas de sensoriamento remoto, incluindo:

- **Classificação de Cobertura do Solo e Uso da Terra**: Atribuição de categorias (por exemplo, floresta, urbano, água, agricultura) a pixels ou regiões.
- **Detecção de Mudanças**: Identificação de alterações na cobertura do solo ou em objetos ao longo do tempo.
- **Reconhecimento de Objetos**: Localização e identificação de objetos específicos como veículos, edifícios ou infraestruturas.
- **Segmentação Semântica**: Classificação de cada pixel em uma imagem em uma categoria específica.

A transição de métodos tradicionais de processamento de imagens e aprendizado de máquina para CNNs no sensoriamento remoto representa uma mudança fundamental na forma como os dados geoespaciais são analisados. Historicamente, o sensoriamento remoto dependia fortemente de índices espectrais, medidas de textura e características geométricas definidas por especialistas. As CNNs, por sua própria natureza, automatizam e otimizam esse processo de extração de características, passando de características projetadas por humanos para representações aprendidas e orientadas por dados. Isso não apenas melhora a acurácia, mas também reduz o esforço manual e o conhecimento específico do domínio necessários para a engenharia de características, democratizando a análise avançada. Isso tem implicações profundas para o campo do sensoriamento remoto. Permite o processamento rápido de grandes volumes de dados de alta resolução e multiespectrais, facilitando o monitoramento ambiental em larga escala, a resposta a desastres, o planejamento urbano e a gestão de recursos com precisão e eficiência sem precedentes. A capacidade das CNNs de generalizar entre diferentes regiões geográficas e tipos de sensores, devido ao seu robusto aprendizado de características, as torna um pilar da análise de sensoriamento remoto moderna.

## Conclusões

Este módulo estabeleceu uma base sólida para a compreensão das Redes Neurais Convolucionais (CNNs) e seu papel transformador no processamento de imagens, com um foco particular em suas aplicações no sensoriamento remoto. A análise detalhada das limitações das MLPs para dados de imagem revelou que a perda de informação espacial devido ao achatamento e a explosão de parâmetros em imagens grandes as tornam impraticáveis para tarefas de visão computacional complexas. A ausência de um viés indutivo para a estrutura espacial inerente às imagens e a propensão ao sobreajuste em modelos com alta capacidade são desafios críticos que as MLPs não conseguem superar eficientemente.

Em contraste, a operação de convolução aprendida nas CNNs representa uma mudança de paradigma. Ao permitir que a rede aprenda filtros ideais diretamente dos dados, as CNNs superam a necessidade de engenharia manual de características, como os filtros clássicos de Sobel ou Gaussiano. Conceitos fundamentais como padding e stride permitem o controle preciso das dimensões dos mapas de características, equilibrando a preservação da informação de borda com a redução de dimensionalidade. O campo receptivo, que cresce com a profundidade da rede, é o mecanismo pelo qual as CNNs constroem uma hierarquia de características, passando de padrões simples de baixo nível (bordas) para representações complexas de alto nível (objetos e cenas).

A arquitetura básica das CNNs, composta por camadas convolucionais, de pooling e totalmente conectadas, forma um sistema hierárquico robusto para extração e classificação de características. A conectividade local e o compartilhamento de parâmetros nas camadas convolucionais atuam como uma forma inerente de regularização, tornando as CNNs mais eficientes em termos de parâmetros e menos propensas ao sobreajuste do que as MLPs para dados de imagem. A visualização de mapas de características oferece uma janela para o funcionamento interno dessas redes, demonstrando empiricamente que elas aprendem características significativas e interpretáveis.

A implementação prática da LeNet-5 no conjunto de dados MNIST serviu como uma demonstração empírica da superioridade das CNNs. A comparação direta com uma MLP no mesmo conjunto de dados revelou que as CNNs alcançam acurácias significativamente mais altas com uma contagem de parâmetros mais eficiente, validando a importância dos vieses indutivos arquiteturais. A necessidade de pré-processamento de dados específico, como o padding de imagens MNIST para a entrada da LeNet-5, sublinha a importância da engenharia de dados e do rastreamento de formas de tensores como ferramentas essenciais de depuração e design.

Finalmente, as propriedades intrínsecas das CNNs — sua capacidade de explorar a estrutura espacial, sua invariância translacional, sua habilidade de construir hierarquias de características e sua aptidão para processar dados de alta dimensionalidade e multiespectrais — as tornam ideais para o sensoriamento remoto. As CNNs representam uma mudança fundamental na análise de dados geoespaciais, automatizando a extração de características e permitindo o processamento de grandes volumes de dados com precisão e eficiência sem precedentes para aplicações como classificação de cobertura do solo, detecção de mudanças e reconhecimento de objetos. A compreensão aprofundada desses princípios prepara os estudantes para os desafios e oportunidades que o Deep Learning oferece para o avanço da ciência e das aplicações em sensoriamento remoto.

## Referências Bibliográficas

1. Deep Learning, acessado em agosto 12, 2025, https://deep-learning-s25.vercel.app/slides/lecture10.pdf
2. Mastering the Multi-Layer Perceptron (MLP) for Image Classification ..., acessado em agosto 12, 2025, https://medium.com/eincode/mastering-the-multi-layer-perceptron-mlp-for-image-classification-a0272baf1e29
3. Convolutional neural network - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/Convolutional_neural_network
4. Fully Connected Layer vs Convolutional Layer - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/deep-learning/fully-connected-layer-vs-convolutional-layer/
5. Convolutional Neural Network (CNN) | by Rishabh Singh - Medium, acessado em agosto 12, 2025, https://medium.com/@RobuRishabh/convolutional-neural-network-cnn-part-1-d1c027913b2b
6. Neural Network Parameters - Learn FluCoMa, acessado em agosto 12, 2025, https://learn.flucoma.org/learn/mlp-parameters/
7. 1.17. Neural network models (supervised) - Scikit-learn, acessado em agosto 12, 2025, https://scikit-learn.org/stable/modules/neural_networks_supervised.html
8. A Case Study on Overfitting in Multiclass Classifiers Using ..., acessado em agosto 12, 2025, https://sol.sbc.org.br/index.php/eniac/article/download/9335/9237/
9. ML Practicum: Image Classification | Machine Learning - Google for Developers, acessado em agosto 12, 2025, https://developers.google.com/machine-learning/practica/image-classification/preventing-overfitting
10. What is Overfitting? - Overfitting in Machine Learning Explained - AWS, acessado em agosto 12, 2025, https://aws.amazon.com/what-is/overfitting/
11. Overfitting in Deep Neural Networks & how to prevent it. | Analytics Vidhya - Medium, acessado em agosto 12, 2025, https://medium.com/analytics-vidhya/the-perfect-fit-for-a-dnn-596954c9ea39
12. Sobel Edge Detection vs. Canny Edge Detection in Computer Vision - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/computer-vision/sobel-edge-detection-vs-canny-edge-detection-in-computer-vision/
13. Kernels (Filters) in convolutional neural network - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/deep-learning/kernels-filters-in-convolutional-neural-network/
14. Math at the Heart of CNN: Part 2 - Svitla Systems, acessado em agosto 12, 2025, https://svitla.com/blog/math-at-the-heart-of-cnn/
15. Math Behind Convolutional Neural Networks - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/deep-learning/math-behind-convolutional-neural-networks/
16. Understanding Convolution in Deep Learning - Tim Dettmers, acessado em agosto 12, 2025, https://timdettmers.com/2015/03/26/convolution-deep-learning/
17. Convolutional Neural Networks - Andrew Gibiansky, acessado em agosto 12, 2025, https://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
18. What are Convolutional Neural Networks? | IBM, acessado em agosto 12, 2025, https://www.ibm.com/think/topics/convolutional-neural-networks
19. Feature Map. What does Feature Map mean in CNN… | by Saba ..., acessado em agosto 12, 2025, https://medium.com/@saba99/feature-map-35ba7e6c689e
20. Full article: A survey of remote sensing image classification based on CNNs, acessado em agosto 12, 2025, https://www.tandfonline.com/doi/full/10.1080/20964471.2019.1657720
21. Land use classification of satellite images with convolutional neural networks (CNNs) - International Association for Computer Information Systems, acessado em agosto 12, 2025, https://www.iacis.org/iis/2024/2_iis_2024_267-276.pdf
22. 7.3. Padding and Stride — Dive into Deep Learning 1.0.3 ..., acessado em agosto 12, 2025, https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
23. Understanding 2D Convolutions in PyTorch | by ML and DL ..., acessado em agosto 12, 2025, https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f
24. Convolutional Neural Networks (CNNs / ConvNets) - CS231n Deep Learning for Computer Vision, acessado em agosto 12, 2025, https://cs231n.github.io/convolutional-networks/
25. Python - ShareTechnote, acessado em agosto 12, 2025, https://www.sharetechnote.com/html/Python_PyTorch_nn_conv2D_01.html
26. How to Calculate Receptive Field Size in CNN | Baeldung on ..., acessado em agosto 12, 2025, https://www.baeldung.com/cs/cnn-receptive-field-size
27. Convolutional Neural Networks (LeNet) - dsgiitr/d2l-pytorch - GitHub, acessado em agosto 12, 2025, https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch08_Convolutional_Neural_Networks/Convolutional_Neural_Networks(LeNet).ipynb
28. 7.6. Convolutional Neural Networks (LeNet) - Dive into Deep Learning, acessado em agosto 12, 2025, http://d2l.ai/chapter_convolutional-neural-networks/lenet.html
29. LeNet-5 - A Classic CNN Architecture - DataScienceCentral.com, acessado em agosto 12, 2025, https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/
30. LeNet - Wikipedia, acessado em agosto 12, 2025, https://en.wikipedia.org/wiki/LeNet
31. LeNet-5: A Practical Approach - DebuggerCafe, acessado em agosto 12, 2025, https://debuggercafe.com/lenet-5-a-practical-approach/
32. LeNet and MNIST handwritten digit recognition | by Khuyen Le - Medium, acessado em agosto 12, 2025, https://lekhuyen.medium.com/lenet-and-mnist-handwritten-digit-classification-354f5646c590
33. LeNet-5 from Scratch with PyTorch A Beginner's Guide | DigitalOcean, acessado em agosto 12, 2025, https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python
34. LeNet Pytorch implementation - Kaggle, acessado em agosto 12, 2025, https://www.kaggle.com/code/shravankumar147/lenet-pytorch-implementation
35. MNIST classification using LeNet on Pytorch - Kaggle, acessado em agosto 12, 2025, https://www.kaggle.com/code/yogeshrampariya/mnist-classification-using-lenet-on-pytorch
36. Reasoning about Shapes in PyTorch, acessado em agosto 12, 2025, https://docs.pytorch.org/tutorials/recipes/recipes/reasoning_about_shapes.html
37. python - What is the class definition of nn.Linear in PyTorch? - Stack ..., acessado em agosto 12, 2025, https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch
38. From MLP to CNN. Neural Networks for MNIST Digit ... - Daniel Gustaw, acessado em agosto 12, 2025, https://gustawdaniel.com/posts/en/mlp-cnn-mnist/
39. Visualization of ConvNets in Pytorch - Python - GeeksforGeeks, acessado em agosto 12, 2025, https://www.geeksforgeeks.org/data-visualization/visualization-of-convents-in-pytorch-python/
40. I created a 3D visual explanation of LeNet-5 using Blender and PyTorch - Reddit, acessado em agosto 12, 2025, https://www.reddit.com/r/learnmachinelearning/comments/1kulyqi/i_created_a_3d_visual_explanation_of_lenet5_using/

41. Pytorch implementation of convolutional neural network visualization techniques - GitHub, acessado em agosto 12, 2025, https://github.com/utkuozbulak/pytorch-cnn-visualizations
42. How to visualize CNN filters, and feature maps? #7803 - GitHub, acessado em agosto 12, 2025, https://github.com/Lightning-AI/pytorch-lightning/discussions/7803
43. Land Use and Land Cover Classification Meets Deep Learning: A Review - PMC, acessado em agosto 12, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10649958/
44. Change Detection for Land-Cover Classification Using Deep Learning - ResearchGate, acessado em agosto 12, 2025, https://www.researchgate.net/publication/393688291_Change_Detection_for_Land-Cover_Classification_Using_Deep_Learning
