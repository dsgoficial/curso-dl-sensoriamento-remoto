---
sidebar_position: 2
title: "Arquiteturas Modernas"
description: "Arquiteturas Modernas de Segmentação Semântica com pytorch-segmentation-models"
tags: [pytorch, pytorch lightning, treinamento, deep learning, pspnet, deeplab, fpn]
---

# Arquiteturas Modernas de Segmentação Semântica com pytorch-segmentation-models**

## **1. Introdução: O Contexto da Segmentação Semântica e o Legado da U-Net**

A segmentação semântica, o processo de atribuir um rótulo de classe a cada pixel em uma imagem, é um dos desafios mais complexos da visão computacional. Redes neurais convolucionais (CNNs) tradicionais, ao utilizarem operações de pooling e downsampling, perdem progressivamente a resolução espacial. A U-Net, com seu design de codificador-decodificador e conexões de salto, estabeleceu uma solução elegante para esse problema, tornando-se uma arquitetura seminal e um ponto de partida ideal para qualquer discussão sobre segmentação de alto nível.

Este módulo de curso irá aprofundar-se em três arquiteturas influentes que se basearam na filosofia da U-Net — DeepLab, PSPNet e FPN —, explorando suas inovações e comparando suas abordagens com a da U-Net para resolver o dilema entre a captura de contexto global e a preservação de detalhes espaciais finos. As arquiteturas serão examinadas por meio de sua implementação prática e fluida usando a biblioteca pytorch-segmentation-models (SMP).

## **2. DeepLab: A Convolução Atrous e o Contexto em Múltiplas Escalas**

A arquitetura DeepLab foi desenvolvida para abordar desafios na segmentação de imagens, especificamente a perda de resolução de características causada por repetidas operações de max pooling e a dificuldade em segmentar objetos que aparecem em múltiplas escalas. Sua abordagem representa um avanço filosófico em relação à U-Net, resolvendo proativamente o problema da perda de resolução no próprio *encoder*, em contraste com a abordagem da U-Net que o corrige reativamente no *decoder* por meio das conexões de salto.

### **Inovações Fundamentais: Convolução Atrous e ASPP**

A principal ferramenta introduzida pela DeepLab é a **convolução atrous** (também conhecida como convolução dilatada). Essa operação de convolução insere buracos ("trous" em francês) entre os pontos de um filtro, expandindo efetivamente o seu campo de visão sem aumentar o número de parâmetros ou o custo computacional. Isso permite que a rede capture um contexto mais amplo, preservando uma resolução de características mais densa no *encoder*.

Para lidar com a questão da segmentação de objetos em várias escalas, a DeepLab propôs o **Atrous Spatial Pyramid Pooling (ASPP)**. Inspirado pelo *spatial pyramid pooling*, o ASPP aplica múltiplas convoluções atrous em paralelo, cada uma com uma taxa de dilatação diferente. Essa técnica sonda a camada de entrada com filtros de campos de visão complementares, permitindo que a rede capture informações de objetos e contexto em múltiplas escalas de forma robusta e eficiente.

### **DeepLabV3+ e a Filosofia Híbrida**

O DeepLabV3+ representa uma extensão significativa da arquitetura, que incorpora um módulo *decoder* simples, mas eficaz, para refinar os limites dos objetos. A decisão de adicionar um *decoder* à arquitetura da DeepLab indica que, embora a convolução atrous tenha sido uma solução inovadora para o problema do *downsampling*, o conceito da U-Net de um *decoder* focado em refinar os contornos continua sendo fundamental. Isso cria uma arquitetura híbrida que combina a força do *encoder* da DeepLab em preservar a densidade de características com a capacidade de refino de limites do *decoder* no estilo U-Net.

### **Implementação Prática com pytorch-segmentation-models**

A biblioteca SMP oferece suporte para a arquitetura DeepLabV3Plus. O modelo pode ser instanciado de forma semelhante aos outros modelos da biblioteca, com a mesma flexibilidade para a escolha do *encoder* e a configuração de parâmetros.

```python
import segmentation_models_pytorch as smp
import torch

# Instanciando um modelo DeepLabV3Plus com o encoder ResNet-101
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights="imagenet",
    in_channels=3,
    classes=21,
    activation='softmax'
)

# Criando um tensor de entrada de exemplo
dummy_input = torch.randn(1, 3, 512, 512)
output_mask = model(dummy_input)

print(f"Forma da máscara de saída: {output_mask.shape}")
```

## **3. PSPNet: A Prioridade no Contexto da Cena Global**

A PSPNet (Pyramid Scene Parsing Network) foi criada para superar uma deficiência comum em redes totalmente convolucionais (FCNs), como a U-Net: a incapacidade de capturar o contexto de cena global. A falta de contexto pode levar a classificações errôneas, como a de um "barco" em um rio ser rotulado como um "carro" devido à sua forma, ignorando completamente o cenário circundante. Para resolver essa falha, a PSPNet priorizou a compreensão do cenário como um todo, indo além da captura de características locais.

### **O Coração da Arquitetura: O Módulo Pyramid Pooling (PPM)**

A inovação central da PSPNet é o seu Módulo de *Pyramid Pooling* (PPM). Este módulo foi meticulosamente projetado para agregar informações de contexto de múltiplas sub-regiões de uma imagem, formando uma representação de "prior global" da cena. O PPM é uma resposta direta à limitação dos campos receptivos empíricos em redes mais rasas, que, apesar de terem um campo de visão teórico grande, falham em incorporar o contexto global de forma significativa.

O mecanismo do PPM é o seguinte: ele aplica operações de pooling adaptativas em quatro escalas diferentes (por exemplo, 1x1, 2x2, 3x3 e 6x6) sobre o mapa de características do *backbone*. Os resultados desses pooling são então redimensionados para o tamanho do mapa de entrada original e concatenados. Essa fusão estratégica permite que a rede analise a imagem em várias granularidades, capturando desde detalhes finos até os padrões mais amplos da cena. A abordagem de PSPNet contrasta com as conexões de salto da U-Net, que se concentram mais em preservar a precisão espacial de detalhes do que em capturar um entendimento semântico da cena completa.

### **Implementação Prática com pytorch-segmentation-models**

A PSPNet também está disponível na biblioteca SMP por meio da API smp.PSPNet(). A implementação na biblioteca permite a configuração dos parâmetros de pooling (pyramid\_sizes) de acordo com as necessidades específicas do conjunto de dados, tornando-a adaptável a uma ampla variedade de problemas.

```python
import segmentation_models_pytorch as smp
import torch

# Instanciando um modelo PSPNet com encoder MobileNetV2
model = smp.PSPNet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    classes=19,
    activation='softmax'
)

# Criando um tensor de entrada de exemplo
dummy_input = torch.randn(1, 3, 256, 256)
output_mask = model(dummy_input)

print(f"Forma da máscara de saída: {output_mask.shape}")
```

## **4. FPN: A Pirâmide de Características Semanticamente Rica**

A FPN (Feature Pyramid Network) foi desenvolvida para resolver um problema inerente às CNNs profundas: embora as camadas mais profundas capturem características ricas em semântica, sua baixa resolução espacial prejudica a detecção e segmentação de objetos pequenos. Enquanto a U-Net usa as conexões de salto para fundir informações do *encoder* com o *decoder* para refinar limites, a FPN vai além, construindo uma pirâmide de características semanticamente ricas em todas as escalas para lidar com a grande variedade de tamanhos de objetos em uma cena.

### **O Design da FPN: Caminhos e Conexões**

A FPN revoluciona a extração de características ao construir uma pirâmide que possui informações semânticas de alto nível em todas as suas escalas. Sua arquitetura consiste em três partes:

1. **Caminho de Baixo para Cima (*Bottom-Up*):** É a rede convolucional padrão (por exemplo, ResNet) que processa a imagem de entrada, criando uma hierarquia de características.  
2. **Caminho de Cima para Baixo (*Top-Down*):** Este caminho propaga as características ricas em semântica das camadas profundas para as camadas rasas, utilizando a operação de *upsampling*.  
3. **Conexões Laterais (*Lateral Connections*):** Estas são as "ligas" da FPN. Elas fundem os mapas de características *upsampleados* do caminho de cima para baixo com os mapas de alta resolução e baixa semântica do caminho de baixo para cima. Essa fusão, geralmente uma soma elemento a elemento, permite que a rede combine o entendimento global com os detalhes espaciais finos, garantindo que o modelo seja eficaz na detecção de objetos de todos os tamanhos. A formulação matemática dessa fusão é expressa como Pi​=Upsample(Pi+1​)+Ci​, onde Pi​ é o mapa de características na camada i e Ci​ é o mapa correspondente do *backbone*.

A FPN é, por natureza, uma arquitetura de extração de características e não uma arquitetura de segmentação completa. Sua inovação reside em como ela estrutura as características do *backbone* para serem sensíveis a múltiplas escalas. Essa característica a torna um "extrator de características universal" para tarefas que exigem sensibilidade a várias escalas, como a detecção de objetos (por exemplo, em *Faster R-CNN* e *Mask R-CNN*) e segmentação de instâncias, além de segmentação semântica. A inclusão da FPN na biblioteca SMP para segmentação demonstra a adaptabilidade e o valor de sua estrutura para uma ampla gama de problemas de visão computacional.

### **Implementação Prática com pytorch-segmentation-models**

A API da SMP para a FPN é consistente com as demais arquiteturas, mantendo a facilidade de uso e a flexibilidade para a escolha do *encoder*.

```python
import segmentation_models_pytorch as smp
import torch

# Instanciando um modelo FPN com o encoder EfficientNet-b4
model = smp.FPN(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

# Criando um tensor de entrada de exemplo
dummy_input = torch.randn(1, 3, 256, 256)
output_mask = model(dummy_input)

print(f"Forma da máscara de saída: {output_mask.shape}")
```

## **5. Análise Comparativa: O Confronto Prático e a Escolha Estratégica**

Não existe um modelo de segmentação universalmente superior; o desempenho ideal de uma arquitetura é altamente dependente do contexto e do conjunto de dados. Para um desenvolvedor de curso, a compreensão e a comunicação desse fato são cruciais. Cada uma das arquiteturas aqui discutidas oferece uma solução única para desafios específicos, e a escolha estratégica é a chave para o sucesso em um projeto real.

### **Vantagens e Desvantagens Estratégicas**

* **U-Net:** Sua principal vantagem é a capacidade de gerar máscaras de alta precisão em *datasets* relativamente pequenos, graças às *skip connections* que preservam a informação espacial. É a escolha ideal para aplicações biomédicas e onde a precisão de borda é a prioridade máxima.  
* **DeepLab:** Sua força reside em segmentar objetos em múltiplas escalas de forma eficiente. A convolução atrous permite um *trade-off* otimizado entre precisão e velocidade, tornando-a uma excelente candidata para tarefas de segmentação de cenas urbanas, onde o tamanho dos objetos varia amplamente.  
* **PSPNet:** A PSPNet tem uma vantagem em tarefas de "análise de cena" e em *datasets* de sensoriamento remoto, onde o contexto global é mais importante que os detalhes finos. O módulo PPM garante que a rede entenda a cena como um todo, o que pode evitar erros de classificação baseados apenas em características locais.  
* **FPN:** A FPN é excepcional para *datasets* com grande variação na escala de objetos. Sua arquitetura de pirâmide de características semanticamente rica assegura que a semântica seja preservada em todas as resoluções, tornando-a um extrator de características robusto para múltiplas tarefas, incluindo a detecção de objetos.

### **Análise de Desempenho Empírico**

A importância da escolha do *backbone* e do *dataset* é reforçada por resultados de pesquisa que podem parecer contraditórios. Em um estudo de segmentação do trato gastrointestinal, a combinação de **DeepLabV3+** com o *encoder* **ResNet50** superou o desempenho da U-Net e da PSPNet em métricas como Coeficiente de Dice e Índice de Jaccard (IoU). Por outro lado, um estudo diferente, focado no mapeamento de ervas daninhas, mostrou que a **PSPNet** superou a U-Net e a DeepLabV3. Esta divergência de resultados enfatiza que a escolha da arquitetura deve ser informada pelo problema que ela foi projetada para resolver.

Para um guia prático, a Tabela 1 oferece uma visão estratégica sobre quando e por que escolher cada arquitetura, enquanto a Tabela 2 apresenta um exemplo quantitativo das diferenças de desempenho.

**Tabela 1: Guia Estratégico de Arquiteturas de Segmentação**

| Arquitetura | Inovação Principal | Vantagem Primária | Casos de Uso Ideal |
| :---- | :---- | :---- | :---- |
| **U-Net** | *Skip Connections* | Precisão de Bordas em Imagens de Alta Resolução | Imagens Biomédicas, Segmentação de Células |
| **DeepLab** | Convolução Atrous / ASPP | Contexto Multi-Escala e Densidade de Características | Cenas Urbanas, Imagens de Satélite |
| **PSPNet** | *Pyramid Pooling* | Contexto Global de Cena | Sensoriamento Remoto, Análise de Paisagem |
| **FPN** | Pirâmide de Características | Eficácia em Várias Escalas | Detecção de Objetos, Cenários com Grande Variação de Tamanho |

**Tabela 2: Comparativo de Desempenho (Encoder: ResNet50)**

| Combinação (Encoder/Decoder) | Coeficiente de Dice | Índice de Jaccard (IoU) | Perda do Modelo |
| :---- | :---- | :---- | :---- |
| **ResNet50 / DeepLabV3+** | 0.9082 | 0.8796 | 0.1177 |
| **ResNet50 / U-Net** | 0.8970 | 0.8786 | 0.1239 |
| **ResNet50 / PSPNet** | 0.8847 | Não disponível | 0.1323 |

Fonte: Adaptado de pesquisa empírica em um estudo de segmentação do trato gastrointestinal.

## **6. Conclusão: O Ecossistema pytorch-segmentation-models**

A biblioteca pytorch-segmentation-models transcende o papel de um simples *wrapper* de código. Ela se estabelece como um ecossistema que reflete o estado da arte na pesquisa de segmentação de imagens, oferecendo um API consistente e flexível. A biblioteca abstrai a complexidade de cada arquitetura, como as *skip connections* da U-Net, a convolução atrous da DeepLab, o módulo PPM da PSPNet e o design de pirâmide da FPN, permitindo que os desenvolvedores se concentrem nas etapas cruciais de preparação de dados, treinamento e otimização.

Esta facilidade de uso acelera drasticamente o ciclo de desenvolvimento, permitindo a rápida prototipagem e experimentação de diferentes combinações de *encoder* e *decoder*. Para o desenvolvedor que busca construir um módulo de curso de *deep learning*, a biblioteca SMP é uma ferramenta essencial. Ela não apenas simplifica a implementação, mas também oferece funcionalidades avançadas, como a integração de módulos de atenção (SCSE) e a adição de saídas auxiliares (aux\_params), que são cruciais para aprimorar o desempenho de modelos em tarefas de segmentação. Ao fornecer acesso a um vasto leque de *backbones* pré-treinados, a biblioteca permite a personalização completa para atender aos requisitos de qualquer projeto, desde aplicações com restrições de latência até tarefas complexas que exigem a máxima precisão.

#### **Referências citadas**

1. DeepLab: Semantic Image Segmentation with Deep Convolutional ..., acessado em agosto 28, 2025, [https://www.cs.jhu.edu/\~ayuille/JHUcourses/VisionAsBayesianInference2022/9/DeepLabJayChen.pdf](https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2022/9/DeepLabJayChen.pdf)  
2. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs \- Liang-Chieh Chen, acessado em agosto 28, 2025, [http://liangchiehchen.com/projects/DeepLab.html](http://liangchiehchen.com/projects/DeepLab.html)  
3. How DeepLabV3 Works | ArcGIS API for Python \- Esri Developer, acessado em agosto 28, 2025, [https://developers.arcgis.com/python/latest/guide/how-deeplabv3-works/](https://developers.arcgis.com/python/latest/guide/how-deeplabv3-works/)  
4. DeepLabv3+ (with checkpoint) – Vertex AI \- Google Cloud Console, acessado em agosto 28, 2025, [https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagesegmentation-deeplabv3](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagesegmentation-deeplabv3)  
5. DeepLabV3 Guide: Key to Image Segmentation \- Ikomia, acessado em agosto 28, 2025, [https://www.ikomia.ai/blog/understanding-deeplabv3-image-segmentation](https://www.ikomia.ai/blog/understanding-deeplabv3-image-segmentation)  
6. segmentation\_models\_pytorch.deeplabv3.model \- Segmentation Models documentation, acessado em agosto 28, 2025, [https://smp.readthedocs.io/en/v0.1.3/\_modules/segmentation\_models\_pytorch/deeplabv3/model.html](https://smp.readthedocs.io/en/v0.1.3/_modules/segmentation_models_pytorch/deeplabv3/model.html)  
7. How PSPNet works? | ArcGIS API for Python \- Esri Developer, acessado em agosto 28, 2025, [https://developers.arcgis.com/python/latest/guide/how-pspnet-works/](https://developers.arcgis.com/python/latest/guide/how-pspnet-works/)  
8. PSPNet (Pyramid Scene Parsing Network) for Image Segmentation \- GeeksforGeeks, acessado em agosto 28, 2025, [https://www.geeksforgeeks.org/computer-vision/pspnet-pyramid-scene-parsing-network-for-image-segmentation/](https://www.geeksforgeeks.org/computer-vision/pspnet-pyramid-scene-parsing-network-for-image-segmentation/)  
9. Trending Papers \- Hugging Face, acessado em agosto 28, 2025, [https://paperswithcode.com/method/pyramid-pooling-module](https://paperswithcode.com/method/pyramid-pooling-module)  
10. PSPNet structure diagram. \- ResearchGate, acessado em agosto 28, 2025, [https://www.researchgate.net/figure/PSPNet-structure-diagram\_fig2\_353372992](https://www.researchgate.net/figure/PSPNet-structure-diagram_fig2_353372992)  
11. segmentation-models-pytorch \- Read the Docs, acessado em agosto 28, 2025, [https://segmentation-modelspytorch.readthedocs.io/en/latest/](https://segmentation-modelspytorch.readthedocs.io/en/latest/)  
12. Feature Pyramid Network (FPN) \- GeeksforGeeks, acessado em agosto 28, 2025, [https://www.geeksforgeeks.org/computer-vision/feature-pyramid-network-fpn/](https://www.geeksforgeeks.org/computer-vision/feature-pyramid-network-fpn/)  
13. FPN Explained — Feature Pyramid Network | by Amit Yadav \- Medium, acessado em agosto 28, 2025, [https://medium.com/@amit25173/fpn-explained-feature-pyramid-network-7c0f65ea8f8b](https://medium.com/@amit25173/fpn-explained-feature-pyramid-network-7c0f65ea8f8b)  
14. Mastering Feature Pyramid Networks \- Number Analytics, acessado em agosto 28, 2025, [https://www.numberanalytics.com/blog/mastering-feature-pyramid-networks](https://www.numberanalytics.com/blog/mastering-feature-pyramid-networks)  
15. Test results of different semantic segmentation models \- ResearchGate, acessado em agosto 28, 2025, [https://www.researchgate.net/figure/Test-results-of-different-semantic-segmentation-models\_tbl4\_357246304](https://www.researchgate.net/figure/Test-results-of-different-semantic-segmentation-models_tbl4_357246304)  
16. Encoder–Decoder Variant Analysis for Semantic Segmentation of ..., acessado em agosto 28, 2025, [https://www.mdpi.com/2306-5354/12/3/309](https://www.mdpi.com/2306-5354/12/3/309)  
17. O que é : U-Net \- IA Tracker, acessado em agosto 28, 2025, [https://iatracker.com.br/glossario/o-que-e-u-net/](https://iatracker.com.br/glossario/o-que-e-u-net/)  
18. DeepLabv3+ \- Computer Vision Wiki \- CloudFactory, acessado em agosto 28, 2025, [https://wiki.cloudfactory.com/docs/mp-wiki/model-architectures/deeplabv3](https://wiki.cloudfactory.com/docs/mp-wiki/model-architectures/deeplabv3)  
19. qubvel-org/segmentation\_models.pytorch: Semantic segmentation models with 500+ pretrained convolutional and transformer-based backbones. \- GitHub, acessado em agosto 28, 2025, [https://github.com/qubvel-org/segmentation\_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)  
20. Unet \- Segmentation Models documentation, acessado em agosto 28, 2025, [https://smp.readthedocs.io/en/latest/models.html](https://smp.readthedocs.io/en/latest/models.html)