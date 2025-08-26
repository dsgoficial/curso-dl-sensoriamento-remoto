---
sidebar_position: 5
title: "Transfer Learning e Fine-Tunning em Sensoriamento Remoto"
description: "Técnicas de transferência de conhecimento aplicadas ao processamento de imagens de satélite e sensoriamento remoto"
tags: [transfer-learning, fine-tuning, feature-extraction, sensoriamento-remoto, domain-adaptation, pytorch]
---

# Transfer Learning e Fine-Tunning em Deep Learning para Sensoriamento Remoto

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_tMEwD-BhdNturtx3RgA6Z9wJXtYd9Bq?usp=sharing)

## 1. Introdução ao Transfer Learning

### 1.1. O que é Transfer Learning? Uma Visão Geral

O Transfer Learning, ou Aprendizagem por Transferência, é uma técnica fundamental em aprendizado de máquina e, em particular, em deep learning, que permite a reutilização do conhecimento adquirido em uma tarefa para melhorar o desempenho em uma nova tarefa relacionada. Em vez de treinar um modelo do zero (com inicialização aleatória), esta abordagem utiliza um modelo já treinado em um grande conjunto de dados (dataset) e em uma tarefa fonte, aproveitando a inteligência aprendida para resolver um problema de destino. A eficácia da técnica reside na premissa de que os recursos e padrões aprendidos nas camadas iniciais de uma rede neural profunda são de natureza genérica e transferíveis para outros domínios.

Um exemplo clássico na computação é a aplicação do conhecimento de uma rede que foi treinada para classificar cachorros para uma nova tarefa de classificar caminhões. As camadas iniciais do modelo original, por exemplo, teriam aprendido a identificar características visuais de baixo nível, como arestas, texturas e formas, que são relevantes para a maioria das tarefas de visão computacional. Este conhecimento fundamental serve como um ponto de partida superior, permitindo que a rede se adapte mais rapidamente e com menos dados a um novo problema. Esta técnica é especialmente valiosa para a construção de redes neurais profundas, que frequentemente exigem grandes volumes de dados para alcançar um desempenho ótimo. O conceito de reutilizar o aprendizado já havia sido explorado em publicações anteriores a 1980, mas ganhou proeminência com o crescimento exponencial de dados e o desenvolvimento de arquiteturas de rede complexas.

### 1.2. Benefícios Chave: Redução de Custos, Eficiência de Dados e Generalização

A adoção do Transfer Learning em projetos de deep learning oferece vantagens substanciais que abordam diretamente algumas das maiores barreiras do campo.

- **Redução de Custos Computacionais**: A reutilização de modelos pré-treinados diminui significativamente o tempo de treinamento necessário para construir modelos para novos problemas. Ao aproveitar as representações já aprendidas, os desenvolvedores podem reduzir a quantidade de épocas e os recursos computacionais, como GPUs e CPUs, necessários para atingir o desempenho desejado. Isso acelera e simplifica o processo de treinamento de modelos.

- **Eficiência de Dados**: O Transfer Learning ajuda a mitigar o problema da escassez de dados, que é um desafio comum em muitas áreas, incluindo o sensoriamento remoto. A aquisição e a rotulagem manual de grandes volumes de dados podem ser caras e demoradas. Ao transferir conhecimento de modelos treinados em vastos datasets públicos, é possível obter modelos de alto desempenho com um número de amostras substancialmente menor no domínio de destino.

- **Melhora da Generalização**: O re-treinamento de um modelo existente com um novo dataset pode melhorar sua capacidade de generalização e inibir o overfitting. A incorporação de conhecimento de múltiplos datasets (o original, grande, e o novo, de destino) resulta em um modelo mais robusto, que potencialmente se sairá melhor em uma variedade mais ampla de dados do que um modelo treinado em apenas um tipo de dataset.

### 1.3. Estratégias Fundamentais: Feature Extraction vs. Fine-tuning

Existem duas abordagens principais para a aplicação de Transfer Learning, cada uma com suas características e cenários de uso ideais.

- **Feature Extraction (Extração de Características)**: Esta abordagem consiste em utilizar o modelo pré-treinado como um extrator de características estático. As camadas convolucionais do modelo são mantidas congeladas, e apenas um novo classificador (geralmente uma ou mais camadas totalmente conectadas) é adicionado ao topo e treinado a partir do zero. O modelo pré-treinado calcula representações ricas e significativas para os novos dados, e o classificador aprende a mapear essas representações para as novas classes. Esta estratégia é particularmente útil quando o dataset de destino é pequeno ou os recursos computacionais são limitados, pois a quantidade de parâmetros treináveis é drasticamente reduzida.

- **Fine-tuning (Ajuste Fino)**: O fine-tuning é uma extensão do feature extraction onde não apenas as novas camadas são treinadas, mas também uma porção das últimas camadas do modelo pré-treinado é "descongelada" e os pesos são ajustados para se adequarem melhor à nova tarefa. O fine-tuning é geralmente empregado quando o dataset de destino é maior e a nova tarefa difere significativamente da tarefa original. Embora seja mais custoso computacionalmente e exija um conjunto de dados mais robusto para evitar o overfitting, o fine-tuning tem o potencial de alcançar um desempenho superior, pois permite que o modelo adapte seus recursos aprendidos de forma mais precisa aos detalhes específicos do novo domínio.

A escolha entre as duas estratégias depende de fatores como o tamanho do dataset de destino, a similaridade entre as tarefas de origem e destino e os recursos computacionais disponíveis. A tabela a seguir sintetiza as principais diferenças.

| Característica | Feature Extraction | Fine-tuning |
|---|---|---|
| **Conceito** | Congela as camadas pré-treinadas, treinando apenas a(s) nova(s) camada(s) de classificação. | Descongela e treina as últimas camadas do modelo pré-treinado, juntamente com as novas camadas. |
| **Dados Necessários** | Efetivo com datasets menores. | Requer um dataset maior para evitar overfitting. |
| **Custo Computacional** | Menor. Treinamento mais rápido e menos intensivo em recursos. | Maior. Treinamento mais demorado e intensivo em recursos. |
| **Risco de Overfitting** | Menor, devido ao número limitado de parâmetros treináveis. | Maior, especialmente com datasets pequenos. |
| **Adaptabilidade** | Limitada; o modelo não pode ajustar os recursos pré-treinados. | Alta; os recursos pré-treinados são adaptados à nova tarefa. |
| **Melhor Cenário de Uso** | Dataset pequeno, similar à tarefa original. | Dataset grande, tarefa possivelmente diferente da original. |

## 2. Guia de Implementação em PyTorch

### 2.1. O Mecanismo de Congelar Camadas (Freezing): Guia Prático com Código

O congelamento de camadas é uma etapa fundamental do Transfer Learning, especialmente no feature extraction e na fase inicial do fine-tuning. O objetivo é evitar que os pesos das camadas pré-treinadas sejam atualizados durante a retropropagação, o que permite aproveitar o conhecimento aprendido sem o risco de "esquecer" os padrões fundamentais. Isso também resulta em uma redução significativa do consumo de memória e do tempo de processamento, pois o otimizador não precisa calcular gradientes para essas camadas.

Em PyTorch, o congelamento de camadas é realizado através do controle do atributo `.requires_grad` de cada parâmetro. Quando este atributo é definido como False, o motor de auto-diferenciação do PyTorch (autograd) não rastreia as operações com este parâmetro, garantindo que ele não seja atualizado.

#### Congelando a Rede Inteira

Para a estratégia de feature extraction, onde todo o backbone pré-treinado deve ser congelado, uma abordagem simples e eficiente é iterar sobre todos os parâmetros do modelo e desativar o cálculo de gradientes.

```python
import torch
import torchvision.models as models

# Carregar um modelo pré-treinado (ex: ResNet-18)
model = models.resnet18(weights='IMAGENET1K_V1')

# Congelar todas as camadas do modelo
for param in model.parameters():
    param.requires_grad = False

# Verificar se os parâmetros estão congelados
for name, param in model.named_parameters():
    print(f"Camada: {name}, requires_grad: {param.requires_grad}")

# A camada final (classificador) ainda precisa ser adaptada e treinada
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 45)  # Exemplo para o dataset RESIC-45
model.fc.requires_grad_(True)  # O requires_grad já é True por padrão em novas camadas, mas é uma boa prática
```

Este código ilustra a simplicidade de congelar todo o backbone. A nova camada final (`model.fc`) é substituída por uma com o número correto de classes, e seus parâmetros são os únicos que serão atualizados durante o treinamento.

#### Congelando Camadas Específicas (Fine-tuning)

No fine-tuning, é comum congelar as primeiras camadas (que aprendem características mais genéricas, como arestas e texturas) e descongelar as camadas mais profundas (que aprendem características mais específicas da tarefa). Para isso, é necessário iterar sobre as camadas do modelo e aplicar o congelamento seletivamente. O método `model.named_children()` é útil para inspecionar e manipular as camadas pelo nome.

O exemplo a seguir mostra como congelar as camadas iniciais de uma ResNet, mantendo as últimas camadas e a nova camada de classificação treináveis.

```python
# Re-carregar o modelo pré-treinado
model = models.resnet18(weights='IMAGENET1K_V1')

# Congelar camadas até 'layer3' (excluindo 'layer4' e 'fc')
for name, child in model.named_children():
    if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        for param in child.parameters():
            param.requires_grad = False

# Substituir a camada final
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 45)

# Otimizador deve incluir apenas os parâmetros treináveis
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Verificar quais camadas são treináveis
print("Camadas treináveis:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

O uso da função `filter()` ao instanciar o otimizador é uma melhor prática para garantir que apenas os parâmetros com `requires_grad=True` sejam incluídos no processo de otimização. Isso economiza memória e recursos computacionais, garantindo que o backpropagation se concentre apenas nas camadas que precisam de ajuste.

### 2.2. Learning Rate Warmup: A Importância de um Início Suave

O learning rate warmup é uma técnica crucial que modifica o agendador de taxa de aprendizado para que ele comece com um valor baixo no início do treinamento e aumente gradualmente até o valor nominal. Embora o conceito seja simples, a sua justificação teórica e os seus benefícios são multifacetados e de grande importância para a estabilidade do treinamento.

A necessidade de um início suave se origina do estado caótico de uma rede neural recém-inicializada. Com pesos aleatórios, as primeiras atualizações de gradiente podem ser grandes, instáveis e imprevisíveis, podendo levar o modelo a uma região ruim da paisagem de perda ou até mesmo à divergência. O warmup age como uma medida de estabilização, limitando a magnitude dessas atualizações iniciais e impedindo que o modelo "se precipite" para um espaço de parâmetros subótimo.

O benefício principal do warmup é que ele permite que o modelo tolere taxas de aprendizado maiores do que de outra forma seria possível. Isso se deve a um efeito no formato da paisagem de perda, em que o warmup empurra o modelo para regiões mais planas e bem-condicionadas do espaço de parâmetros, onde taxas de aprendizado maiores não causam instabilidade. Além disso, em otimizadores adaptativos como o Adam, o warmup atua como um método de redução de variância, permitindo que o otimizador colete estatísticas de gradiente mais precisas dos primeiros mini-lotes antes de aplicar grandes atualizações. Sem o warmup, a variância nas estimativas iniciais dos momentos do gradiente pode levar a um desempenho degradado e falhas de treinamento.

A etapa de warmup pode ser feita simplesmente com uma taxa de aprendizado baixa e com parte da rede congelada.

### 2.3. Lidando com a Batch Normalization

A Batch Normalization (BN) é uma técnica crucial em redes neurais modernas que normaliza as ativações de cada camada com base nas estatísticas do mini-batch (média e variância), estabilizando o processo de treinamento. Esta estabilização permite o uso de taxas de aprendizado maiores, acelera a convergência e reduz a dependência da inicialização dos pesos.

#### O Grande Problema do Fine-tuning e a Batch Normalization

Um dos maiores desafios no fine-tuning é o manuseio das camadas de Batch Normalization. Embora a BN seja benéfica para o treinamento do zero, sua aplicação no fine-tuning pode ser contraproducente. Durante o treinamento, a BN usa as estatísticas do mini-batch atual para normalizar as ativações. No entanto, para a inferência e validação, ela usa estatísticas de execução (running stats), que são médias móveis calculadas ao longo de todo o treinamento.

O ponto crítico é que as estatísticas dos mini-batches no seu novo dataset de sensoriamento remoto (o domínio de destino) podem ser muito diferentes das estatísticas de execução que a rede aprendeu durante o pré-treinamento na ImageNet (o domínio de origem). Se as camadas BN forem treinadas, elas começarão a calcular novas médias e variâncias com base nos mini-batches do novo dataset, que são tipicamente menores. Este cálculo, feito com base em dados de um domínio diferente e de um número de amostras reduzido, pode ser impreciso e instável. A consequência direta é que as novas estatísticas de running podem "destruir o que o modelo aprendeu", levando à degradação do desempenho ou a falhas no treinamento.

A solução para este problema é manter as camadas de Batch Normalization em modo de avaliação (`eval()`) durante todo o processo de fine-tuning. Isso força a camada a usar as robustas estatísticas de execução (running stats) pré-treinadas da ImageNet, garantindo a estabilidade e preservando o conhecimento adquirido.

#### model.eval() vs. param.requires_grad = False

É crucial entender a distinção entre `model.eval()` e `param.requires_grad = False`, pois eles executam funções diferentes e complementares no contexto da BN.

- **param.requires_grad = False**: Este comando apenas congela os parâmetros treináveis da camada (os pesos e biases) e impede que eles sejam atualizados durante a retropropagação. Ele não afeta o comportamento da camada em si. Por exemplo, uma camada BN ainda pode continuar a calcular e a atualizar suas estatísticas de execução com base nos novos mini-batches, mesmo se seus pesos estiverem congelados. Isso ainda levaria aos problemas de instabilidade.

- **model.eval()**: Este comando, por outro lado, muda o comportamento da camada. Ele instrui todas as camadas do modelo (ou uma camada específica, se chamado individualmente) a se comportarem no modo de inferência. Para a Batch Normalization, isso significa que a camada usará suas estatísticas de running fixas para a normalização, em vez de recalcular a média e a variância a partir do mini-batch.

Portanto, a melhor prática para o fine-tuning de um modelo com camadas BN é usar a combinação de ambos os comandos: `param.requires_grad = False` para congelar os pesos das camadas convolucionais (se for o caso) e `model.eval()` para garantir que as camadas BN se comportem de forma estável, usando as estatísticas pré-treinadas. Isso é fundamental para evitar a degeneração do modelo durante o treinamento em novos dados.

## 3. Transfer Learning Aplicado a Sensoriamento Remoto

### 3.1. O Contexto do Sensoriamento Remoto: Desafios e Oportunidades

A área de sensoriamento remoto está em um período de crescimento sem precedentes, impulsionada pela disponibilidade de dados de alta resolução de satélites e aeronaves. A imensa quantidade de dados gerados exige soluções eficientes para a análise de imagens, uma tarefa que as técnicas tradicionais de aprendizado de máquina não conseguem mais lidar de forma eficaz. A análise manual de características é impraticável, e a necessidade de modelos de IA para tarefas como classificação de cenas e detecção de objetos é cada vez maior.

No entanto, a aplicação de deep learning no sensoriamento remoto enfrenta desafios significativos. O principal deles é a escassez de grandes datasets de alta qualidade e bem anotados, pois a rotulagem de imagens de satélite exige um conhecimento especializado e é um processo caro e demorado. A aquisição de dados é cara, e os datasets geralmente são pequenos, o que pode levar a problemas de overfitting quando modelos complexos são treinados do zero.

Neste contexto, o Transfer Learning surge como uma solução promissora e indispensável. A técnica permite que a comunidade de sensoriamento remoto contorne a necessidade de grandes datasets rotulados, alcançando altas taxas de acurácia com um número limitado de amostras. Estudos de caso demonstraram que é possível obter classificações de alta precisão (acima de 96% de acurácia) com poucas centenas de imagens por classe, algo que seria inviável sem o uso de modelos pré-treinados.

### 3.2. O Desafio da Divergência de Domínio (Domain Shift)

A aplicação do Transfer Learning em sensoriamento remoto, embora poderosa, não é isenta de desafios. O principal deles é a divergência de domínio (domain shift), que é a diferença na distribuição dos dados entre o domínio de origem (o dataset de pré-treinamento) e o domínio de destino (o dataset de sensoriamento remoto). Este problema é a maior limitação da técnica padrão, mas também se revela a principal oportunidade para o desenvolvimento de métodos mais avançados.

O domain shift é particularmente grave no sensoriamento remoto devido a diversas causas:

- **Diferenças de Sensores**: As imagens de satélite podem ser capturadas por diferentes sensores, cada um com suas próprias características espectrais, resolução e calibração, resultando em representações visuais distintas do mesmo objeto.

- **Variações Temporais e Sazonais**: As condições de iluminação, a cobertura de nuvens, as sombras e os ciclos de vegetação mudam drasticamente com as estações e o clima. Uma floresta em uma imagem de inverno pode ser visualmente muito diferente da mesma floresta em uma imagem de verão.

- **Variações de Localização Geográfica**: As características da superfície, como a aparência de edifícios ou a composição do solo, variam enormemente entre diferentes regiões geográficas, mesmo para a mesma classe de cena.

Modelos pré-treinados em datasets de imagens naturais como a ImageNet, que é composta por imagens RGB de objetos do dia-a-dia, têm limitações inerentes ao lidar com dados de sensoriamento remoto. As características de baixo nível, como arestas e texturas, podem ser transferidas com sucesso, mas as características de alto nível, como a aparência de um campo de pouso ou de uma instalação industrial, são fundamentalmente diferentes. Além disso, muitas imagens de sensoriamento remoto contêm objetos muito pequenos (ex: veículos em uma imagem de satélite) ou dependem de bandas espectrais fora do espectro visível (infravermelho, etc.), informações que um modelo treinado na ImageNet simplesmente não possui.

### 3.3. Escolhendo a Estratégia de Pré-Treinamento Adequada

A escolha do ponto de partida para o Transfer Learning é uma decisão estratégica. A ImageNet continua sendo uma excelente base, mas não é a única opção.

- **Quando Usar a ImageNet**: Apesar das limitações, modelos pré-treinados na ImageNet são um ponto de partida eficaz para a classificação de cenas de satélite, especialmente com datasets RGB como o RESIC-45. As redes convolucionais aprendem a extrair características visuais genéricas que são úteis para muitas tarefas, e o fine-tuning pode adaptar as últimas camadas para reconhecer os padrões específicos das cenas de satélite. Estudos mostram que modelos como a ResNet-50 pré-treinada na ImageNet alcançaram altas acurácias (96%) em datasets como o RESIC-45.

- **Modelos Pré-treinados em Domínios Similares**: Para tarefas mais específicas, como a detecção de objetos em imagens de satélite, a evidência sugere que o pré-treinamento em datasets de sensoriamento remoto é superior. Um estudo com o dataset DOTA (um conjunto de dados para detecção de objetos em imagens aéreas) demonstrou que o pré-treinamento em DOTA, juntamente com o pré-treinamento na ImageNet, resultou em melhorias significativas nas pontuações F1 para a detecção de veículos, aeronaves e navios. Isso confirma que, quanto mais similar for o domínio de pré-treinamento ao domínio de destino, maior será o ganho de acurácia esperado.

- **O Futuro: O Papel do Self-Supervised Learning (SSL)**: Uma das maiores promessas para o futuro do sensoriamento remoto é o Self-Supervised Learning (Aprendizagem Autossupervisionada). O SSL permite que modelos aprendam representações de dados sem a necessidade de anotações humanas, resolvendo diretamente o problema da escassez de dados rotulados. O SSL explora o vasto oceano de dados de sensoriamento remoto não rotulados, que é um recurso abundante. Modelos como o DINOv3 da Meta AI já demonstraram a capacidade de aprender características visuais universais a partir de bilhões de imagens, com uma versão especializada para tarefas de satélite. Tais modelos podem ser transferidos para novos domínios com a promessa de desempenho superior, mesmo com a falta de dados rotulados.
