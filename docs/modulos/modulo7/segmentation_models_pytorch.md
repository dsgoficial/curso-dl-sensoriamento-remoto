---
sidebar_position: 5
title: "Segmentação Semântica Multiclasse com U-Net e segmentation-models-pytorch"
description: "Explicando o training loop de segmentação semântica"
tags: [unet, segmentação, conexões-de-salto, deep-learning, pytorch]
---

**Treinamento U-Net no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csYh1gcwkiXdljr9Yw5PRh5YpwMwpgll?usp=sharing)

# Segmentação Semântica Multiclasse com U-Net e segmentation-models-pytorch

## 1. Introdução: Fundamentos da Segmentação Multiclasse

A segmentação semântica é uma tarefa fundamental na área de visão computacional que transcende a simples identificação de objetos. Diferentemente da classificação, que atribui um único rótulo a uma imagem inteira, ou da deteção de objetos, que delimita objetos com caixas envolventes, a segmentação semântica busca classificar **cada pixel** de uma imagem com um rótulo de classe correspondente. O resultado é uma "máscara" que define com precisão as regiões pertencentes a diferentes categorias, como fundo, veículo, pessoa, estrada, entre outras. A segmentação multiclasse, em particular, estende este conceito para mais de duas categorias, permitindo a distinção entre múltiplos objetos de interesse e o seu ambiente.

### 1.1. A Arquitetura U-Net: Uma Visão Geral

A U-Net, introduzida em 2015, revolucionou o campo da segmentação de imagens, especialmente em aplicações biomédicas, devido à sua capacidade de produzir segmentações precisas mesmo com um conjunto de dados limitado. A arquitetura é denominada "U-Net" pela sua forma característica em "U", que é dividida em duas partes principais: um caminho de **encoder** (ou caminho de contração) e um caminho de **decoder** (ou caminho de expansão).

O encoder atua como um extrator de características, diminuindo gradualmente as dimensões espaciais da imagem de entrada (por exemplo, através de operações de pooling ou convoluções com stride maior que 1) enquanto aumenta a profundidade dos canais, capturando características contextuais de alto nível. Em seguida, o decoder reconstrói a imagem de segmentação, recuperando a resolução espacial por meio de camadas de upsampling (como a ConvTranspose2d). A inovação crucial da U-Net reside nas **"skip connections"** (conexões de salto) que ligam os mapas de características de resolução correspondente entre o encoder e o decoder. Estas conexões permitem que as informações de detalhes finos, perdidas durante o processo de downsampling do encoder, sejam transferidas diretamente para o decoder. Isso resulta em uma recuperação de bordas mais precisa e, em última análise, em máscaras de segmentação com contornos mais exatos.

### 1.2. A Biblioteca segmentation-models-pytorch (smp)

Para simplificar a implementação de arquiteturas complexas como a U-Net, a biblioteca segmentation-models-pytorch (conhecida como smp) oferece uma API de alto nível que permite a criação de um modelo com apenas algumas linhas de código. A smp é um recurso valioso para engenheiros de machine learning e investigadores, pois fornece acesso a uma vasta gama de arquiteturas de segmentação, incluindo a U-Net e a Unet++.

Além disso, a biblioteca suporta a utilização de uma grande variedade de encoders (backbones) pré-treinados em grandes bases de dados de classificação de imagens como a ImageNet. Esta funcionalidade é particularmente útil, pois permite que o modelo aproveite as características visuais já aprendidas, levando a uma convergência mais rápida e a um desempenho superior, especialmente quando o conjunto de dados de treino é de tamanho limitado. A smp é uma ferramenta robusta que abstrai a complexidade da construção do modelo, permitindo que os utilizadores se concentrem na preparação dos dados e no ciclo de treino.

## 2. Preparação de Dados para a Tarefa Multiclasse

O sucesso de qualquer pipeline de segmentação semântica depende criticamente da preparação correta dos dados de entrada, em particular das máscaras de verdade (ground truth). Uma confusão comum, especialmente para quem está a migrar de tarefas de deteção de objetos ou classificação, é a expectativa de ter várias máscaras de entrada para o treino. É crucial entender que para a segmentação multiclasse, o formato dos dados de verdade esperado pelo modelo e pelas funções de perda é um único tensor que representa a classe de cada pixel.

### 2.1. Desmistificando o Formato das Máscaras (Ground Truth)

A distinção entre o modo **multiclass** e o modo **multilabel** é um ponto fundamental. No modo multiclass, a suposição é que cada pixel na imagem pertence a **exatamente uma** das classes de interesse. Em contraste, o modo multilabel permite que um pixel pertença a múltiplas classes simultaneamente. A maioria das tarefas de segmentação semântica, como a segmentação de objetos em imagens aéreas ou o mapeamento de solos em agricultura, enquadra-se no cenário multiclass.

Para o modo multiclass, o formato esperado para o tensor de verdade (y_true) é de um único canal, com a forma `(H, W)`. Cada valor de pixel neste tensor é um **índice de classe numérico**, geralmente um LongTensor, que varia de 0 a nb_classes - 1. O valor 0 é frequentemente reservado para o fundo ou para uma classe de "não-objeto". Este formato compacto é mais eficiente do que o uso de codificação one-hot, que seria necessário para o modo multilabel ou em algumas implementações específicas.

### 2.2. O Processo de Conversão de Máscaras

Frequentemente, as máscaras de segmentação para conjuntos de dados multiclasse vêm em formatos de imagem, como arquivos .png, onde cada classe é representada por uma cor RGB sólida. Para converter estas máscaras visuais para o formato de índice numérico exigido pelo PyTorch, é necessário um processo de mapeamento.

O primeiro passo é definir um mapeamento entre os valores de cor RGB (ou labels de cores) e os índices de classe inteiros. Por exemplo, a cor preta pode corresponder ao índice `0` (fundo), enquanto uma cor como pode corresponder ao índice 1 (primeira classe de objeto). A implementação de uma função que lê a imagem de máscara de cor e, para cada pixel, substitui o valor RGB pelo seu índice de classe correspondente, resulta num tensor H x W de inteiros. Esta etapa é fundamental para o pré-processamento de dados e deve ser realizada de forma consistente para todo o conjunto de dados.

### 2.3. Implementação de um Dataset Customizado

Para carregar e processar as imagens e as máscaras de segmentação, um DataLoader padrão do PyTorch, como o ImageFolder, não é adequado porque este último é projetado para tarefas de classificação e não sabe como carregar e emparelhar imagens de entrada com as suas máscaras de verdade. A solução robusta é criar uma classe de Dataset customizada, que deve herdar de torch.utils.data.Dataset e implementar três métodos essenciais:

- **`__init__(self,...)`**: Este construtor inicializa a classe, carregando os caminhos dos arquivos de imagem e máscara.
- **`__len__(self)`**: Este método retorna o número total de amostras no conjunto de dados.
- **`__getitem__(self, idx)`**: Este é o método mais importante. Dado um índice, ele carrega a imagem e a máscara correspondente do disco, aplica as transformações necessárias e retorna o par (imagem, máscara) como tensores.

A implementação de transformações de aumento de dados (como rotação, crop aleatório ou flip) requer uma consideração adicional. Para manter o alinhamento espacial entre a imagem e a máscara, a mesma transformação geométrica deve ser aplicada de forma coordenada a ambos os tensores. Por exemplo, se a imagem de entrada for virada horizontalmente, a máscara de verdade também deve ser virada da mesma forma para que os rótulos de pixel continuem a corresponder aos objetos corretos. Este cuidado é vital para garantir que o modelo aprenda a correlação espacial correta entre as características visuais e os rótulos de pixel.

## 3. Configuração do Modelo e Saída dos Logits

A utilização da biblioteca smp simplifica a configuração do modelo U-Net, mas o sucesso do treino depende de uma configuração precisa dos parâmetros-chave, especialmente no contexto de uma tarefa multiclasse.

### 3.1. Criação e Configuração do Modelo U-Net com smp

Para instanciar um modelo U-Net com a biblioteca smp, a sintaxe é concisa e intuitiva:

```python
import segmentation_models_pytorch as smp

# Exemplo de criação de modelo U-Net para 4 classes
model = smp.Unet(
    encoder_name="resnet34",    # Escolhe o encoder (backbone)
    encoder_weights="imagenet", # Usa pesos pré-treinados da ImageNet
    in_channels=3,              # Canais de entrada (3 para imagens RGB)
    classes=4,                  # Número de classes de segmentação (ex: 3 classes de objetos + fundo)
    activation=None             # Configuração crucial para a perda
)
```

Os parâmetros fornecidos na criação do modelo são:

- **encoder_name**: Define a arquitetura do encoder (ou backbone), como resnet34 ou efficientnet-b0. A escolha afeta o número de parâmetros e a performance.
- **encoder_weights**: O uso de pesos pré-treinados, como imagenet, inicializa o encoder com características visuais já aprendidas, o que acelera a convergência e melhora o desempenho.
- **in_channels**: Especifica o número de canais da imagem de entrada. 3 é o valor padrão para imagens RGB.
- **classes**: Este é o parâmetro mais crítico para a segmentação multiclasse. O valor de classes determina o número de canais na camada de saída final do modelo. Cada canal na saída do modelo corresponderá à probabilidade (ou logit) de um pixel pertencer a uma classe específica.

### 3.2. A Saída do Modelo: Logits e a Importância de activation=None

A saída do modelo U-Net, antes da aplicação de qualquer função de ativação, é um tensor conhecido como **logits**. Este tensor tem a forma `(B, C, H, W)`. Os valores de logit são brutos, podendo ser positivos, negativos ou zero, e representam a "força" da evidência para cada classe em cada pixel.

Um ponto de design crucial ao criar o modelo com smp é definir o parâmetro activation como **None**. Isto garante que a camada de saída do modelo retorne os logits brutos em vez de aplicar uma função de ativação final, como softmax ou sigmoid. O motivo para esta abordagem é que as funções de perda mais comuns, como a torch.nn.CrossEntropyLoss e as implementações em smp (DiceLoss, FocalLoss), são otimizadas para trabalhar diretamente com logits. Elas incorporam operações como log_softmax internamente, que são numericamente mais estáveis e evitam possíveis problemas de underflow ou overflow que poderiam ocorrer ao calcular o log de probabilidades muito pequenas, especialmente em GPUs de precisão mista. Portanto, a prática padrão e recomendada é passar os logits brutos do modelo diretamente para a função de perda durante o treino.

## 4. Escolha e Implementação de Funções de Perda

A função de perda é o motor do treino, guiando a otimização da rede para minimizar a diferença entre as previsões do modelo e as máscaras de verdade (ground truth). A escolha da função de perda pode ter um impacto significativo na performance e na estabilidade do treino.

### 4.1. CrossEntropyLoss: A Linha de Base (Baseline)

A torch.nn.CrossEntropyLoss é a função de perda padrão e mais utilizada para problemas de classificação multiclasse, incluindo a segmentação semântica. A sua aplicação neste contexto é por pixel, tratando cada pixel como um ponto de dados a ser classificado em uma das C classes.

Uma característica notável desta função de perda, em comparação com outras como a DiceLoss, é a estabilidade do seu gradiente. O gradiente da CrossEntropyLoss em relação aos logits é simples e bem-comportado (p - t, onde p é a saída do softmax e t é o alvo), o que geralmente leva a um treino mais estável. Em contraste, o gradiente da DiceLoss pode ser numericamente instável e "feio" em certas condições, como quando as previsões (p) e os alvos (t) são pequenos simultaneamente. Por esta razão, a CrossEntropyLoss é frequentemente a primeira escolha para um modelo de linha de base. É essencial lembrar a regra de formato de entrada: a CrossEntropyLoss espera os logits brutos do modelo com a forma `(B, C, H, W)` e um tensor de verdade (ground truth) com a forma `(B, H, W)` contendo os índices de classe.

### 4.2. Lidando com o Desequilíbrio de Classes

Um desafio comum em conjuntos de dados de segmentação é o desequilíbrio de classes, onde a classe de fundo pode ocupar a vasta maioria dos pixels. A CrossEntropyLoss pode ser sensível a este problema, uma vez que as classes minoritárias podem contribuir de forma insignificante para a perda total, levando o modelo a negligenciá-las. Para mitigar este problema, existem funções de perda alternativas:

- **Dice Loss**: A DiceLoss foca-se na métrica de sobreposição (o coeficiente de Dice), que é menos sensível ao desequilíbrio de classes do que a CrossEntropyLoss.
- **Focal Loss**: Introduzida pela FAIR para a deteção de objetos, a FocalLoss visa resolver o problema de desequilíbrio de classes, "diminuindo o peso" de exemplos bem classificados, forçando o modelo a focar-se em exemplos "difíceis". Ela utiliza dois parâmetros-chave: gamma, que controla o grau de foco, e alpha, que equilibra o peso entre classes. A smp implementa a FocalLoss com estes parâmetros, permitindo um ajuste fino do processo de treino.

### 4.3. A Perda Combinada (Dice+CrossEntropy)

Uma recomendação prática e eficaz é a utilização de uma perda combinada, como a **DiceCELoss** disponível em algumas implementações. Esta abordagem capitaliza os pontos fortes de ambas as funções: a estabilidade de gradiente da CrossEntropyLoss e a robustez da DiceLoss contra o desequilíbrio de classes. A perda combinada resulta num treino mais eficaz e na produção de um modelo mais robusto, que otimiza simultaneamente a precisão por pixel e a sobreposição geral da máscara.

A Tabela 1 oferece uma visão comparativa das funções de perda discutidas, auxiliando na decisão de qual é a mais adequada para um problema específico.

### Tabela 1: Comparativo de Funções de Perda Comuns

| Função de Perda | Prós | Contras | Aplicação Recomendada |
|---|---|---|---|
| CrossEntropyLoss | Gradientes estáveis, treino robusto. | Sensível a classes desequilibradas, pode negligenciar classes minoritárias. | Linha de base para problemas sem desequilíbrio de classes severo. |
| DiceLoss | Robusta a desequilíbrio de classes. Mede diretamente a sobreposição. | Gradientes menos estáveis, pode levar a um treino instável. | Problemas com desequilíbrio de classes substancial (fundo vs. objeto). |
| FocalLoss | Prioriza pixels "difíceis". Eficaz com classes severamente desequilibradas. | Requer ajuste fino dos parâmetros α e γ. | Problemas com desequilíbrio de classes extremo, como deteção de objetos raros. |
| DiceCELoss (Combinada) | Combina estabilidade da CE com robustez da Dice. | Pode ser ligeiramente mais complexa de implementar de raiz. | Abordagem por defeito para a maioria dos problemas de segmentação multiclasse. |

## 5. O Ciclo de Treino e Validação

O ciclo de treino de um modelo de segmentação segue a estrutura padrão do PyTorch. Uma vez que o Dataset e o DataLoader customizados estão prontos e o modelo (smp.Unet) e a função de perda estão configurados, o treino pode começar.

### 5.1. Estrutura do Loop de Treino

O loop de treino clássico para uma época consiste nos seguintes passos, executados para cada batch de dados:

1. **model.train()**: Coloca o modelo em modo de treino, ativando camadas como Dropout e BatchNorm.
2. **forward pass**: O batch de imagens de entrada (x) é passado pelo modelo para obter o tensor de logits de saída (y_pred).
3. **Cálculo da perda**: A perda é calculada comparando os logits de saída (y_pred) com o tensor de verdade (y_true).
4. **backward pass**: loss.backward() calcula os gradientes da perda em relação aos parâmetros do modelo, propagando o erro de volta pela rede.
5. **Otimização**: optimizer.step() atualiza os pesos do modelo com base nos gradientes calculados.
6. **optimizer.zero_grad()**: Zera os gradientes para o próximo batch, evitando o acúmulo.

A cada época, um ciclo de validação deve ser executado para monitorizar a performance do modelo em dados não vistos. Para a validação, o modelo é colocado em modo de avaliação (model.eval()) e o forward pass é executado sem o cálculo de gradientes (with torch.no_grad()) para otimizar a velocidade.

### 5.2. Métricas de Avaliação

A precisão por pixel é uma métrica simples, mas o desequilíbrio de classes pode mascarar a baixa performance nas classes minoritárias. Por esta razão, métricas mais robustas são preferíveis, como:

- **Mean Intersection over Union (mIoU)**: O mIoU mede a sobreposição média entre as máscaras de previsão e de verdade para cada classe.
- **Dice Coefficient**: O coeficiente de Dice, ou F1-score, é uma medida estatística de similaridade de conjunto, também utilizada para avaliar o desempenho de segmentação e é intimamente relacionada com a DiceLoss.

O cálculo destas métricas na etapa de validação fornece uma visão mais precisa do desempenho real do modelo. A Tabela 2 apresenta o formato dos tensores em cada etapa do pipeline, um recurso útil para depuração e compreensão do fluxo de dados.

### Tabela 2: Formato dos Tensores no Pipeline

| Etapa | Descrição | Formato do Tensor | Exemplo |
|---|---|---|---|
| Imagem de Entrada | Imagem de entrada RGB. | `(B, 3, H, W)` | `(8, 3, 256, 256)` |
| Máscara de Ground Truth | Máscara de verdade com índices de classe. | `(B, H, W)` | `(8, 256, 256)` |
| Saída do Modelo (Logits) | Logits brutos de cada classe. | `(B, C, H, W)` | `(8, 4, 256, 256)` |
| Máscara de Previsão Final | Máscara de previsão com índices de classe. | `(B, H, W)` | `(8, 256, 256)` |

## 6. Inferência e Pós-processamento: Convertendo Feature Maps em Classes

O utilizador questiona especificamente como "transformar os feature maps nas classes, decidindo qual classe é a escolhida". Esta etapa é o cerne do processo de inferência e ocorre após o modelo ter sido treinado e pronto para uso. O processo de inferência converte a saída de logit do modelo numa máscara de segmentação final, pronta para visualização.

### 6.1. O Workflow de Inferência

O processo de inferência em uma nova imagem segue os seguintes passos:

1. **Carregar o modelo treinado**: O modelo salvo em disco (.pth) é carregado em memória.
2. **Pré-processar a imagem de entrada**: A nova imagem é carregada, redimensionada para as dimensões esperadas pelo modelo, e normalizada com os mesmos parâmetros utilizados no treino.
3. **Executar o forward pass**: A imagem pré-processada é alimentada ao modelo. É crucial utilizar o contexto with torch.no_grad() para desabilitar o cálculo de gradientes e otimizar a velocidade de inferência.

### 6.2. Convertendo Logits em Probabilidades: A Operação softmax

O resultado do forward pass do modelo é um tensor de logits, com a forma `(B, C, H, W)`. Estes valores brutos não são diretamente interpretáveis como probabilidades. O passo inicial para o pós-processamento é aplicar a função softmax ao longo da dimensão dos canais (classes), dim=1.

A função softmax opera pixel a pixel. Para cada pixel, ela transforma os C valores de logit em C valores de probabilidade, garantindo que a soma destas probabilidades para aquele pixel seja igual a 1. A saída da operação softmax é um novo tensor com a mesma forma `(B, C, H, W)`, mas agora contendo as probabilidades para cada classe em cada pixel. Por exemplo, se a saída for [0.1, 0.8, 0.05, 0.05] para um pixel específico, a probabilidade de ser a segunda classe é de 80%.

### 6.3. Decidindo a Classe: A Operação argmax

A operação softmax fornece as probabilidades para cada classe em cada pixel, mas o objetivo final é obter uma única classe para cada pixel da máscara de previsão. O passo seguinte é a aplicação da função **argmax**.

A função argmax (argumento do máximo) atua sobre o tensor de probabilidades resultante da softmax. Para cada pixel, ela percorre a dimensão dos canais (dim=1) e retorna o índice da classe com a probabilidade mais alta. O resultado é um tensor de um único canal com a forma `(B, H, W)`, onde cada valor de pixel é o índice da classe prevista. Este tensor representa a máscara de segmentação final, com cada pixel rotulado com o índice da classe que o modelo considerou mais provável.

### 6.4. Visualização da Máscara de Previsão

A máscara de previsão final, que é um tensor `(B, H, W)` de inteiros, não é diretamente visualizável como uma imagem colorida. Para a visualização, os índices de classe (0, 1, 2,...) devem ser mapeados de volta para as cores RGB que representam cada classe. Este processo inverte o mapeamento de cores que foi realizado durante a preparação dos dados. O resultado é uma imagem colorida que mostra a segmentação do modelo de forma visualmente intuitiva.

### 6.5. Exemplo de Código para a Construção da Máscara

A conversão da saída bruta do modelo em uma máscara de segmentação com índices de classe é uma etapa crucial na inferência. O código a seguir demonstra como realizar as operações de softmax e argmax de forma concisa e eficiente usando PyTorch:

```python
import torch
import torch.nn.functional as F

def post_process_inference(model_output_logits):
    """
    Converte a saída de logits do modelo em uma máscara de segmentação final.
    
    Args:
        model_output_logits (torch.Tensor): O tensor de saída do modelo
        com a forma (B, C, H, W).
    
    Returns:
        torch.Tensor: A máscara de previsão final com a forma (B, H, W),
        contendo os índices de classe para cada pixel.
    """
    # 1. Aplicar Softmax para converter logits em probabilidades.
    # A operação Softmax é aplicada na dimensão dos canais (dim=1).
    probabilities = F.softmax(model_output_logits, dim=1)
    
    # 2. Aplicar Argmax para obter o índice da classe com a maior probabilidade.
    # O Argmax é aplicado na mesma dimensão dos canais (dim=1).
    # O resultado é um tensor com a forma (B, H, W), onde cada valor é o índice
    # da classe predita.
    final_mask = torch.argmax(probabilities, dim=1)
    
    return final_mask

# Exemplo de uso:
# Suponha que `output` é o tensor de logits bruto retornado do modelo U-Net
# `output.shape` -> (batch_size, num_classes, height, width)
# Por exemplo: (1, 4, 256, 256)
#
# `predicted_mask` -> (1, 256, 256) com valores inteiros de 0 a 3
#
# predicted_mask = post_process_inference(output)
```

## 7. Conclusões e Recomendações

O treino de um modelo U-Net para segmentação semântica multiclasse com a biblioteca segmentation-models-pytorch é uma tarefa que, embora simplificada pela API de alto nível, exige uma compreensão aprofundada de vários aspetos do pipeline de machine learning. O sucesso depende da atenção a detalhes técnicos cruciais, desde a preparação dos dados até a fase final de inferência.

O pipeline de sucesso resume-se a:

1. **Preparação de Dados**: O ground truth para a segmentação multiclass deve ser um tensor de um único canal `(H, W)` com índices de classe inteiros. Máscaras de cores (PNG) devem ser convertidas para este formato. O uso de um Dataset customizado é a abordagem padrão para carregar dados de segmentação.

2. **Configuração do Modelo**: O modelo smp.Unet deve ser instanciado com o parâmetro classes definido para o número total de classes. É imperativo definir **activation=None** para garantir que a saída do modelo sejam logits brutos, que são esperados pela maioria das funções de perda.

3. **Escolha da Função de Perda**: A CrossEntropyLoss é uma excelente linha de base, mas para lidar com o desequilíbrio de classes, a DiceLoss, a FocalLoss ou uma perda combinada como DiceCELoss são opções superiores, oferecendo um treino mais estável e resultados mais precisos.

4. **Pós-processamento de Inferência**: O processo de conversão dos feature maps (logits) para uma máscara de previsão final é uma sequência de duas operações. Primeiro, a softmax é aplicada sobre a dimensão dos canais para obter as probabilidades por pixel. Em seguida, a argmax é utilizada para selecionar a classe com a probabilidade máxima em cada pixel, resultando na máscara de segmentação final.

A compreensão deste fluxo de trabalho e das nuances de cada passo é fundamental para alcançar um desempenho ótimo. A utilização de encoders pré-treinados, como os disponíveis no smp, representa uma otimização significativa, permitindo que o modelo aprenda de forma mais eficiente a partir de um conjunto de dados personalizado e limitado, ao mesmo tempo que mantém a flexibilidade necessária para uma ampla gama de aplicações em visão computacional.

## Referências citadas

1. Multiclass Segmentation in PyTorch using U-Net - Idiot Developer, acessado em agosto 26, 2025, https://idiotdeveloper.com/multiclass-segmentation-in-pytorch-using-u-net/
2. Image Segmentation with Pytorch - Kaggle, acessado em agosto 26, 2025, https://www.kaggle.com/code/nikhil1e9/image-segmentation-with-pytorch
3. Unet - Segmentation Models documentation, acessado em agosto 26, 2025, https://smp.readthedocs.io/en/latest/models.html
4. segmentation-models-pytorch - Read the Docs, acessado em agosto 26, 2025, https://segmentation-modelspytorch.readthedocs.io/en/latest/
5. Segmentation Models PyTorch - Kaggle, acessado em agosto 26, 2025, https://www.kaggle.com/datasets/vad13irt/segmentation-models-pytorch
6. Multiclass Segmentation · Issue #798 · qubvel-org/segmentation_models.pytorch - GitHub, acessado em agosto 26, 2025, https://github.com/qubvel/segmentation_models.pytorch/issues/798
7. Mask Shape for Multi Label Image Segmentation - PyTorch Forums, acessado em agosto 26, 2025, https://discuss.pytorch.org/t/mask-shape-for-multi-label-image-segmentation/172014
8. Multiclass Segmentation - PyTorch Forums, acessado em agosto 26, 2025, https://discuss.pytorch.org/t/multiclass-segmentation/54065
9. Ultimate Guide | Image Classification| Pytorch - Kaggle, acessado em agosto 26, 2025, https://www.kaggle.com/code/kartikpardeshi/ultimate-guide-image-classification-pytorch
10. Datasets & DataLoaders — PyTorch Tutorials 2.8.0+cu128 documentation, acessado em agosto 26, 2025, https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
11. Suggestions on loading and augmenting image masks for multi-class image segmentation in Pytorch? : r/computervision - Reddit, acessado em agosto 26, 2025, https://www.reddit.com/r/computervision/comments/ghuvr9/suggestions_on_loading_and_augmenting_image_masks/
12. qubvel-org/segmentation_models.pytorch: Semantic segmentation models with 500+ pretrained convolutional and transformer-based backbones. - GitHub, acessado em agosto 26, 2025, https://github.com/qubvel-org/segmentation_models.pytorch
13. U-Net output has 256 channels - vision - PyTorch Forums, acessado em agosto 26, 2025, https://discuss.pytorch.org/t/u-net-output-has-256-channels/161949
14. PyTorch: Multi-class segmentation loss value != 0 when using target image as the prediction, acessado em agosto 26, 2025, https://stackoverflow.com/questions/71000059/pytorch-multi-class-segmentation-loss-value-0-when-using-target-image-as-the
15. Dice-coefficient loss function vs cross-entropy - Cross Validated - Stack Exchange, acessado em agosto 26, 2025, https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy
16. Losses - Segmentation Models documentation - Read the Docs, acessado em agosto 26, 2025, https://smp.readthedocs.io/en/latest/losses.html
17. pmc.ncbi.nlm.nih.gov, acessado em agosto 26, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12225904/#:~:text=Dice%20loss%20(DL)%20is%20another,smooth%2C%20thus%20difficult%20to%20optimize.
18. Implementing Focal Loss in PyTorch for Class Imbalance | by Amit Yadav - Medium, acessado em agosto 26, 2025, https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
19. segmentation_models.pytorch/segmentation_models_pytorch/losses/focal.py at main ... - GitHub, acessado em agosto 26, 2025, https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
20. Multiclass semantic segmentation DeepLabV3 - vision - PyTorch Forums, acessado em agosto 26, 2025, https://discuss.pytorch.org/t/multiclass-semantic-segmentation-deeplabv3/124190
21. Dice Score — PyTorch-Metrics 1.8.1 documentation - Lightning AI, acessado em agosto 26, 2025, https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html
22. milesial/Pytorch-UNet: PyTorch implementation of the U-Net for image semantic segmentation with high quality images - GitHub, acessado em agosto 26, 2025, https://github.com/milesial/Pytorch-UNet
23. Semantic Segmentation Inference Using PyTorch — CV-CUDA Beta documentation, acessado em agosto 26, 2025, https://cvcuda.github.io/CV-CUDA/samples/python_samples/segmentation/segmentation_pytorch.html
24. deep learning models inference and deployment with C++(3): semantic segmentation model | by ZeonlungPun | Medium, acessado em agosto 26, 2025, https://medium.com/@zeonlungpun/deep-learning-models-inference-and-deployment-with-c-3-semantic-segmentation-model-883fd557126f
25. France1/unet-multiclass-pytorch: Multiclass semantic ... - GitHub, acessado em agosto 26, 2025, https://github.com/France1/unet-multiclass-pytorch