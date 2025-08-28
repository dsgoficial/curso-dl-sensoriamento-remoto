---
sidebar_position: 4
title: "Inferência com Janela Deslizante"
description: "Segmentação semântica em larga escala com pytorch_toolbelt e sliding window"
tags: [janela deslizante, segmentação, pytorch, inferência]
---

# Inferência em Larga Escala para Segmentação Semântica: Um Guia Otimizado e Teórico com pytorch_toolbelt

## 1. O Problema da Inferência em Imagens de Alta Resolução e a Justificativa para a Janela Deslizante

### 1.1. As Limitações Físicas do Hardware: VRAM e o Tamanho dos Tensores CUDA

Modelos de segmentação semântica, como as redes U-Net, exigem que a imagem completa seja processada de uma só vez para gerar uma máscara de saída densa e de alta resolução. No entanto, o processamento de imagens com dimensões muito grandes, como 5000x5000 pixels ou mais, rapidamente excede a capacidade de memória disponível em GPUs comerciais, levando a erros de "memória insuficiente" (Out of Memory - OOM).¹ As limitações no tamanho máximo dos tensores CUDA, que variam de acordo com o driver e a versão da GPU, são um obstáculo fundamental.¹ Arquiteturas de redes neurais profundas com um grande número de parâmetros e camadas densas podem consumir toda a VRAM disponível até mesmo em resoluções relativamente modestas, como 1024x1024 pixels, tornando a inferência em imagens maiores uma tarefa inviável sem uma estratégia de gerenciamento de dados inteligente.¹

A técnica de janela deslizante aborda essa limitação de forma direta. Em vez de alimentar a imagem inteira para o modelo, ela a divide em tiles (pedaços) menores e sobrepostos.¹ Cada tile é pequeno o suficiente para caber na memória da GPU, garantindo que o limite superior de uso de VRAM seja previsível e controlável, independentemente do tamanho da imagem original.¹ Após a inferência de cada tile individualmente, as previsões são costuradas de volta para reconstruir a máscara de segmentação completa.

### 1.2. A Questão do Contexto e do Campo Receptivo nas Redes de Segmentação

O problema da inferência em larga escala não se resume apenas à capacidade da VRAM. As arquiteturas de redes de segmentação semântica, especialmente aquelas baseadas em Redes Neurais Convolucionais (CNNs), operam com um campo receptivo (receptive field) limitado.³ O campo receptivo de uma unidade de ativação em uma camada é a área da imagem de entrada que influencia a saída dessa unidade. Embora redes profundas possam ter campos receptivos grandes, uma imagem de 5000x5000 pixels é vasta o suficiente para que um pixel no centro de um tile não tenha seu contexto global de vizinhança na imagem completa.

O algoritmo de janela deslizante mitiga essa questão ao focar a análise em regiões menores. Ao processar cada tile de forma independente, a técnica permite que o modelo analise uma infinidade de pequenas regiões.² Isso preserva o contexto local e as relações espaciais entre os pixels adjacentes, que são cruciais para a precisão da segmentação.⁴ A técnica garante que o modelo esteja operando dentro do escopo de campo receptivo para o qual foi treinado, que geralmente se adequa melhor a imagens de tamanho intermediário.

A aplicabilidade da janela deslizante vai além da visão computacional. É uma estratégia fundamental para lidar com dados sequenciais não-estacionários ou com variabilidade, um problema análogo à "deriva de conceito" (concept drift) em fluxos de dados contínuos.⁵ Em campos como a análise de séries temporais médicas, a técnica permite a estimativa de parâmetros que variam ao longo do tempo, como taxas de transmissão de patógenos.⁶ No contexto de ressonância magnética funcional (fMRI), ela ajuda a analisar a conectividade cerebral dinâmica (dFC) em diferentes intervalos de tempo.⁷ A janela deslizante em imagens, portanto, é uma manifestação de uma estratégia de processamento de dados mais ampla, projetada para lidar com a não-estacionariedade e a variabilidade de escala, garantindo que a análise se mantenha relevante para o contexto local.

A Tabela 1 a seguir resume as vantagens e desvantagens da técnica.

**Tabela 1: Vantagens e Desvantagens da Janela Deslizante para Segmentação Semântica**

| Aspecto | Vantagens | Desvantagens |
|---------|-----------|--------------|
| **Memória** | Permite o processamento de imagens de alta resolução que não caberiam na VRAM da GPU, garantindo um uso de memória previsível.¹ | Pode consumir mais memória RAM do sistema para armazenar os tiles se não for otimizado. |
| **Precisão** | Melhora a precisão ao focar no contexto local de cada tile, o que pode ser crucial para redes com campo receptivo limitado.² | Sem uma estratégia de fusão adequada, pode gerar artefatos visuais nas bordas dos tiles (o "efeito checkerboard"). |
| **Tempo de Inferência** | Permite a inferência em lotes de tiles, o que otimiza a utilização do hardware e pode ser mais rápido que processar imagens gigantescas com modelos menos eficientes. | O tempo de inferência total é a soma do tempo de inferência de todos os tiles, o que pode ser significativamente mais lento do que uma inferência em uma única passagem para imagens menores. |
| **Complexidade** | Fornece uma solução modular e eficiente para o problema de larga escala.⁴ | A implementação completa, incluindo o fatiamento, o processamento de bordas, a fusão de resultados e as otimizações de I/O, é mais complexa do que uma única chamada de inferência.⁸ |

## 2. Fundamentos e Implementação Essencial com pytorch_toolbelt

A biblioteca pytorch_toolbelt oferece um conjunto de utilidades otimizadas para PyTorch, incluindo um pipeline pronto para inferência em imagens de grande escala. O código de referência fornecido exemplifica a abordagem recomendada, que será dissecada em seus componentes-chave.

### 2.1. Componentes-chave: ImageSlicer e a Geração de Tiles

O processo começa com a classe ImageSlicer. Ela é o orquestrador do pipeline de fatiamento. A inicialização da classe requer as dimensões da imagem de entrada (image.shape) e os parâmetros que definem o comportamento do fatiamento: tile_size e tile_step.¹

• **tile_size**: Define as dimensões da janela de processamento. No exemplo, (512, 512).
• **tile_step**: Define o passo ou o stride da janela. No exemplo, (256, 256), o que significa que as janelas se sobreporão.

A classe ImageSlicer não apenas fatia a imagem, mas também calcula automaticamente as coordenadas de cada tile (tiler.crops) e gera um tensor de peso (tiler.weight) que será utilizado para a fusão das previsões.¹ O fatiamento inteligente lida com as "casos de borda", como quando a imagem não é perfeitamente divisível pelo tamanho do tile, garantindo que todas as partes da imagem original sejam cobertas sem que a janela exceda os limites da imagem.²

### 2.2. O Ciclo de Inferência em Lotes (DataLoader e pin_memory)

Após o fatiamento, os tiles são convertidos em tensores PyTorch. O DataLoader é então utilizado para agrupar esses tensores em lotes (batch_size=8) e iterar sobre eles de forma eficiente, o que é fundamental para aproveitar a paralelização da GPU.¹

A configuração pin_memory=True no DataLoader é uma otimização de desempenho crucial.¹ Ao ativá-la, o PyTorch aloca a memória no CPU em um estado "page-locked" ou "pinned".⁹ Essa alocação especial permite uma transferência de dados direta e mais rápida da RAM da CPU para a VRAM da GPU, contornando a sobrecarga do sistema operacional associada a transferências de dados convencionais.⁹ O resultado é uma transferência de dados assíncrona e não-bloqueante, o que significa que o DataLoader pode começar a carregar o próximo lote de tiles para o GPU enquanto o modelo ainda está processando o lote atual. Isso minimiza o tempo de inatividade da GPU, um gargalo comum em pipelines de inferência intensivos em dados, especialmente quando a latência é um fator crítico.⁹

### 2.3. O Coração da Técnica: Fusão com CudaTileMerger

O CudaTileMerger é o componente mais inovador do pipeline pytorch_toolbelt. Ele é responsável por acumular as previsões de cada tile e mesclá-las no resultado final. Ao ser instanciado, ele aloca um grande buffer CUDA (merger = CudaTileMerger(...)) na VRAM, que terá o tamanho da imagem final. Em cada iteração do loop de inferência, a chamada merger.integrate_batch(pred_batch, coords_batch) utiliza as coordenadas dos tiles (coords_batch) para integrar as previsões (pred_batch) na posição correta do buffer CUDA.¹

O design do CudaTileMerger é um diferencial significativo. Ao realizar a fusão (merging) diretamente na GPU, a biblioteca evita a necessidade de transferir os resultados de cada tile de volta para a CPU para serem mesclados, e depois de volta para a GPU para outras operações. Essa transferência constante de dados entre a CPU e a GPU é um dos maiores gargalos de desempenho em pipelines de deep learning.⁹ O CudaTileMerger mantém todo o processo na VRAM, resultando em um fluxo de trabalho muito mais eficiente e rápido. O ImageSlicer em outras bibliotecas, como a image-slicer genérica, muitas vezes exigiria que o usuário implementasse a lógica de fusão manualmente, provavelmente na CPU, o que anularia as otimizações de desempenho da GPU.¹¹

A Tabela 2 apresenta uma análise detalhada do código de referência para um entendimento aprofundado de cada etapa.

**Tabela 2: Análise Detalhada do Código de Referência da pytorch_toolbelt**

| Linha de Código | Explicação da Funcionalidade | Componente Principal | Relação com os Conceitos Teóricos |
|-----------------|------------------------------|---------------------|-----------------------------------|
| `tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256))` | Inicializa o fatiador de imagem, definindo a estratégia de sliding window com sobreposição. | ImageSlicer | Configura o fatiamento da imagem grande em tiles de tamanho fixo, crucial para a gestão de VRAM.¹ |
| `merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)` | Inicializa o mesclador na GPU, alocando um buffer com base na forma (shape) alvo e na estratégia de peso predefinida. | CudaTileMerger | Prepara a área de memória da GPU para a fusão das previsões, utilizando a estratégia de ponderação para mitigar artefatos.¹ |
| `for tiles_batch, coords_batch in DataLoader(...)` | Itera sobre os tiles em lotes, otimizando o processamento em paralelo na GPU. | DataLoader | Agrega tiles em lotes para acelerar o forward pass, com pin_memory=True para transferências CPU-GPU mais rápidas.¹ |
| `pred_batch = model(tiles_batch)` | Executa a inferência para o lote de tiles atual. | model(...) | Realiza a previsão de segmentação para cada tile dentro do campo receptivo do modelo, garantindo precisão local.⁴ |
| `merger.integrate_batch(pred_batch, coords_batch)` | Acumula as previsões do lote no buffer da GPU, usando as coordenadas para posicionar cada tile corretamente. | CudaTileMerger | Realiza a fusão das previsões na própria VRAM, evitando transferências custosas entre CPU e GPU.¹ |
| `merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)` | Finaliza a fusão, converte o tensor CUDA para uma matriz NumPy e ajusta as dimensões. | merger.merge() | Converte o resultado final para um formato utilizável (NumPy), permitindo salvar ou visualizar a máscara de segmentação completa. |

## 3. Estratégias Avançadas de Fusão de Janelas (Blending)

### 3.1. A Importância da Sobreposição e do tiler.weight

O uso de tiles sobrepostos é crucial para a precisão da inferência.⁴ As previsões mais precisas de uma rede convolucional geralmente estão no centro de sua entrada, longe das bordas, onde as operações de convolução podem sofrer com efeitos de padding ou falta de contexto. A sobreposição garante que cada pixel da imagem final seja o resultado de previsões de vários tiles, onde cada pixel teve a chance de estar no centro de pelo menos uma janela.

No entanto, uma simples média das previsões sobrepostas pode introduzir artefatos visuais perceptíveis, como o "efeito checkerboard" ou "feathering".¹² O pytorch_toolbelt resolve isso utilizando um tensor de peso (tiler.weight), que é pré-calculado pelo ImageSlicer.¹ Esse tensor atribui pesos mais altos aos pixels no centro de cada tile e pesos progressivamente menores em direção às bordas. O CudaTileMerger usa esses pesos para realizar uma média ponderada das previsões¹⁴, dando maior importância às previsões do centro da janela, que são consideradas mais confiáveis.¹² Isso resulta em uma transição suave e contínua entre os tiles, eliminando os artefatos de borda.

### 3.2. Comparativo de Estratégias: Constante vs. Gaussiana

A estratégia de fusão ponderada não é exclusiva da pytorch_toolbelt. Outras bibliotecas, como a MONAI, que foca em imagens médicas 3D, oferecem controle explícito sobre o método de blending.¹⁶ A MONAI, por exemplo, permite ao usuário escolher entre dois modos principais:

• **mode="constant"**: Similar a uma média simples, onde todos os pixels sobrepostos recebem o mesmo peso. Embora simples de implementar, este método é mais propenso a artefatos visuais.

• **mode="gaussian"**: Utiliza uma função de janela gaussiana para a fusão, que por sua natureza atribui os maiores pesos ao centro do tile e pesos decrescentes em direção às bordas.¹⁶ Isso cria uma transição natural e "suave" entre os tiles.¹²

A escolha de design entre a pytorch_toolbelt e a MONAI ilustra um trade-off comum no desenvolvimento de bibliotecas. A pytorch_toolbelt oferece uma solução "pronta para usar" otimizada, com uma estratégia de peso fixo que funciona bem para a maioria dos casos de uso de visão computacional em 2D. Por outro lado, a MONAI, com sua origem em pesquisa em imagens médicas, fornece controle granular sobre a estratégia de fusão, permitindo que o pesquisador ajuste o overlap, o modo de blending e outros parâmetros de acordo com as especificidades do dataset (e.g., dados CT ou MRI, que são inerentemente 3D).¹⁶

A Tabela 3 resume essas estratégias de fusão.

**Tabela 3: Comparativo de Estratégias de Fusão (Blending) de Previsões**

| Estratégia | Descrição | Prós e Contras | Cenário de Aplicação Recomendado |
|------------|-----------|----------------|----------------------------------|
| **Constant** | Todos os pixels sobrepostos recebem o mesmo peso. | **Prós**: Simples de implementar.<br>**Contras**: Propenso a artefatos de borda visuais e transições bruscas. | Tarefas onde a precisão de borda não é crítica ou o overlap é mínimo. |
| **Gaussian** | Utiliza uma função de janela gaussiana para atribuir pesos, com os maiores pesos no centro do tile. | **Prós**: Produz transições suaves e mitiga artefatos de blending visuais.<br>**Contras**: Mais complexo computacionalmente para implementar do zero. | Tarefas onde a qualidade visual da máscara de segmentação é importante, como em imagens médicas ou de alta resolução para inspeção visual. |
| **pytorch_toolbelt** | A biblioteca gera um tensor de peso fixo (tiler.weight) que é usado pelo CudaTileMerger para uma média ponderada. | **Prós**: Solução pronta para uso, eficiente na GPU e com resultados de alta qualidade.<br>**Contras**: Menos flexível do que abordagens que permitem a escolha do modo de blending em tempo de execução. | A maioria dos projetos de inferência de segmentação em larga escala em 2D que necessitam de um pipeline otimizado e de fácil implementação. |

## 4. Otimizações de Desempenho e Gerenciamento de Memória

A inferência por janela deslizante é, por si só, uma otimização de memória fundamental. No entanto, o desempenho do pipeline pode ser aprimorado com outras práticas de gerenciamento de memória em PyTorch.

### 4.1. Otimização do Pipeline de Inferência

Durante a fase de inferência (inference), os gradientes não são necessários para o backward pass do modelo. O uso do gerenciador de contexto `with torch.no_grad():` desativa o cálculo e o armazenamento desses gradientes, o que libera uma quantidade significativa de VRAM.¹⁷ A implementação fornecida já adere a esta prática, que é essencial para minimizar o consumo de memória durante a inferência.

Além disso, é importante gerenciar o ciclo de vida dos tensores. A exclusão explícita de variáveis intermediárias não mais utilizadas com `del` pode liberar memória crucial.¹⁰ Da mesma forma, sempre que possível, deve-se preferir operações in-place, indicadas pelo sufixo de sublinhado (_), como `x.add_(y)`. Tais operações modificam o tensor existente em vez de criar um novo, evitando o inchaço desnecessário da memória.¹⁰

### 4.2. Compiladores e Otimizações de Nível de Grafo

Uma das otimizações mais significativas e modernas no ecossistema PyTorch é a introdução do torch.compile.¹⁸ Esta função é um compilador de código define-by-run que traduz o modelo para um formato otimizado, como o Triton para GPUs ou OpenMP para CPUs.¹⁸

O torch.compile reduz a sobrecarga (overhead) do interpretador Python e pode aplicar otimizações de nível de grafo que fundem múltiplas operações em um único kernel de computação.¹⁹ Ao envolver o modelo com torch.compile(model), o pipeline de inferência torna-se mais rápido e eficiente, especialmente em modelos menores ou com múltiplas chamadas de forward pass. A combinação de otimizações de I/O (pin_memory), gerenciamento de memória em tempo de execução (torch.no_grad(), del) e otimizações de compilação (torch.compile) forma um pipeline de inferência de alta performance pronto para ambientes de produção.

## 5. Alternativas e Tendências Futuras

### 5.1. Abordagens de Agregação de Contexto Arquitetural

A necessidade de capturar contexto global em imagens de larga escala pode ser abordada no nível da arquitetura do modelo, sem a necessidade de um pipeline de janela deslizante explícito. As "Redes de Pirâmide" (Pyramid Networks) são um exemplo dessa abordagem.³ Elas utilizam camadas de pooling em diferentes escalas para agregar informações de features globais, mitigando a limitação do campo receptivo em uma única passagem (forward pass).³

A "Adaptive Pyramid Context Network" (APCNet) é uma arquitetura que utiliza Módulos de Contexto Adaptativos (ACMs) para construir representações contextuais multi-escala. Ela capta informações holísticas da imagem para guiar a estimativa de features locais, buscando resolver o problema de ambiguidade causado pela análise de regiões pequenas.³

### 5.2. O Futuro: Sliding Window Attention e Outras Inovações

O conceito da janela deslizante continua relevante, mesmo em arquiteturas de ponta como os Transformers. No contexto dos Vision Transformers, onde o mecanismo de atenção full-attention pode ser computacionalmente proibitivo para imagens de alta resolução, o "Sliding Window Attention" restringe o cálculo de atenção a uma janela local de tokens.²¹ Isso reduz a complexidade quadrática do cálculo de atenção, tornando a inferência em imagens grandes mais viável, enquanto ainda preserva o contexto local.²¹

A evolução do conceito de janela deslizante em arquiteturas como LongFormer (para NLP) e o "Sliding Window Attention" (para visão) demonstra a persistência e a importância de uma estratégia de processamento de dados que balanceia a necessidade de contexto global com a viabilidade computacional.

### 5.3. Comparativo de Ferramentas e Ecossistema

A escolha da ferramenta de inferência ideal depende do caso de uso e dos requisitos do projeto. A Tabela 4 consolida o comparativo entre as principais abordagens discutidas.

**Tabela 4: Comparativo de Bibliotecas de Inferência em Larga Escala**

| Biblioteca/Ferramenta | Domínio de Foco | Suporte a GPU | Estratégia de Fusão | Nível de Abstração |
|----------------------|-----------------|---------------|---------------------|-------------------|
| **pytorch_toolbelt** | Visão computacional 2D (imagens de satélite, etc.)¹ | Completo (CPU e GPU)¹ | Média ponderada via tiler.weight (predefinida)¹ | Abstração de alto nível (Slicer/Merger)¹ |
| **MONAI** | Imagens médicas 3D¹⁶ | Completo (CPU e GPU)¹⁶ | Escolha entre constant ou gaussian (paramétrico)¹⁶ | Abstração de alto nível (Inferer e Tiler)¹⁶ |
| **image-slicer** | Processamento de imagens genérico 2D²² | Não²² | Não suportado (requer implementação manual)¹¹ | Abstração de alto nível (fatiamento e salvamento)²² |
| **torch.nn.functional.unfold/fold** | Qualquer tensor N-dimensional¹² | Completo (CPU e GPU) | Requer implementação manual da lógica de fusão¹² | Abstração de baixo nível (kernel de operação) |

## Conclusões

O relatório demonstra que a inferência por janela deslizante é uma técnica indispensável para a aplicação de modelos de segmentação semântica em imagens de alta resolução. A pytorch_toolbelt oferece uma implementação otimizada, que é notavelmente eficiente devido à sua capacidade de manter todo o pipeline de fatiamento e fusão (slicing e merging) na VRAM da GPU, um diferencial fundamental em comparação a abordagens genéricas que podem exigir transferências de dados entre CPU e GPU.

A precisão do resultado final é diretamente influenciada pela sobreposição dos tiles e pela estratégia de fusão, com a média ponderada, baseada no tiler.weight, sendo uma solução robusta para evitar artefatos de borda.

Para o desenvolvedor de um curso, as recomendações se concentram em apresentar a técnica como uma solução completa para o problema de inferência em larga escala. O curso deve cobrir os fundamentos teóricos (o motivo do sliding window e a questão do campo receptivo), a implementação prática com o pytorch_toolbelt (dissecando o código e explicando o papel de cada componente) e as otimizações de desempenho (gerenciamento de memória e compilação de código). Ao fazer isso, o curso fornecerá um conhecimento profundo e aplicável, preparando os alunos para lidar com desafios de inferência em ambientes de produção com dados de alta complexidade.

## Referências citadas

1. BloodAxe/pytorch-toolbelt: PyTorch extensions for fast R&D prototyping and Kaggle farming, acessado em agosto 26, 2025, https://github.com/BloodAxe/pytorch-toolbelt

2. What is Sliding Window in Object Detection: Complete Overview of Methods & Tools, acessado em agosto 26, 2025, https://supervisely.com/blog/how-sliding-window-improves-neural-network-models/

3. Adaptive Pyramid Context Network for Semantic Segmentation - CVF Open Access, acessado em agosto 26, 2025, https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf

4. Sliding Window Technique — reduce the complexity of your algorithm | by Data Overload, acessado em agosto 26, 2025, https://medium.com/@data-overload/sliding-window-technique-reduce-the-complexity-of-your-algorithm-5badb2cf432f

5. Designing Adaptive Algorithms Based on Reinforcement Learning for Dynamic Optimization of Sliding Window Size in Multi-Dimensional Data Streams - arXiv, acessado em agosto 26, 2025, https://arxiv.org/html/2507.06901v1

6. A sliding window approach to optimize the time-varying parameters of a spatially-explicit and stochastic model of COVID-19 - PMC, acessado em agosto 26, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10021528/

7. Sliding window functional connectivity inference with nonstationary autocorrelations and cross-correlations - PMC, acessado em agosto 26, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11212997/

8. Sliding Window In Image Processing - HeyCoach | Blogs, acessado em agosto 26, 2025, https://heycoach.in/blog/sliding-window-in-image-processing/

9. When to Set pin_memory to True in PyTorch | by Hey Amit | Data ..., acessado em agosto 26, 2025, https://medium.com/data-scientists-diary/when-to-set-pin-memory-to-true-in-pytorch-75141c0f598d

10. Optimizing Memory Usage in PyTorch Models - MachineLearningMastery.com, acessado em agosto 26, 2025, https://machinelearningmastery.com/optimizing-memory-usage-pytorch-models/

11. tiler - PyPI, acessado em agosto 26, 2025, https://pypi.org/project/tiler/0.1.2/

12. How to seemlessly blend (B x C x H x W) tensor tiles together to hide tile boundaries?, acessado em agosto 26, 2025, https://stackoverflow.com/questions/59537390/how-to-seemlessly-blend-b-x-c-x-h-x-w-tensor-tiles-together-to-hide-tile-bound

13. I saw this on my sliding door window… is this even possible?? The detail on the wings is very clear and there is even a spot where u can see the beak… is there a way to know what this was??? And if so, how is this bird not dead on my porch?!? : r/whatsthisbird - Reddit, acessado em agosto 26, 2025, https://www.reddit.com/r/whatsthisbird/comments/179zmby/i_saw_this_on_my_sliding_door_window_is_this_even/

14. Compute the weighted average in PyTorch, acessado em agosto 26, 2025, https://discuss.pytorch.org/t/compute-the-weighted-average-in-pytorch/133070

15. torcheval.metrics.functional.mean - PyTorch documentation, acessado em agosto 26, 2025, https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.mean.html

16. Inference methods — MONAI 1.5.0 Documentation - Project MONAI, acessado em agosto 26, 2025, https://docs.monai.io/en/stable/inferers.html

17. How to optimize memory usage in PyTorch? - GeeksforGeeks, acessado em agosto 26, 2025, https://www.geeksforgeeks.org/deep-learning/how-to-optimize-memory-usage-in-pytorch/

18. PyTorch 2.x, acessado em agosto 26, 2025, https://pytorch.org/get-started/pytorch-2-x/

19. PyTorch Tutorials 2.8.0+cu128 documentation, acessado em agosto 26, 2025, https://docs.pytorch.org/tutorials/

20. Volumetric Semantic Segmentation using Pyramid Context Features - UC Berkeley EECS, acessado em agosto 26, 2025, https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bioimages/bakbkm_iccv2013.pdf

21. Sliding Window Attention - GeeksforGeeks, acessado em agosto 26, 2025, https://www.geeksforgeeks.org/computer-vision/sliding-window-attention/

22. image-slicer - PyPI, acessado em agosto 26, 2025, https://pypi.org/project/image-slicer/