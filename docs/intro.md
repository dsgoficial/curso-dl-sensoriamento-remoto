---
sidebar_position: 1
---

# Deep Learning Aplicado ao Sensoriamento Remoto

**Bem-vindo!** Este é um curso avançado de 56 horas (16h EAD + 40h presenciais) destinado a estudantes de mestrado e doutorado, ministrado pelo **Cap Philipe Borba**.

## Sobre o Curso

Este curso oferece uma formação completa em Deep Learning com foco específico em aplicações de Sensoriamento Remoto. Através de uma abordagem hands-on, você desenvolverá competências essenciais para processar e analisar dados geoespaciais usando redes neurais profundas.

### Objetivos de Aprendizagem

- Dominar os fundamentos matemáticos do Deep Learning
- Implementar redes neurais convolucionais para análise de imagens de satélite
- Processar dados multiespectrais e geoespaciais
- Construir pipelines completos de treinamento e avaliação
- Aplicar técnicas avançadas de segmentação e classificação

## Estrutura do Curso

### Primeira Semana - EAD (16 horas)

#### Módulo 1: Fundamentos e Contexto
**Duração:** 4h

Introdução ao curso, evolução da IA e fundamentos de processamento de imagens.
- [Visão Geral do Módulo 1](./modulos/modulo1/index)
- [1.1 Introdução e Evolução da IA (40min)](./modulos/modulo1/introducao)
- [1.2 Setup do Ambiente (10min)](./modulos/modulo1/setup)
- [1.3 Revisão Matemática (50min)](./modulos/modulo1/matematica)
- [1.4 Fundamentos de Processamento de Imagens (2h)](./modulos/modulo1/processamento-imagens)

#### Módulo 2: Redes Neurais - Teoria e Práticas
**Duração:** 4h

Este módulo estabelece os fundamentos teóricos e práticos das redes neurais, introduzindo o PyTorch como framework principal. Os alunos aprenderão desde os conceitos matemáticos básicos (gradientes e otimização) até a implementação de um Multi-Layer Perceptron (MLP).

- [Visão Geral do Módulo 2](./modulos/modulo2/visao_geral)
- [2.1 PyTorch Fundamentals](./modulos/modulo2/pytorch_vs_numpy)
- [2.2 Cálculo para Deep Learning](./modulos/modulo2/calculo_dl)
- [2.3 O Perceptron e Redes Neurais](./modulos/modulo2/perceptron)

#### Módulo 3: Treinamento de Redes Neurais
**Duração:** 4h

Este módulo aprofunda as técnicas de treinamento eficiente e robusto, cobrindo a estruturação de dados, otimização de performance do DataLoader e a importância de métricas e regularização. Os alunos dominarão o ciclo completo de treinamento, incluindo funções de perda avançadas, schedulers de learning rate, e estratégias de Early Stopping e Checkpointing.

- [Visão Geral do Módulo 3](./modulos/modulo3/visao_geral)
- [3.1 Training Loop Completo](./modulos/modulo3/training_loop)
- [3.2 Dataset e DataLoader Avançado](./modulos/modulo3/dataloader)
- [3.3 Funções de Perda (Losses)](./modulos/modulo3/losses)
- [3.4 Learning Rate Schedulers](./modulos/modulo3/learning_rate_schedulers)
- [3.5 Checkpointing](./modulos/modulo3/checkpointing)
- [3.6 Avaliação e Diagnóstico do Treinamento](./modulos/modulo3/avaliacao_treinamento)
- [3.7 Técnicas de Regularização](./modulos/modulo3/regularizers)
- [3.8 Exercício Prático Integrado](./modulos/modulo3/treinamento_completo)

#### Módulo 4: Introdução às CNNs
**Duração:** 4h

Transição de MLPs para CNNs e primeira implementação prática.

- [4.1 Da Convolução Clássica às CNNs (2h)](./modulos/modulo4/convolucao-classica-cnns/)
- [4.2 LeNet no MNIST (2h)](./modulos/modulo4/lenet-mnist/)

### Segunda Semana - Presencial (40 horas)

#### Dia 1: Arquiteturas CNN Avançadas
**Duração:** 8h

Este dia é dedicado ao estudo e implementação de arquiteturas avançadas de Redes Neurais Convolucionais (CNNs), explorando desde a AlexNet até a ResNet, além de técnicas de Transfer Learning.

- [Visão Geral do Módulo 5](./modulos/modulo5/visao_geral)
- [5.1 AlexNet: O Marco Inicial das CNNs Modernas](./modulos/modulo5/alexnet)
- [5.2 Inception: Eficiência e Multi-Escala](./modulos/modulo5/inception)
- [5.3 VGG: Simplicidade e Profundidade](./modulos/modulo5/vgg_family)
- [5.4 ResNet: Redes Residuais e Profundidade Extrema](./modulos/modulo5/resnet)

#### Dia 2: CNNs Avançadas
**Duração:** 8h

- [6.1 Transfer Learning: Reutilizando Modelos Pré-Treinados](./modulos/modulo6/transfer_learning)
- [6.2 Data Augmentation](./modulos/modulo6/data_augmentation_albumentations)
- [6.3 Exercícios](./modulos/modulo6/exercicios)

#### Dia 3: Segmentação Semântica
**Duração:** 8h

- **Manhã:** Conceitos e implementação da FCN, SegNet, U-Net e do Training Loop de Segmentação Semântica (4h)
- **Tarde:** Treinamento de U-Net usando o Dataset ISPRS Potsdam (4h)

#### Dia 4: Preparação de Dados Profissional
**Duração:** 8h

- **Manhã:** PyTorch Lightning (4h)
- **Tarde:** Ferramentas Geoespaciais - QGIS, Rasterio, GeoPandas, Processamento em Paralelo (ProcessPool e ThreadPool). Criação de Dataset Custom com dados DSG, Balanceamento e Treinamento (4h)

#### Dia 5: Estado da Arte e Projeto Final
**Duração:** 8h

- **Manhã:** Segmentation Models PyTorch e DeepLab v3+ (4h)
- **Tarde:** Treinamento com dados customizados (4h)

## Datasets Utilizados

- **[MNIST](./recursos/datasets/mnist/)** - Aquecimento com LeNet
- **[RESIC-45](./recursos/datasets/resic45/)** - Classificação de cenas (45 classes)
- **[ISPRS Potsdam](./recursos/datasets/isprs-potsdam/)** - Segmentação urbana (6 classes)
- **[Dataset DSG Custom](./recursos/datasets/dsg-custom/)** - Projeto final com dados da Diretoria de Serviço Geográfico

## Recursos Disponíveis

- **[Notebooks Jupyter](./exercicios/)**: Exercícios práticos interativos
- **[Setup e Ferramentas](./recursos/setup/)**: Guias de configuração do ambiente
- **[Bibliografia](./recursos/referencias/)**: Referências e leituras complementares

## Competências Desenvolvidas

Ao concluir este curso, você será capaz de:

- ✅ **Processar dados geoespaciais** para Deep Learning
- ✅ **Construir pipelines completos** de treinamento
- ✅ **Implementar arquiteturas CNN** especializadas
- ✅ **Avaliar modelos** com métricas adequadas para sensoriamento remoto
- ✅ **Preparar dados reais** para produção
- ✅ **Usar ferramentas profissionais** como PyTorch Lightning e Segmentation Models

:::tip Metodologia Híbrida
O conteúdo EAD prepara a base teórica, enquanto as atividades presenciais focam em aplicações práticas e projetos reais.
:::

## Materiais de Apoio

Aqui estão alguns materiais adicionais para complementar seus estudos:

- **[Deep Learning Book](https://www.deeplearningbook.org)**: Um dos livros mais completos sobre Deep Learning.
- **[Dive into Deep Learning](https://d2l.ai)**: Livro interativo com exemplos práticos.
- **[Neural Networks by 3 Blue One Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&pp=0gcJCWUEOCosWNin)**: Série de vídeos explicando redes neurais de forma visual e intuitiva.
- **[Tutoriais de PyTorch](https://www.youtube.com/watch?v=2S1dgHpqCdk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz)**: Playlist oficial para aprender PyTorch.
- **[Tutoriais de PyTorch Lightning](https://www.youtube.com/watch?v=XbIN9LaQycQ&list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP)**: Playlist oficial para aprender PyTorch Lightning.

## Próximos Passos

- [🚀 Começar Módulo 1](./modulos/modulo1/index)