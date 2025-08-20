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
- [Visão Geral do Módulo 1](./modulos/modulo1)
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

#### Dia 1: Consolidação e Ferramentas
**8 horas presenciais**

- **Manhã:** Síntese e Técnicas de Regularização (4h)
- **Tarde:** Ferramentas Geoespaciais - QGIS, Rasterio, GeoPandas (4h)

#### Dia 2: CNNs Avançadas
**8 horas presenciais**

- **Manhã:** VGG, ResNet, Transfer Learning (4h)
- **Tarde:** Classificação de Cenas RESIC-45 (4h)

#### Dia 3: Segmentação Semântica
**8 horas presenciais**

- **Manhã:** U-Net e Dataset ISPRS Potsdam (4h)
- **Tarde:** Data Augmentation e PyTorch Lightning (4h)

#### Dia 4: Preparação de Dados Profissional
**8 horas presenciais**

- **Manhã:** Dataset Custom com dados DSG (4h)
- **Tarde:** Balanceamento e Treinamento (4h)

#### Dia 5: Estado da Arte e Projeto Final
**8 horas presenciais**

- **Manhã:** Segmentation Models PyTorch e DeepLab v3+ (4h)
- **Tarde:** Projeto Integrador e Apresentações (4h)

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

## Próximos Passos

- [🚀 Começar Módulo 1](./modulos/modulo1/)