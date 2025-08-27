---
sidebar_position: 1
title: "Visão Geral do Módulo 7"
description: "Segmentação Semântica com FCN, SegNet, U-Net e segmentation-models-pytorch"
tags: [segmentação, fcn, segnet, unet, segmentation-models-pytorch, deep-learning, pytorch]
---

:::info Objetivo
Este módulo explora arquiteturas avançadas para segmentação semântica, incluindo Fully Convolutional Networks (FCN), SegNet, U-Net e a biblioteca segmentation-models-pytorch. Os alunos aprenderão a implementar essas arquiteturas em PyTorch e aplicá-las a tarefas de segmentação em sensoriamento remoto.
:::

# Estrutura do Módulo

## [7.1 Fully Convolutional Networks (FCN)](./fcn)

**Tópicos Principais:**
- **Introdução à Segmentação Semântica:** Diferenças entre classificação e segmentação
- **Arquitetura FCN:** Encoder-decoder e convoluções transpostas
- **Inovações:** Upscaling por meio de combinação de convoluções transpostas (deconvolução) e operações de unpooling
- **Implementação Prática:** FCN com PyTorch para segmentação de imagens

---

## [7.2 SegNet](./segnet)

**Tópicos Principais:**
- **Arquitetura Encoder-Decoder:** Uso de max-unpooling para upsampling
- **Inovações:** Armazenamento de índices de pooling para eficiência
- **Comparação com FCN:** Trade-offs entre memória e precisão
- **Implementação Prática:** Construção da SegNet em PyTorch

---

## [7.3 U-Net: Precisão em Segmentação Médica e Sensoriamento Remoto](./unet)

**Tópicos Principais:**
- **Arquitetura em Forma de U:** Encoder-decoder com conexões de salto
- **Inovações:** Concatenação de características por meio de skip connections para precisão em contornos
- **Aplicações:** Segmentação médica e mapeamento geoespacial
- **Implementação Prática:** U-Net em PyTorch com datasets personalizados

---

## [7.4 Segmentação Semântica Multiclasse com segmentation-models-pytorch](./segmentation_models_pytorch)

**Tópicos Principais:**
- **Biblioteca segmentation-models-pytorch:** Introdução e vantagens
- **Arquitetura U-Net com smp:** Uso de encoders pré-treinados e configuração do modelo
- **Preparação de Dados:** Máscaras multiclasse e datasets customizados
- **Funções de Perda:** CrossEntropyLoss, DiceLoss e perdas combinadas
- **Inferência:** Conversão de logits em máscaras finais com softmax e argmax
- **Implementação Prática:** Pipeline completo com treinamento e validação

---

## Preparação para o Projeto Final

Ao concluir este módulo, você estará preparado para:

- ✅ Implementar FCN, SegNet, U-Net e modelos com segmentation-models-pytorch em PyTorch
- ✅ Comparar arquiteturas em termos de eficiência e precisão
- ✅ Aplicar segmentação semântica a dados de sensoriamento remoto
- ✅ Avaliar modelos com métricas como IoU e Dice Score
- ✅ Integrar técnicas avançadas como data augmentation e transfer learning

:::tip Próximo Passo
O próximo módulo será dedicado à construção de dataset, onde você aplicará todo o conhecimento adquirido para resolver um problema real de segmentação semântica.
:::

## FAQ do Módulo

<details>
<summary><strong>O que é segmentação semântica?</strong></summary>
<p>Segmentação semântica é a tarefa de classificar cada pixel de uma imagem em uma categoria específica, como "estrada", "prédio" ou "vegetação".</p>
</details>

<details>
<summary><strong>Qual a principal inovação da FCN?</strong></summary>
<p>A FCN introduziu o conceito de predições densas e skip connections, permitindo a segmentação semântica de ponta a ponta.</p>
</details>

<details>
<summary><strong>O que torna a SegNet eficiente?</strong></summary>
<p>A SegNet utiliza max-unpooling com índices de pooling armazenados, reduzindo o número de parâmetros e melhorando a eficiência de memória.</p>
</details>

<details>
<summary><strong>Por que a U-Net é amplamente utilizada?</strong></summary>
<p>A U-Net é conhecida por sua precisão em contornos e sua capacidade de segmentar objetos pequenos, graças às conexões de salto que combinam características de diferentes níveis.</p>
</details>

<details>
<summary><strong>Quais as vantagens da biblioteca segmentation-models-pytorch?</strong></summary>
<p>A biblioteca segmentation-models-pytorch simplifica a implementação de arquiteturas complexas como U-Net, oferecendo suporte a encoders pré-treinados e funções de perda otimizadas para segmentação.</p>
</details>

## Navegação

**Anterior:** [Módulo 6: Arquiteturas de Segmentação Semântica](../modulo6/) 