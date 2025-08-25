---
sidebar_position: 1
title: "Visão Geral do Módulo 5"
description: "Arquiteturas CNN Avançadas e Transfer Learning"
tags: [cnn, alexnet, inception, vgg, resnet, transfer learning]
---

:::info Objetivo
Este módulo explora arquiteturas avançadas de Redes Neurais Convolucionais (CNNs) e a técnica de Transfer Learning. Os alunos aprenderão sobre as principais inovações que moldaram o campo da visão computacional, desde a AlexNet até as ResNets, e como reutilizar modelos pré-treinados para novas tarefas.
:::

# Estrutura do Módulo

**Template Exercício de Revisão** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17r1wszKcr7dXSGts6KrxTNcb5tGrxvFP?usp=sharing)

**Solução do Exercício de Revisão no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tXaLDV0L8q07rEpkdTmwRhQSRxLMx91G?usp=sharing)


## [5.1 AlexNet: O Marco Inicial das CNNs Modernas](./alexnet)

**Template do Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LtZZjLQWm7vnf33s1okd0WmvxAvTaLe5?usp=sharing)

**Solução do Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17mvatPet-R803WzN8GVoZorWzyB3UAu8?usp=sharing)

**Tópicos Principais:**
- **Contexto Histórico:** A revolução da AlexNet no ImageNet em 2012
- **Arquitetura Detalhada:** Camadas convolucionais, pooling e totalmente conectadas
- **Inovações:** ReLU, Dropout, Data Augmentation e uso de GPUs
- **Implementação Prática:** Construção da AlexNet em PyTorch
- **Impacto:** Como a AlexNet abriu caminho para arquiteturas modernas

---

## [5.2 Inception: Eficiência e Multi-Escala](./inception)

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RHc_hbUSHQHcvmK08Nk3TEgfa2dXLts3?usp=sharing)

**Tópicos Principais:**
- **GoogLeNet:** A introdução do módulo Inception e sua vitória no ImageNet 2014
- **Arquitetura Modular:** Convoluções paralelas de diferentes tamanhos
- **Inovações:** Convoluções 1x1 para redução de dimensionalidade
- **Evoluções:** Inception v2, v3 e Inception-ResNet
- **Implementação Prática:** Uso de modelos pré-treinados em PyTorch

---

## [5.3 VGG: Simplicidade e Profundidade](./vgg_family)

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qFkwYpW3LRFhp25qPB1B3Rnppm1U0fmo?usp=sharing)

**Tópicos Principais:**
- **Princípios de Design:** Uso de filtros 3x3 e aumento progressivo de profundidade
- **Arquiteturas Clássicas:** VGG-16 e VGG-19
- **Vantagens e Limitações:** Simplicidade vs. alto custo computacional
- **Implementação Prática:** Construção da VGG em PyTorch e uso de modelos pré-treinados

---

## [5.4 ResNet: Redes Residuais e Profundidade Extrema](./resnet)

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ikHEdPPz5WjaBWvLg8PrF0O0RUme7Zc6?usp=sharing)

**Tópicos Principais:**
- **Problemas Resolvidos:** Gradiente desvanecente e degradação
- **Conexões Residuais:** O conceito de aprendizado residual
- **Arquiteturas Clássicas:** ResNet-18, ResNet-50, ResNet-101 e ResNet-152
- **Variações:** Wide ResNet e ResNeXt
- **Implementação Prática:** Construção de ResNets em PyTorch e adaptação para Transfer Learning

---

## [5.5 Transfer Learning: Reutilizando Modelos Pré-Treinados](./transfer_learning)

**Exercício no Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_tMEwD-BhdNturtx3RgA6Z9wJXtYd9Bq?usp=sharing)

**Tópicos Principais:**
- **Conceito:** O que é Transfer Learning e por que é útil
- **Estratégias:** Feature Extraction vs. Fine-Tuning
- **Benefícios:** Redução de custos computacionais e eficiência de dados
- **Aplicações Práticas:** Classificação de cenas com RESIC-45
- **Implementação Prática:** Transfer Learning com PyTorch usando modelos pré-treinados

---

## Preparação para Módulos Avançados

Ao concluir este módulo, você deve estar preparado para:

- ✅ Compreender as inovações das principais arquiteturas CNNs
- ✅ Implementar AlexNet, Inception, VGG e ResNet em PyTorch
- ✅ Utilizar Transfer Learning para novas tarefas de classificação
- ✅ Comparar arquiteturas em termos de eficiência e desempenho
- ✅ Aplicar estratégias de Fine-Tuning e Feature Extraction

:::tip Próximo Passo
Os próximos módulos abordarão técnicas avançadas de segmentação semântica e preparação de dados geoespaciais.
:::

## FAQ do Módulo

<details>
<summary><strong>Por que a AlexNet foi um marco na visão computacional?</strong></summary>
<p>A AlexNet revolucionou o campo ao vencer o ImageNet 2012 com uma taxa de erro significativamente menor, introduzindo inovações como ReLU, Dropout e uso de GPUs para treinamento eficiente.</p>
</details>

<details>
<summary><strong>O que torna o módulo Inception eficiente?</strong></summary>
<p>O módulo Inception utiliza convoluções paralelas de diferentes tamanhos e convoluções 1x1 para reduzir a dimensionalidade, permitindo a extração de características em múltiplas escalas com eficiência computacional.</p>
</details>

<details>
<summary><strong>Qual a principal inovação da ResNet?</strong></summary>
<p>A ResNet introduziu conexões residuais, que permitem o treinamento de redes muito profundas ao mitigar o problema do gradiente desvanecente e facilitar o aprendizado de mapeamentos de identidade.</p>
</details>

<details>
<summary><strong>Quando usar Feature Extraction vs. Fine-Tuning no Transfer Learning?</strong></summary>
<p>Use Feature Extraction quando o dataset de destino for pequeno e similar ao dataset de origem. Use Fine-Tuning quando o dataset de destino for maior ou a tarefa for significativamente diferente.</p>
</details>

<details>
<summary><strong>Quais são os benefícios do Transfer Learning?</strong></summary>
<p>O Transfer Learning reduz custos computacionais, melhora a eficiência de dados e permite alcançar alta performance com datasets menores, reutilizando conhecimento de modelos pré-treinados.</p>
</details>

## Navegação

**Anterior:** [Módulo 4: Introdução às CNNs](../modulo4/)