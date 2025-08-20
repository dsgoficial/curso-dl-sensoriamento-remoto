---
sidebar_position: 1
title: "Visão Geral do Módulo 2"
description: "Fundamentos de Redes Neurais e PyTorch"
tags: [fundamentos, perceptron, mlp, pytorch, gradientes]
---

:::info Objetivo
Este módulo estabelece os fundamentos teóricos e práticos das redes neurais, introduzindo PyTorch como framework principal. Os alunos aprenderão desde os conceitos matemáticos básicos (gradientes e otimização) até a implementação completa de um Multi-Layer Perceptron (MLP), criando uma base sólida para as arquiteturas mais avançadas dos módulos subsequentes.
:::

# Estrutura do Módulo

## [2.1 PyTorch Fundamentals](./modulo2/pytorch-fundamentals)
**Duração:** 45min

Transição do NumPy para PyTorch e introdução aos tensors como estrutura de dados fundamental.

**Tópicos Principais:**
- **Tensors vs NumPy Arrays:** Diferenças essenciais e quando usar cada um
- **Criação e Manipulação de Tensors:**
  - Tipos de dados (`dtype`) e dispositivos (`device`)
  - Operações básicas e broadcasting
  - Movimentação entre CPU e GPU
- **Conversão de Operações:** Adaptação de código NumPy para PyTorch
- **Prática no Colab:** Exercícios hands-on de manipulação de tensors

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h5UEJ4O4cGA5VY3xTczQTjjXJM15yymT?usp=sharing)

---

## [2.2 Cálculo para Deep Learning](./modulo2/calculo-dl)
**Duração:** 40min

Revisão focada dos conceitos matemáticos essenciais para compreender o funcionamento das redes neurais.

**Tópicos Principais:**
- **Derivadas e Gradientes:** Intuição visual e interpretação geométrica
- **Regra da Cadeia:** Conceito fundamental para backpropagation
- **Gradiente Descendente:**
  - Algoritmo de otimização básico
  - Taxa de aprendizagem e convergência
  - Implementação prática com visualização
- **Prática no Colab:** Visualização interativa do gradiente descendente

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UiZknApSIUskyPZoRwqQoTs96YCB08fL?usp=sharing)

**Intervalo:** 10min

---

## [2.3 O Perceptron e Redes Neurais](./modulo2/perceptron-redes)
**Duração:** 1h

Introdução aos fundamentos das redes neurais, desde o perceptron simples até conceitos avançados.

**Tópicos Principais:**
- **O Perceptron:** Modelo matemático e limitações da separabilidade linear
- **Funções de Ativação:** ReLU, Sigmoid e Tanh
  - Comparação de propriedades
  - Problema do gradiente evanescente
- **Forward e Backward Propagation:**
  - Construção do grafo computacional
  - Autograd do PyTorch
  - Acumulação de gradientes

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A5yhlyKKzm7VU3qtE3Cr9rbYYNsTXYBx?usp=sharing)

**Intervalo:** 10min

---

## [2.4 Implementação Prática: MLP Completo](./modulo2/mlp-implementacao)
**Duração:** 1h15min

Implementação completa de um Multi-Layer Perceptron, integrando todos os conceitos aprendidos.

**Tópicos Principais:**
- **Estrutura `nn.Module`:** Base para construção de redes em PyTorch
- **Dataset e DataLoader:** Gerenciamento eficiente de dados
  - Criação de datasets personalizados
  - Configuração de batches e carregamento paralelo
- **Loop de Treinamento Completo:**
  - Organização do código de treinamento
  - Modos de treinamento vs. avaliação
  - Checkpointing e salvamento de modelos
- **Otimizadores:** SGD vs. Adam e configuração de hiperparâmetros
- **Visualização com TensorBoard:** Monitoramento do treinamento
- **Exercício Prático:** MLP no dataset MNIST

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKdzzgJ7N2MmAIsgnziiLfmtgU-z79Zs?usp=sharing)

## Preparação para o Módulo 3

Ao concluir este módulo, você deve estar preparado para:

- ✅ Manipular tensors PyTorch e entender suas vantagens sobre NumPy
- ✅ Compreender os fundamentos matemáticos por trás do treinamento de redes neurais
- ✅ Implementar e treinar um MLP completo do zero
- ✅ Usar ferramentas essenciais do PyTorch (`nn.Module`, `DataLoader`, otimizadores)
- ✅ Monitorar e visualizar o processo de treinamento
- ✅ Entender os conceitos que serão expandidos para CNNs

:::tip Próximo Passo
O Módulo 3 aprofunda os tópicos do treinamento de redes neurais.
:::

## FAQ do Módulo

<details>
<summary><strong>Por que usar PyTorch em vez de implementar tudo do zero?</strong></summary>
<p>PyTorch automatiza cálculos de gradientes complexos, oferece otimizações de GPU e fornece componentes testados e otimizados. Isso permite focar na arquitetura e lógica do modelo em vez de detalhes de implementação.</p>
</details>

<details>
<summary><strong>É necessário entender profundamente o cálculo por trás das redes neurais?</strong></summary>
<p>Uma compreensão sólida dos gradientes e otimização é essencial para debugar problemas de treinamento, escolher hiperparâmetros adequados e entender por que certas técnicas funcionam.</p>
</details>

<details>
<summary><strong>Quando usar SGD vs. Adam?</strong></summary>
<p>Adam é geralmente mais robusto e converge mais rapidamente, sendo uma boa escolha padrão. SGD com momentum pode alcançar melhor generalização em alguns casos, especialmente com scheduling adequado da taxa de aprendizagem.</p>
</details>

<details>
<summary><strong>Por que começar com MLPs se vamos usar CNNs para imagens?</strong></summary>
<p>MLPs introduzem conceitos fundamentais (forward/backward pass, otimização, PyTorch) de forma mais simples. CNNs são extensões dos MLPs com operações específicas para dados espaciais.</p>
</details>

## Navegação

**Anterior:** [Módulo 1: Fundamentos e Processamento de Imagens](../modulo1/)  
**Próximo:** [Módulo 3: Treinamento de redes neurais](../modulo3/)

## Progresso do Curso

Módulo 2 de 4 (EAD) • Segundo Dia