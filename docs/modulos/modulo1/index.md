---
sidebar_position: 1
title: "Visão Geral do Módulo 1"
description: "Introdução ao curso, evolução da IA e fundamentos de processamento de imagens"
tags: [fundamentos, IA, processamento-imagens]
---

:::info Objetivo
Este módulo introdutório prepara os alunos com os fundamentos necessários para compreender a aplicação de Deep Learning em dados de sensoriamento remoto. Começamos com uma apresentação do curso e contextualização histórica, seguida por configuração prática do ambiente e revisão de conceitos matemáticos e de processamento de imagens.
:::

# Estrutura do Módulo

## [1.1 Introdução ao Curso e Evolução da IA no Sensoriamento Remoto](./modulo1/introducao)
**Duração:** 40min

Uma visão abrangente do curso e da evolução da IA aplicada ao sensoriamento remoto.

**Tópicos Principais:**
- **Apresentação do curso e metodologia** - Interação com alunos para colher expectativas
- **Contextualização e Evolução da IA** - Das origens até o Deep Learning atual
- **Aplicações de DL ao Sensoriamento Remoto:**
  - Detecção de objetos
  - Detecção de mudanças
  - Superresolução
  - Transferência de estilo
  - Classificação de cena

---

## [1.2 Setup do Ambiente](./modulo1/setup)
**Duração:** 10min

Configuração rápida e eficiente do ambiente de desenvolvimento.

**Tópicos Principais:**
- Configuração do Google Colab com ativação de GPU
- Instalação automatizada de bibliotecas e template de notebooks

**Intervalo:** 10min

---

## [1.3 Revisão Matemática com NumPy](./modulo1/matematica)
**Duração:** 50min

Revisão focada dos conceitos matemáticos essenciais com implementação prática.

**Tópicos Principais:**
- **Revisão de NumPy, Matplotlib e OpenCV**
- **Práticas no Colab:** Implementação hands-on de operações básicas e visualizações

**Intervalo:** 10min

---

## [1.4 Fundamentos de Processamento de Imagens](./modulo1/processamento-imagens)
**Duração:** 2h

Base sólida em processamento digital de imagens usando NumPy.

**Tópicos Principais:**
- **Conceitos Básicos:**
  - Imagens como matrizes NumPy (grayscale, RGB, multiespectral)
  - Tipos de dados e conversões
  - Operações básicas
- **Filtros e Convolução:**
  - Filtros espaciais clássicos (média, Gaussiano, Sobel, Laplaciano)
  - Implementação de convolução 2D
  - Conceitos de sliding window e padding

## Preparação para o Módulo 2

Ao concluir este módulo, você deve estar preparado para:

- ✅ Compreender o contexto histórico e aplicações do Deep Learning em SR
- ✅ Configurar e usar eficientemente o ambiente Google Colab
- ✅ Manipular imagens como arrays NumPy
- ✅ Implementar filtros básicos de processamento de imagens
- ✅ Compreender os conceitos de convolução que serão expandidos para CNNs

:::tip Próximo Passo
O Módulo 2 introduz os fundamentos de redes neurais, construindo sobre os conceitos de processamento de imagens aprendidos aqui.
:::

## FAQ do Módulo

<details>
<summary><strong>Por que começamos com processamento clássico se vamos usar Deep Learning?</strong></summary>
<p>Compreender filtros clássicos é essencial para entender o que as CNNs aprendem automaticamente. Os conceitos de convolução são fundamentais em ambas as abordagens.</p>
</details>

<details>
<summary><strong>Preciso conhecer OpenCV profundamente?</strong></summary>
<p>Não é necessário domínio avançado. Usamos OpenCV principalmente para operações básicas de I/O e algumas transformações específicas.</p>
</details>

<details>
<summary><strong>As técnicas clássicas ainda são relevantes na era do Deep Learning?</strong></summary>
<p>Sim! Elas são usadas em pré-processamento, análise exploratória de dados e para compreender o que as redes neurais estão "aprendendo".</p>
</details>

## Navegação

**Próximo:** [Módulo 2: Redes Neurais - Teoria e Práticas](../modulo2/)

## Progresso do Curso

Módulo 1 de 4 (EAD) • Primeira Semana