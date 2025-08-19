---
layout: module
title: "Módulo 1: Fundamentos e Contexto"
description: "Introdução ao curso, evolução da IA e fundamentos de processamento de imagens"
duration: "4h"
order: 1
toc: true
math: true
prev_module: null
next_module: "/modulos/modulo2/"
---

# {{ page.title }}

{: .concept-box}
**Objetivo**: Estabelecer as bases do curso com contextualização histórica da IA, configuração do ambiente e fundamentos essenciais de processamento de imagens digitais.

## Visão Geral do Módulo

Este módulo introdutório prepara os alunos com os fundamentos necessários para compreender a aplicação de Deep Learning em dados de sensoriamento remoto. Começamos com uma apresentação do curso e contextualização histórica, seguida por configuração prática do ambiente e revisão de conceitos matemáticos e de processamento de imagens.

### Estrutura do Módulo

## [1.1 Introdução ao Curso e Evolução da IA no Sensoriamento Remoto]({{ '1-intro.md' | relative_url }})
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

**Recursos:**
- [Slides: Evolução da IA no SR]({{ '/recursos/slides/evolucao-ia-sr.pdf' | relative_url }})
- [Exemplos de Aplicações]({{ '/recursos/exemplos-aplicacoes/' | relative_url }})

---

## [1.2 Setup do Ambiente]({{ '2-setup.md' | relative_url }})
**Duração:** 10min

Configuração rápida e eficiente do ambiente de desenvolvimento.

**Tópicos Principais:**
- Configuração do Google Colab com ativação de GPU
- Instalação automatizada de bibliotecas e template de notebooks

**Recursos:**
- [Template de Notebook]({{ '/recursos/templates/notebook-base.ipynb' | relative_url }})

**Intervalo:** 10min

---

## [1.3 Revisão Matemática com NumPy]({{ '3-revisao_matematica.md' | relative_url }})
**Duração:** 50min

Revisão focada dos conceitos matemáticos essenciais com implementação prática.

**Tópicos Principais:**
- **Revisão de NumPy, Matplotlib e OpenCV**
- **Práticas no Colab:** Implementação hands-on de operações básicas e visualizações

**Recursos:**
- [Cheat Sheet: NumPy + Matplotlib]({{ '/recursos/cheatsheets/numpy-matplotlib/' | relative_url }})

**Intervalo:** 10min

---

## [1.4 Fundamentos de Processamento de Imagens]({{ '4-fundamentos_proc_imagens.md' | relative_url }})
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

{: .concept-box .tip}
**Próximo Passo**: O Módulo 2 introduz os fundamentos de redes neurais, construindo sobre os conceitos de processamento de imagens aprendidos aqui.

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

<div class="module-navigation">
    <div class="nav-item">
        <span class="nav-label">Anterior:</span>
        <span class="nav-disabled">Início do Curso</span>
    </div>
    <div class="nav-item">
        <span class="nav-label">Próximo:</span>
        <a href="{{ page.next_module | relative_url }}">Módulo 2: Redes Neurais - Teoria e Práticas →</a>
    </div>
</div>

## Progresso do Curso

<div class="progress-bar">
    <div class="progress-fill" style="width: 25%"></div>
</div>
<p class="progress-text">Módulo 1 de 4 (EAD) • Primeira Semana</p>