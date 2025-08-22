---
sidebar_position: 5
title: "Por que CNNs são Superiores a MLPs para Visão Computacional?"
description: "Análise Comparativa da Eficiência e Performance das Arquiteturas"
tags: [cnn vs mlp, eficiência, parâmetros compartilhados, correlação espacial, performance]
---

# 5. Por que CNNs são Superiores a MLPs para Visão Computacional?

A superioridade das CNNs sobre os MLPs para tarefas de visão computacional não é apenas teórica, mas também demonstrada empiricamente por meio de uma análise comparativa de desempenho. As diferenças arquitetônicas fundamentais se traduzem em resultados significativamente melhores em cenários práticos.

## 5.1. A Raiz da Superioridade: Extração de Características e Eficiência de Parâmetros

O MLP, com sua dependência de entradas achatadas, perde a correlação espacial entre os pixels, uma informação crítica para a interpretação de imagens.¹ Além disso, a arquitetura totalmente conectada leva à explosão de parâmetros, tornando a rede impraticável para dados de alta dimensão.⁴

A CNN, por outro lado, foi projetada para resolver esses problemas. A arquitetura convolucional possui uma fase de aprendizado de características embutida, que ocorre através das camadas convolucionais e de pooling antes da camada de classificação totalmente conectada.²⁸ Os pesos compartilhados dos filtros garantem que a rede aprenda a detectar padrões importantes, como bordas e texturas, em qualquer lugar da imagem, com um número de parâmetros drasticamente reduzido.⁴ Essa eficiência de parâmetros permite que as CNNs sejam muito mais profundas e poderosas do que os MLPs, enquanto mantêm a capacidade de generalização e evitam a maldição da dimensionalidade.⁴
