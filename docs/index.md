---
layout: default
title: "Deep Learning Aplicado ao Sensoriamento Remoto"
description: "Curso de Deep Learning Aplicado ao Sensoriamento Remoto para pós-graduação"
---

# Deep Learning Aplicado ao Sensoriamento Remoto

{: .concept-box .note}
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

<div class="module-grid">
    <div class="module-card">
        <h3><a href="{{ '/modulos/modulo1/' | relative_url }}">Módulo 1: Fundamentos e Contexto</a></h3>
        <p><strong>Duração:</strong> 4h</p>
        <p>Introdução ao curso, evolução da IA e fundamentos de processamento de imagens.</p>
        <ul>
            <li><a href="{{ '/modulos/modulo1/1-intro' | relative_url }}">1.1 Introdução e Evolução da IA (40min)</a></li>
            <li><a href="{{ '/modulos/modulo1/2-setup' | relative_url }}">1.2 Setup do Ambiente (10min)</a></li>
            <li><a href="{{ '/modulos/modulo1/3-revisao_matematica' | relative_url }}">1.3 Revisão Matemática (50min)</a></li>
            <li><a href="{{ '/modulos/modulo1/4-fundamentos_proc_imagens' | relative_url }}">1.4 Fundamentos de Processamento de Imagens (2h)</a></li>
        </ul>
    </div>
    
    <div class="module-card">
        <h3><a href="{{ '/modulos/modulo2/' | relative_url }}">Módulo 2: Redes Neurais - Teoria e Práticas</a></h3>
        <p><strong>Duração:</strong> 4h</p>
        <p>Fundamentos de redes neurais e primeiras implementações práticas.</p>
        <ul>
            <li><a href="{{ '/modulos/modulo2/fundamentos/' | relative_url }}">2.1 Fundamentos de Redes Neurais (2h)</a></li>
            <li><a href="{{ '/modulos/modulo2/treinamento-pytorch/' | relative_url }}">2.2 Teoria do Treinamento e PyTorch (2h)</a></li>
        </ul>
    </div>
    
    <div class="module-card">
        <h3><a href="{{ '/modulos/modulo3/' | relative_url }}">Módulo 3: Treinamento de Redes Neurais</a></h3>
        <p><strong>Duração:</strong> 4h</p>
        <p>Training loops completos e fenômenos do treinamento.</p>
        <ul>
            <li><a href="{{ '/modulos/modulo3/training-loop/' | relative_url }}">3.1 Training Loop Completo (2h)</a></li>
            <li><a href="{{ '/modulos/modulo3/fenomenos-treinamento/' | relative_url }}">3.2 Fenômenos do Treinamento (2h)</a></li>
        </ul>
    </div>
    
    <div class="module-card">
        <h3><a href="{{ '/modulos/modulo4/' | relative_url }}">Módulo 4: Introdução às CNNs</a></h3>
        <p><strong>Duração:</strong> 4h</p>
        <p>Transição de MLPs para CNNs e primeira implementação prática.</p>
        <ul>
            <li><a href="{{ '/modulos/modulo4/convolucao-classica-cnns/' | relative_url }}">4.1 Da Convolução Clássica às CNNs (2h)</a></li>
            <li><a href="{{ '/modulos/modulo4/lenet-mnist/' | relative_url }}">4.2 LeNet no MNIST (2h)</a></li>
        </ul>
    </div>
</div>

### Segunda Semana - Presencial (40 horas)

<div class="schedule-grid">
    <div class="day-card">
        <h3>Dia 1: Consolidação e Ferramentas</h3>
        <p><strong>8 horas presenciais</strong></p>
        <ul>
            <li>Manhã: Síntese e Técnicas de Regularização (4h)</li>
            <li>Tarde: Ferramentas Geoespaciais - QGIS, Rasterio, GeoPandas (4h)</li>
        </ul>
    </div>
    
    <div class="day-card">
        <h3>Dia 2: CNNs Avançadas</h3>
        <p><strong>8 horas presenciais</strong></p>
        <ul>
            <li>Manhã: VGG, ResNet, Transfer Learning (4h)</li>
            <li>Tarde: Classificação de Cenas RESIC-45 (4h)</li>
        </ul>
    </div>
    
    <div class="day-card">
        <h3>Dia 3: Segmentação Semântica</h3>
        <p><strong>8 horas presenciais</strong></p>
        <ul>
            <li>Manhã: U-Net e Dataset ISPRS Potsdam (4h)</li>
            <li>Tarde: Data Augmentation e PyTorch Lightning (4h)</li>
        </ul>
    </div>
    
    <div class="day-card">
        <h3>Dia 4: Preparação de Dados Profissional</h3>
        <p><strong>8 horas presenciais</strong></p>
        <ul>
            <li>Manhã: Dataset Custom com dados DSG (4h)</li>
            <li>Tarde: Balanceamento e Treinamento (4h)</li>
        </ul>
    </div>
    
    <div class="day-card">
        <h3>Dia 5: Estado da Arte e Projeto Final</h3>
        <p><strong>8 horas presenciais</strong></p>
        <ul>
            <li>Manhã: Segmentation Models PyTorch e DeepLab v3+ (4h)</li>
            <li>Tarde: Projeto Integrador e Apresentações (4h)</li>
        </ul>
    </div>
</div>

## Datasets Utilizados

- **[MNIST]({{ '/recursos/datasets/mnist/' | relative_url }})** - Aquecimento com LeNet
- **[RESIC-45]({{ '/recursos/datasets/resic45/' | relative_url }})** - Classificação de cenas (45 classes)
- **[ISPRS Potsdam]({{ '/recursos/datasets/isprs-potsdam/' | relative_url }})** - Segmentação urbana (6 classes)
- **[Dataset DSG Custom]({{ '/recursos/datasets/dsg-custom/' | relative_url }})** - Projeto final com dados da Diretoria de Serviço Geográfico

## Recursos Disponíveis

- **[Notebooks Jupyter]({{ '/exercicios/' | relative_url }})**: Exercícios práticos interativos
- **[Setup e Ferramentas]({{ '/recursos/setup/' | relative_url }})**: Guias de configuração do ambiente
- **[Bibliografia]({{ '/recursos/referencias/' | relative_url }})**: Referências e leituras complementares

## Competências Desenvolvidas

Ao concluir este curso, você será capaz de:

- ✅ **Processar dados geoespaciais** para Deep Learning
- ✅ **Construir pipelines completos** de treinamento
- ✅ **Implementar arquiteturas CNN** especializadas
- ✅ **Avaliar modelos** com métricas adequadas para sensoriamento remoto
- ✅ **Preparar dados reais** para produção
- ✅ **Usar ferramentas profissionais** como PyTorch Lightning e Segmentation Models

{: .concept-box .tip}
**Metodologia Híbrida**: O conteúdo EAD prepara a base teórica, enquanto as atividades presenciais focam em aplicações práticas e projetos reais.

## Próximos Passos

<div class="action-buttons">
    <a href="{{ '/modulos/modulo1/' | relative_url }}" class="btn btn-primary">Começar Módulo 1</a>
    <a href="{{ '/recursos/setup-inicial/' | relative_url }}" class="btn btn-secondary">Guia de Setup</a>
    <a href="{{ '/cronograma/' | relative_url }}" class="btn btn-secondary">Ver Cronograma Completo</a>
</div>