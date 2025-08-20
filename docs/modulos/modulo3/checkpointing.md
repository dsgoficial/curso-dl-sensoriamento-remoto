---
sidebar_position: 6
title: "Checkpointing"
description: "Salvar e Restaurar Pesos (Checkpointing)"
tags: [state_dict, torch.save(), checkpoint]
---

# 1. Salvar e Restaurar Pesos (Checkpointing)

Checkpointing é o processo de salvar o estado de um modelo durante o treinamento. É uma prática fundamental para garantir a resiliência e replicabilidade do processo de treinamento, permitindo retomar o treinamento em caso de interrupções inesperadas ou planejar sessões de treinamento em múltiplas etapas. Além disso, permite salvar versões do modelo em pontos chave para posterior avaliação e seleção do melhor desempenho.

## Como Salvar state_dict do Modelo e Otimizador:

Em PyTorch, os parâmetros aprendíveis (pesos e vieses) de um modelo `torch.nn.Module` são armazenados em um dicionário de estado interno chamado `state_dict`. Para salvar apenas os parâmetros aprendidos do modelo para inferência, é recomendado salvar o `state_dict` usando `torch.save()`. Esta abordagem oferece a maior flexibilidade para restaurar o modelo posteriormente.

Para salvar um checkpoint geral, que pode ser usado tanto para inferência quanto para retomar o treinamento, é crucial salvar mais do que apenas o `state_dict` do modelo. O `state_dict` do otimizador também deve ser salvo, pois contém buffers e parâmetros que são atualizados durante o processo de treinamento. Informações adicionais, como o número da época atual e a perda mais recente, também podem ser incluídas para uma restauração precisa. Uma convenção comum em PyTorch é usar a extensão de arquivo `.tar` para esses checkpoints.

## Exemplo de Código: Salvando um Checkpoint Completo

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definir um modelo simples (exemplo)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Exemplo: 10 features de entrada, 1 saída
    
    def forward(self, x):
        return self.fc(x)

# Instanciar modelo e otimizador
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Simular estado de treinamento
epoch = 5
loss = 0.1234

# Definir caminho do checkpoint
checkpoint_path = 'checkpoint_completo.pth.tar'

# Salvar o checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # Adicionar mais metadados de treinamento se necessário
}

torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint salvo em: {checkpoint_path}")
```

## Como Restaurar Pesos para Retomar Treinamento:

Para retomar o treinamento a partir de um checkpoint, o processo envolve carregar o dicionário salvo e aplicar os `state_dict` ao modelo e ao otimizador. É fundamental inicializar o modelo e o otimizador com a mesma arquitetura e tipo antes de carregar seus estados. Após carregar os estados, o modelo deve ser definido para o modo de treinamento (`model.train()`) para garantir que camadas como Dropout e Batch Normalization se comportem corretamente para o aprendizado contínuo. O treinamento pode então ser continuado a partir da época salva.

## Exemplo de Código: Restaurando para Retomar Treinamento

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Definir a mesma arquitetura do modelo
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Caminho do checkpoint
checkpoint_path = 'checkpoint_completo.pth.tar'

# 1. Inicializar modelo e otimizador (com a mesma arquitetura/tipo)
model_resumed = SimpleModel()
optimizer_resumed = optim.SGD(model_resumed.parameters(), lr=0.01)  # LR pode ser diferente, mas o tipo deve ser o mesmo

# 2. Carregar o checkpoint
checkpoint = torch.load(checkpoint_path)

#  Aplicar os state_dict e recuperar metadados
model_resumed.load_state_dict(checkpoint['model_state_dict'])
optimizer_resumed.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Começar da próxima época
loaded_loss = checkpoint['loss']

print(f"Modelo restaurado da época {start_epoch-1} com perda: {loaded_loss:.4f}")

# 4. Definir o modelo para modo de treinamento para continuar o aprendizado
model_resumed.train()

# Exemplo de como continuar o loop de treinamento
# for epoch in range(start_epoch, num_total_epochs):
#     # Código de treinamento aqui...
#     print(f"Continuando treinamento na época {epoch}")
```

## Como Restaurar Pesos para Inferência:

Para usar um modelo salvo para inferência (fazer predições em novos dados), o processo é semelhante, mas com uma etapa crucial adicional: o modelo deve ser definido para o modo de avaliação (`model.eval()`). Este comando garante que camadas como Dropout e Batch Normalization se comportem de forma consistente para inferência, desativando o Dropout e utilizando estatísticas populacionais (médias e variâncias aprendidas durante o treinamento) para Batch Normalization, em vez de estatísticas do batch atual. A falha em chamar `model.eval()` pode levar a resultados de inferência inconsistentes.

## Exemplo de Código: Restaurando para Inferência

```python
import torch
import torch.nn as nn

# Definir a mesma arquitetura do modelo
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Caminho do checkpoint (pode ser apenas o state_dict do modelo)
model_weights_path = 'model_weights_for_inference.pth'  # Ou o checkpoint_completo.pth.tar

# 1. Salvar apenas os pesos do modelo (recomendado para inferência)
model_inference_save = SimpleModel()
torch.save(model_inference_save.state_dict(), model_weights_path)
print(f"Pesos do modelo salvos para inferência em: {model_weights_path}")

# 2. Inicializar o modelo
model_for_inference = SimpleModel()

#  Carregar o state_dict
# Se o arquivo for um checkpoint completo, extraia apenas o model_state_dict
# checkpoint = torch.load(model_weights_path)
# model_for_inference.load_state_dict(checkpoint['model_state_dict'])

model_for_inference.load_state_dict(torch.load(model_weights_path))

# 4. Definir o modelo para modo de avaliação
model_for_inference.eval()

# 5. Realizar inferência (com torch.no_grad() ou torch.inference_mode())
dummy_input = torch.randn(1, 10)  # Exemplo de entrada

with torch.no_grad():  # Desativa o cálculo de gradientes para economizar memória e acelerar
    prediction = model_for_inference(dummy_input)

print(f"Predição em modo de inferência: {prediction.item():.4f}")
```

# 2. Inferência Eficiente

A inferência, ou o processo de fazer predições com um modelo treinado, requer práticas específicas para garantir eficiência e acurácia. As duas práticas mais importantes em PyTorch são o uso de `model.eval()` e `torch.no_grad()` (ou `torch.inference_mode()` para versões mais recentes).

## model.eval():

Esta função muda o modelo do modo de treinamento para o modo de avaliação. Isso é crucial porque certas camadas, como Dropout e Batch Normalization, se comportam de maneira diferente durante o treinamento para auxiliar na generalização e prevenir o overfitting. No modo de treinamento, Dropout desativa neurônios aleatoriamente e Batch Normalization usa estatísticas do batch atual. No modo de avaliação, Dropout é desativado (todos os neurônios são mantidos) e Batch Normalization usa estatísticas populacionais aprendidas (média e variância acumuladas durante o treinamento), garantindo resultados consistentes.

## torch.no_grad() / torch.inference_mode():

`torch.no_grad()` é um gerenciador de contexto que desativa o cálculo de gradientes dentro de seu bloco. Durante a inferência, não há necessidade de atualizar os pesos do modelo, portanto, computar e armazenar gradientes é desnecessário e consome memória e tempo computacional. `torch.inference_mode()` é uma alternativa mais recente e potencialmente mais rápida para o mesmo propósito. A combinação de `model.eval()` com `torch.no_grad()` (ou `torch.inference_mode()`) é essencial para uma inferência otimizada, garantindo velocidade e menor consumo de memória.

## Exemplo de Código: Inferência Otimizada

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

# Carregar um modelo pré-treinado (exemplo)
model = resnet50(pretrained=True)

# Criar um tensor de entrada dummy
dummy_input = torch.randn(1, 3, 224, 224)  # Batch de 1 imagem, 3 canais, 224x224

# 1. Definir o modelo para o modo de avaliação
model.eval()

# 2. Usar o gerenciador de contexto para desativar o cálculo de gradientes
with torch.inference_mode():  # Ou with torch.no_grad(): para versões mais antigas
    output = model(dummy_input)

print("Saída da inferência otimizada (sem gradientes, em modo eval):")
print(output.shape)
```
