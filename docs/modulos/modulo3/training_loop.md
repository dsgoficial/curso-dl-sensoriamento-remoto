---
sidebar_position: 2
title: "Training Loop Completo: Integrando Tudo"
description: "Fundamentos de Redes Neurais e PyTorch"
tags: [treinamento, loss, forward pass, backward pass, backpropagation, optimizer]
---

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16AOqOIHJng2-atZz86RZyZU0wezIJJHk?usp=sharing)


# **Anatomia do Training Loop**

## **Estrutura fundamental: forward, loss, backward, step**

Um loop de treinamento completo segue quatro passos essenciais para cada batch de dados:

1.  **Forward Pass**: Propagar um batch de dados através da rede neural para obter as predições.
2.  **Loss Calculation**: Comparar as predições com os alvos (verdades) usando uma função de perda.
3.  **Backward Pass**: Calcular os gradientes da perda em relação a todos os parâmetros do modelo usando `loss.backward()`.
4.  **Optimizer Step**: Atualizar os pesos do modelo usando o otimizador com base nos gradientes calculados.

Essa sequência se repete para cada batch dentro de cada **epoch** (uma passada completa pelo dataset).

## **Dataset e DataLoader para MNIST**

O **MNIST** é um dataset clássico de 70.000 imagens 28x28 de dígitos manuscritos (0 a 9). A biblioteca `torchvision` oferece uma maneira fácil de carregar este e outros datasets.

O **`torch.utils.data.DataLoader`** é um wrapper fundamental que:

  * Divide o dataset em **mini-batches** de um tamanho específico (`batch_size`).
  * **Embaralha** a ordem dos dados a cada epoch (`shuffle=True`).
  * Permite o carregamento paralelo dos dados (`num_workers`).

<!-- end list -->

```python
import torchvision
from torch.utils.data import DataLoader

# Define as transformações para as imagens (converte para tensor e normaliza)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Carrega o dataset de treino e teste
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Cria os DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Exemplo de iteração através do DataLoader
for images, labels in train_loader:
    print(f"Shape do batch de imagens: {images.shape}")
    print(f"Shape do batch de labels: {labels.shape}")
    break # Para na primeira iteração para mostrar o shape
```

## **Modo train vs eval: quando e por que**

O PyTorch permite que certas camadas se comportem de forma diferente durante o treino e a inferência. Por exemplo, camadas como **`Dropout`** (que desativa neurônios aleatoriamente) e **`Batch Normalization`** (que normaliza a saída de uma camada) têm um comportamento diferente.

  * `model.train()`: Habilita o modo de treino.
  * `model.eval()`: Habilita o modo de avaliação/inferência.

É uma boa prática sempre chamar `model.train()` antes do loop de treinamento e `model.eval()` antes do loop de validação/teste para garantir o comportamento correto do modelo.

## **Implementação guiada: Primeiro training loop funcional**

Vamos juntar todos os conceitos em um primeiro training loop funcional para o dataset MNIST.

```python
# Importa o modelo e os módulos necessários
# Supondo que a classe MLP e os DataLoaders já foram definidos acima
import torch.optim as optim

# Hiperparâmetros
epochs = 5
learning_rate = 0.01

# Modelo, otimizador e loss
model = MLP(input_size=28*28, hidden_size=128, output_size=10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Loop de treinamento
model.train() # Coloca o modelo em modo de treino

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Ajusta o formato da entrada e move para o device
        data = data.view(data.size(0), -1) # Flatten image
        data, target = data.to(device), target.to(device)

        # 2. Zera os gradientes
        optimizer.zero_grad()

        # 3. Forward pass
        output = model(data)

        # 4. Calcula a loss
        loss = loss_function(output, target)

        # 5. Backward pass e otimização
        loss.backward()
        optimizer.step()

        # Acompanhamento do progresso
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("\nTreinamento concluído!")
```
