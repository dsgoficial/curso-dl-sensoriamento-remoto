---
sidebar_position: 1
title: "PyTorch Fundamentals"
description: "pytorch"
tags: [pytorch, nn, numpy]
---

### **Tensors vs NumPy: Diferenças Essenciais**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h5UEJ4O4cGA5VY3xTczQTjjXJM15yymT?usp=sharing)

PyTorch Tensors e NumPy Arrays são ambos containers multidimensionais para dados. No entanto, suas diferenças são fundamentais para o desenvolvimento de deep learning.

| Característica | **NumPy Arrays** | **PyTorch Tensors** |
| :--- | :--- | :--- |
| **Execução** | Exclusivamente em CPU. | **CPU e GPU**. A aceleração via GPU é crítica para o treinamento de modelos em larga escala. |
| **Gradientes** | Não possui mecanismo nativo. | **`Autograd`**: Sistema de diferenciação automática para calcular gradientes. |
| **Integração** | É a base para manipulação de dados em Python. | Integra-se nativamente com o ecossistema PyTorch (`nn.Module`, `torch.optim`). |
| **Memória** | Não há rastreamento. | Pode rastrear o "grafo computacional" para backpropagation. |

**Conversão entre Tensors e NumPy Arrays:**

A conversão entre os dois formatos é simples, mas é crucial entender o compartilhamento de memória.

```python
import torch
import numpy as np

# Criação de um NumPy array
numpy_array = np.array([1, 2, 3])
print(f"NumPy array original: {numpy_array}")

# Conversão 1: `torch.from_numpy()` (compartilha a mesma memória)
torch_from_np = torch.from_numpy(numpy_array)
print(f"Tensor criado com `from_numpy()`: {torch_from_np}")
numpy_array[0] = 99  # Modificando o array NumPy
print(f"Tensor após modificação no NumPy: {torch_from_np}") # O tensor também muda!

# Conversão 2: `torch.tensor()` (cria uma cópia)
torch_copy = torch.tensor(numpy_array)
print(f"\nTensor criado com `torch.tensor()` (cópia): {torch_copy}")
numpy_array[0] = 101 # Modificando o array NumPy novamente
print(f"Tensor da cópia após modificação no NumPy: {torch_copy}") # O tensor não é afetado

# Conversão de Tensor para NumPy
tensor_to_np = torch_from_np.numpy()
print(f"\nTensor convertido para NumPy: {tensor_to_np}")
torch_from_np[1] = 202 # Modificando o tensor
print(f"NumPy array após modificação no tensor: {tensor_to_np}") # O array NumPy também muda!
```

Use `torch.from_numpy()` ou `.numpy()` para eficiência de memória quando não houver risco de modificação inesperada. Use `torch.tensor()` quando precisar de uma cópia independente.

### **Criação e Manipulação Básica de Tensors**

**Criação de Tensors:**

Tensors podem ser criados de várias maneiras, com controle sobre seus tipos de dados (`dtype`) e dispositivos (`device`).

```python
# Criação de um tensor a partir de uma lista
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Tensor de float32: {x}")

# Tensors preenchidos com valores específicos
zeros_tensor = torch.zeros(2, 3) # Tensor 2x3 de zeros
ones_tensor = torch.ones(2, 3, dtype=torch.int16) # Tensor 2x3 de uns (inteiros)
rand_tensor = torch.randn(2, 3) # Tensor 2x3 com números aleatórios (distribuição normal)

print(f"\nTensor de zeros:\n{zeros_tensor}")
print(f"Tensor de uns:\n{ones_tensor}")
print(f"Tensor aleatório:\n{rand_tensor}")

# Propriedades de um tensor
print(f"\nShape do tensor aleatório: {rand_tensor.shape}")
print(f"Tipo de dado: {rand_tensor.dtype}")
print(f"Dispositivo (onde o tensor está armazenado): {rand_tensor.device}")
```

Para mover um tensor para a GPU, use o método `.to()`:

```python
# Verifica se a GPU está disponível
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tensor_gpu = rand_tensor.to(device)
print(f"\nTensor movido para o dispositivo: {tensor_gpu.device}")
```

**Manipulação de Tensors:**

A sintaxe de manipulação de tensors é muito similar à de NumPy.

```python
tensor = torch.arange(12).reshape(3, 4)
print(f"Tensor original:\n{tensor}")

# Indexing e slicing
print(f"\nPrimeira linha: {tensor[0]}")
print(f"Elemento na posição [1, 2]: {tensor[1, 2]}")
print(f"Todas as linhas, coluna 1: {tensor[:, 1]}")
print(f"Última coluna: {tensor[..., -1]}")

# Reshape (muda o formato do tensor, mantendo os dados)
reshaped_tensor = tensor.view(2, 6) # O mesmo que `.reshape()`
print(f"\nTensor redimensionado para 2x6:\n{reshaped_tensor}")

# Transposição (troca as dimensões)
transposed_tensor = tensor.T
print(f"\nTensor transposto:\n{transposed_tensor}")

# Operações in-place (`_` suffix) vs. out-of-place
tensor.add_(100) # Adiciona 100 ao tensor original (in-place)
print(f"\nTensor após operação `add_` (in-place):\n{tensor}")
```

### **Prática: Conversão de Operações NumPy para PyTorch**

Para facilitar a transição, vamos refazer algumas operações comuns de NumPy usando a sintaxe de PyTorch.

**Operações Matriciais:**

  * **Adição/Subtração:** `A + B` ou `torch.add(A, B)`.
  * **Multiplicação Element-wise:** `A * B` ou `torch.mul(A, B)`.
  * **Multiplicação de Matrizes:** `A @ B` ou `torch.matmul(A, B)`.

**Broadcasting:**

O broadcasting funciona de maneira similar ao NumPy, estendendo implicitamente as dimensões de um tensor menor para que se ajustem a um maior durante uma operação.

```python
# Exemplo: Adição de um vetor a uma matriz
matriz = torch.arange(9).reshape(3, 3)
vetor = torch.tensor([10, 20, 30])

soma = matriz + vetor
print(f"Matriz:\n{matriz}")
print(f"Vetor:\n{vetor}")
print(f"Resultado do broadcasting:\n{soma}")
```
