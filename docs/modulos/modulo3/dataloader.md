---
sidebar_position: 3
title: "Dataset e Dataloader"
description: "Preparação e Carregamento de Dados usando Dataset e Dataloader em Pytorch"
tags: [treinamento, dataset, dataloader, desbalanceamento de classes]
---

# 1. Preparação e Carregamento de Dados usando Dataset e Dataloader em Pytorch

Em PyTorch, a manipulação de dados é desacoplada do código de treinamento do modelo para promover modularidade e legibilidade. Para isso, são fornecidas duas primitivas de dados principais: `torch.utils.data.Dataset` e `torch.utils.data.DataLoader`.

Um Dataset é responsável por armazenar as amostras de dados e seus rótulos correspondentes. Ele pode ser utilizado para carregar conjuntos de dados pré-existentes, como o FashionMNIST, ou para criar conjuntos de dados personalizados a partir de arquivos próprios. A biblioteca `torchvision.datasets` oferece vários datasets pré-carregados que herdam de `torch.utils.data.Dataset` e implementam funções específicas para os dados, sendo úteis para prototipagem e benchmarking.

Para criar uma classe CustomDataset (conjunto de dados personalizado), é necessário implementar três funções essenciais:

- **`__init__`**: Esta função é executada uma vez, durante a instanciação do objeto Dataset. É tipicamente utilizada para inicializar diretórios de imagens, arquivos de anotações e transformações de dados.

- **`__len__`**: Esta função deve retornar o número total de amostras no conjunto de dados.

- **`__getitem__`**: Esta função é a mais crítica, responsável por carregar e retornar uma única amostra do dataset dado um índice. Com base no índice, ela identifica a localização dos dados (por exemplo, uma imagem no disco), aplica transformações e recupera o rótulo correspondente.

O DataLoader, por sua vez, envolve o Dataset em um iterável, facilitando o acesso às amostras. Durante o treinamento de um modelo, é comum a necessidade de passar amostras em "minibatches", reembaralhar os dados a cada época para reduzir o overfitting, e utilizar multiprocessamento para acelerar a recuperação dos dados. O DataLoader abstrai essa complexidade, oferecendo uma API simples para essas operações.

# 2. Otimização de Performance do DataLoader

O desempenho do carregamento de dados pode ser um gargalo significativo no treinamento de modelos de deep learning. O PyTorch oferece várias funcionalidades no DataLoader para otimizar este processo e maximizar a utilização dos recursos computacionais disponíveis.

## 2.1. Multiprocessamento com num_workers

O parâmetro `num_workers` especifica quantos subprocessos usar para carregar os dados. Por padrão, `num_workers=0` significa que o carregamento de dados acontece no processo principal, o que pode ser ineficiente para datasets grandes ou transformações computacionalmente intensivas.

**Benefícios do Multiprocessamento:**
- Paralelização do carregamento de dados
- Overlap entre computação do modelo e preparação dos próximos batches
- Melhor utilização de CPUs multi-core

**Considerações:**
- Muito útil quando `__getitem__` envolve I/O intensivo (leitura de arquivos) ou transformações pesadas
- O número ideal de workers depende do hardware disponível e da complexidade das operações
- Regra prática: começar com `num_workers = 4 * num_GPUs` e ajustar conforme necessário
- Overhead de criação de processos pode ser contraproducente para datasets muito simples

## 2.2. Prefetch Factor

O `prefetch_factor` (disponível a partir do PyTorch 1.4) controla quantos batches são pré-carregados por cada worker. O valor padrão é 2, o que significa que cada worker mantém 2 batches prontos na memória.

**Vantagens:**
- Reduz tempo de espera entre batches
- Melhora throughput quando o modelo processa batches rapidamente
- Permite overlap mais eficiente entre carregamento e processamento

**Desvantagens:**
- Maior uso de memória (especialmente importante para imagens grandes)
- Valores muito altos podem causar out-of-memory

## 2.3. Drop Last

O parâmetro `drop_last` determina se o último batch incompleto deve ser descartado quando o tamanho do dataset não é divisível pelo batch_size.

**Quando usar `drop_last=True`:**
- Modelos sensíveis ao tamanho do batch (ex: BatchNorm em alguns casos)
- Quando batch size inconsistente pode afetar a convergência
- Durante treinamento com técnicas que dependem de batch size fixo

**Quando usar `drop_last=False` (padrão):**
- Durante avaliação/teste para não perder amostras
- Quando o modelo é robusto a variações no batch size
- Para datasets pequenos onde cada amostra é valiosa

## 2.4. Pin Memory

O `pin_memory=True` aloca tensores em memória pinned (page-locked), facilitando transferências mais rápidas entre CPU e GPU.

**Benefícios:**
- Transferências assíncronas CPU→GPU mais eficientes
- Reduz latência de movimentação de dados
- Particularmente útil para GPUs com pouca memória

**Custos:**
- Maior uso de RAM do sistema
- Pode degradar performance se muita memória for pinned

## 2.5. Persistent Workers

O `persistent_workers=True` (PyTorch 1.7+) mantém os worker processes vivos entre épocas, evitando overhead de criação/destruição.

**Vantagens:**
- Elimina tempo de startup dos workers a cada época
- Mantém caches em memória entre épocas
- Melhoria significativa para datasets com inicialização pesada

**Desvantagens:**
- Workers mantidos em memória mesmo quando não usados
- Pode complicar debugging em alguns casos

### Exemplo de Código: DataLoader Otimizado

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import time
import psutil
import os

# Dataset personalizado para demonstrar I/O intensivo
class IOIntensiveDataset(Dataset):
    def __init__(self, num_samples=1000, feature_size=1000):
        self.num_samples = num_samples
        self.feature_size = feature_size
        # Simular metadados que seriam carregados do disco
        self.metadata = [f"sample_{i}.npy" for i in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simular carregamento de arquivo (I/O intensivo)
        time.sleep(0.001)  # Simula latência de I/O
        
        # Simular transformações computacionalmente intensivas
        data = torch.randn(self.feature_size)
        
        # Operação custosa: normalização e augmentação
        data = (data - data.mean()) / (data.std() + 1e-8)
        noise = torch.randn_like(data) * 0.1
        data = data + noise
        
        label = torch.randint(0, 10, (1,)).item()
        return data, label

def benchmark_dataloader(dataloader, description, num_batches=50):
    """Benchmark para medir performance do DataLoader"""
    print(f"\n--- {description} ---")
    
    start_time = time.time()
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        # Simular processamento do modelo
        time.sleep(0.01)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Tempo médio por batch: {avg_time_per_batch:.3f}s")
    print(f"Uso de memória: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    
    return total_time

# Criar dataset
dataset = IOIntensiveDataset(num_samples=500, feature_size=1000)

# Configurações para teste
batch_size = 32
num_test_batches = 30

print("=== Benchmark de Configurações do DataLoader ===")

  Configuração básica (baseline)
dataloader_basic = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0,  # Sem multiprocessamento
    drop_last=False,
    pin_memory=False
)

time_basic = benchmark_dataloader(dataloader_basic, "Configuração Básica", num_test_batches)

# 2. Com multiprocessamento
dataloader_workers = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,  # 4 worker processes
    drop_last=False,
    pin_memory=False
)

time_workers = benchmark_dataloader(dataloader_workers, "Com 4 Workers", num_test_batches)

# 3. Configuração otimizada
dataloader_optimized = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    prefetch_factor=3,  # Prefetch 3 batches por worker
    drop_last=True,     # Descartar último batch incompleto
    pin_memory=True,    # Memória pinned para GPU
    persistent_workers=True  # Manter workers vivos
)

time_optimized = benchmark_dataloader(dataloader_optimized, "Configuração Otimizada", num_test_batches)

# 4. Comparação com diferentes números de workers
print("\n=== Teste de Escalabilidade (num_workers) ===")
worker_counts = [0, 1, 2, 4, 8]
worker_times = []

for num_workers in worker_counts:
    dataloader_test = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else 2,
        pin_memory=True if num_workers > 0 else False
    )
    
    time_taken = benchmark_dataloader(
        dataloader_test, 
        f"Workers: {num_workers}", 
        20  # Menos batches para teste rápido
    )
    worker_times.append(time_taken)

# Análise de speedup
print("\n=== Análise de Speedup ===")
baseline_time = worker_times[0]  # num_workers=0
for i, (workers, time_taken) in enumerate(zip(worker_counts, worker_times)):
    speedup = baseline_time / time_taken
    print(f"Workers: {workers:2d} | Tempo: {time_taken:5.2f}s | Speedup: {speedup:.2f}x")

print(f"\nMelhoria geral (básico → otimizado): {time_basic/time_optimized:.2f}x speedup")
```

## 2.6. Diretrizes para Otimização

**Para Datasets com I/O Intensivo:**
- Use `num_workers=4-8` como ponto de partida
- Considere `prefetch_factor=3-4` para overlap máximo
- Ative `pin_memory=True` se usando GPU
- Use `persistent_workers=True` para evitar overhead de startup

**Para Datasets Pequenos ou Simples:**
- `num_workers=0-2` pode ser suficiente
- `prefetch_factor` padrão (2) geralmente adequado
- `pin_memory=False` para economizar RAM

**Para Treinamento em Múltiplas GPUs:**
- `num_workers = 4 * num_GPUs` como regra inicial
- Monitore uso de CPU e memória para ajuste fino
- Considere `drop_last=True` para sincronização entre GPUs

**Debugging e Profiling:**
- Use `num_workers=0` durante debugging para evitar problemas de multiprocessamento
- Monitore CPU utilization e I/O wait para identificar gargalos
- Use ferramentas como `torch.profiler` para análise detalhada

# 3. Lidando com Desbalanceamento de Classes

O desbalanceamento de classes é um problema comum em machine learning, onde a distribuição das classes em um dataset é desigual. Por exemplo, em um problema de classificação binária, se 90% das amostras pertencem à classe A e apenas 10% à classe B, o modelo pode se tornar enviesado em relação à classe A. Este viés ocorre porque a função de perda é dominada pelos erros da classe majoritária, levando a um desempenho subótimo na classe minoritária, que muitas vezes é a mais crítica em aplicações do mundo real. Um modelo treinado em dados desbalanceados pode alcançar alta acurácia simplesmente prevendo a classe majoritária, mas falhar miseravelmente em identificar instâncias da classe minoritária.

Para mitigar o impacto do desbalanceamento de classes, diversas técnicas podem ser empregadas, incluindo:

### Técnicas de Reamostragem:

- **Oversampling (Sobreamostragem)**: Aumenta o número de amostras na classe minoritária, duplicando amostras existentes ou gerando novas através de aumento de dados (data augmentation). O SMOTE (Synthetic Minority Over-sampling Technique) é uma abordagem sofisticada que cria novas amostras interpolando entre exemplos da classe minoritária.

- **Undersampling (Subamostragem)**: Reduz o número de amostras na classe majoritária para equilibrar o dataset. Embora simples, pode levar à perda de informações valiosas se a classe majoritária for drasticamente reduzida.

### Outras técnicas:

- **Ponderação de Classes (Class Weighting)**: Ajusta a função de perda para penalizar mais o modelo por classificar incorretamente as classes minoritárias. Isso pode ser feito definindo o parâmetro `weight` em funções de perda como `CrossEntropyLoss`.

- **Geração de Dados Sintéticos**: Além do SMOTE, redes generativas como GANs (Generative Adversarial Networks) podem ser usadas para criar novas amostras para a classe minoritária.

É crucial aplicar essas estratégias apenas aos dados de treinamento, nunca aos dados de validação ou teste, pois isso distorceria a representação do "mundo real" e levaria a avaliações enviesadas do desempenho do modelo.

## 3.1. WeightedRandomSampler para Balanceamento de Batches

O `WeightedRandomSampler` em PyTorch é uma técnica de reamostragem que garante que cada batch processado durante o treinamento tenha uma representação balanceada de classes. É particularmente útil para datasets desbalanceados, pois ajuda a evitar que o modelo seja enviesado em relação à classe majoritária.

O `WeightedRandomSampler` funciona atribuindo um peso a cada amostra no dataset. Durante a criação do batch, amostras com pesos mais altos têm maior probabilidade de serem selecionadas, aumentando efetivamente a representação das classes minoritárias em cada batch de treinamento. Os pesos não precisam somar 1; o PyTorch os escala internamente.

#### Cálculo de Pesos:

Uma abordagem comum para calcular os pesos é atribuí-los inversamente proporcionais à frequência da classe. Isso significa que classes com menos amostras terão pesos mais altos. O cálculo envolve os seguintes passos:

1. **Contar amostras por classe**: Determine quantas amostras pertencem a cada classe única.

2. **Calcular pesos de frequência inversa**: Para cada classe, o peso é calculado como `1. / contagem_de_amostras_da_classe`. Isso garante que classes mais raras recebam pesos maiores.

3. **Atribuir pesos a amostras individuais**: Itere sobre os rótulos das amostras e atribua o peso da classe calculada a cada amostra correspondente.

#### Implementação do WeightedRandomSampler:

Após calcular o tensor `samples_weight`, o `WeightedRandomSampler` é inicializado com esses pesos e o número total de amostras a serem sorteadas. É crucial definir `replacement=True` para permitir que amostras sejam sorteadas mais de uma vez, o que é necessário para a sobreamostragem eficaz de classes minoritárias. O `num_samples` do sampler é geralmente definido como o comprimento total do dataset de treinamento (`len(dataset)`).

### Exemplo de Código: WeightedRandomSampler com Dataset Sintético

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt

#  Criar um Dataset Sintético Desbalanceado
class ImbalancedSyntheticDataset(Dataset):
    def __init__(self, num_total_samples=1000, imbalance_ratio=0.9):
        # 90% classe 0, 10% classe 1
        num_class0 = int(num_total_samples * imbalance_ratio)
        num_class1 = num_total_samples - num_class0
        
        self.data = torch.randn(num_total_samples, 10)  # 10 features
        self.targets = torch.cat((torch.zeros(num_class0), torch.ones(num_class1))).long()
        
        # Embaralhar para garantir que as classes não estejam ordenadas
        indices = torch.randperm(num_total_samples)
        self.data = self.data[indices]
        self.targets = self.targets[indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Função para plotar a distribuição das classes em batches
def plot_batch_distribution(dataloader, title):
    class_counts_in_batches = {0: [], 1: []}
    
    for _, labels in dataloader:
        unique, counts = torch.unique(labels, return_counts=True)
        counts_dict = dict(zip(unique.tolist(), counts.tolist()))
        class_counts_in_batches[0].append(counts_dict.get(0, 0))
        class_counts_in_batches[1].append(counts_dict.get(1, 0))
    
    num_batches = len(class_counts_in_batches[0])
    batch_indices = np.arange(num_batches)
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    plt.bar(batch_indices - width/2, class_counts_in_batches[0], width, label='Classe 0')
    plt.bar(batch_indices + width/2, class_counts_in_batches[1], width, label='Classe 1')
    plt.xlabel('Número do Batch')
    plt.ylabel('Contagem de Amostras')
    plt.title(title)
    plt.xticks(batch_indices, [str(i+1) for i in batch_indices])
    plt.legend()
    plt.tight_layout()
    plt.show()

# Criar dataset desbalanceado
imbalanced_dataset = ImbalancedSyntheticDataset()

# Verificar distribuição original
unique_targets, counts_targets = torch.unique(imbalanced_dataset.targets, return_counts=True)
print(f"Distribuição original do dataset: {dict(zip(unique_targets.tolist(), counts_targets.tolist()))}")

# DataLoader sem balanceamento com configuração otimizada
dataloader_unbalanced = DataLoader(
    imbalanced_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
print("\nDistribuição de batches sem WeightedRandomSampler:")
plot_batch_distribution(dataloader_unbalanced, "Distribuição de Classes por Batch (Sem Balanceamento)")

# Calcular pesos para WeightedRandomSampler
class_sample_count = np.array([len(np.where(imbalanced_dataset.targets.numpy() == t)[0]) 
                              for t in np.unique(imbalanced_dataset.targets.numpy())])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in imbalanced_dataset.targets.numpy()])
samples_weight = torch.from_numpy(samples_weight).double()  # Importante: WeightedRandomSampler espera DoubleTensor

# DataLoader com WeightedRandomSampler e configuração otimizada
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
dataloader_balanced = DataLoader(
    imbalanced_dataset, 
    batch_size=64, 
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    drop_last=True  # Útil para manter batch size consistente
)

print("\nDistribuição de batches com WeightedRandomSampler:")
plot_batch_distribution(dataloader_balanced, "Distribuição de Classes por Batch (Com WeightedRandomSampler)")
```

# 4. Exemplo Completo: Dataset e DataLoader Otimizado

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

  Definição de um CustomDataset com transformações
class CustomSyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        # Gerar dados sintéticos: features (x) e labels (y)
        self.data = torch.randn(num_samples, 10)  # 10 features
        self.targets = torch.randint(0, 2, (num_samples,))  # 2 classes (0 ou 1)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, target = self.data[idx], self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target

# 2. Carregamento de um Dataset pré-existente (FashionMNIST)
# Transformações básicas para o FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalização para [-1, 1]
])

# Dataset de treinamento e teste
train_dataset_mnist = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset_mnist = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 3. Criação de DataLoaders Otimizados
# DataLoader para o dataset sintético
synthetic_dataset = CustomSyntheticDataset(num_samples=1000)
synthetic_dataloader = DataLoader(
    synthetic_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,          # Multiprocessamento
    prefetch_factor=2,      # Prefetch 2 batches por worker
    pin_memory=True,        # Memória pinned para GPU
    drop_last=True          # Descartar último batch incompleto
)

# DataLoader para o FashionMNIST (treinamento)
train_dataloader_mnist = DataLoader(
    train_dataset_mnist, 
    batch_size=64, 
    shuffle=True,
    num_workers=4,
    prefetch_factor=3,
    pin_memory=True,
    persistent_workers=True,  # Manter workers vivos entre épocas
    drop_last=True
)

# DataLoader para o FashionMNIST (teste)
test_dataloader_mnist = DataLoader(
    test_dataset_mnist, 
    batch_size=64, 
    shuffle=False,           # Não embaralhar no teste
    num_workers=2,           # Menos workers para teste
    pin_memory=True,
    drop_last=False          # Não descartar amostras no teste
)

print(f"Número de batches no DataLoader sintético: {len(synthetic_dataloader)}")
print(f"Número de batches no DataLoader FashionMNIST (treino): {len(train_dataloader_mnist)}")
print(f"Número de batches no DataLoader FashionMNIST (teste): {len(test_dataloader_mnist)}")

# Iterando através de um DataLoader
print("\nExemplo de iteração no DataLoader sintético:")
for batch_idx, (features, labels) in enumerate(synthetic_dataloader):
    print(f"Batch {batch_idx+1}: Features shape {features.shape}, Labels shape {labels.shape}")
    if batch_idx == 2:  # Imprime 3 batches para demonstração
        break

# Demonstração de diferentes configurações do DataLoader
print("\n=== Comparação de Configurações ===")

# Configuração mínima
basic_loader = DataLoader(synthetic_dataset, batch_size=32)
print(f"Básico - Batches: {len(basic_loader)}")

# Configuração otimizada
optimized_loader = DataLoader(
    synthetic_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)
print(f"Otimizado - Batches: {len(optimized_loader)}")
```

### Exercício Prático: Comparativo de Treinamento no MNIST com e sem Balanceamento de Classes

Este exercício visa demonstrar o impacto do desbalanceamento de classes no desempenho do modelo e como o `WeightedRandomSampler` pode mitigar esse problema, além de mostrar o efeito das otimizações do DataLoader. Será utilizado o dataset MNIST, que será artificialmente desbalanceado.

**Objetivo**: Treinar um modelo simples no MNIST desbalanceado, primeiro sem balanceamento e depois com `WeightedRandomSampler`, comparando as métricas de desempenho e o tempo de treinamento com diferentes configurações do DataLoader.

**Passos**:

1. **Preparar o Dataset MNIST Desbalanceado**: Crie uma versão desbalanceada do MNIST, reduzindo drasticamente o número de amostras para algumas classes (e.g., dígitos '1' e '7').

2. **Definir Modelo e Loop de Treinamento**: Crie um modelo de classificação simples (e.g., MLP ou CNN pequena) e um loop de treinamento padrão.

3. **Treinamento Sem Balanceamento**: Treine o modelo usando um DataLoader normal no dataset desbalanceado. Avalie o desempenho em um conjunto de teste balanceado.

4. **Treinamento Com Balanceamento**: Calcule os pesos para o `WeightedRandomSampler` com base na distribuição das classes no dataset desbalanceado. Crie um DataLoader usando este sampler. Treine o mesmo modelo e avalie o desempenho.

5. **Comparação de Performance do DataLoader**: Compare diferentes configurações do DataLoader (básico vs otimizado) em termos de tempo de treinamento.

6. **Análise Comparativa**: Compare as métricas de acurácia, precisão, recall e F1-score para cada classe, observando a melhoria no desempenho das classes minoritárias com o sampler e as melhorias de performance com DataLoader otimizado.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

  Preparar o Dataset MNIST Desbalanceado
# Transformações
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalização padrão do MNIST
])

# Carregar o dataset MNIST completo
full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Criar um dataset de treinamento desbalanceado
def create_imbalanced_mnist(dataset, imbalance_classes=[1, 2], retain_ratio=0.05):
    indices = []
    imbalanced_indices = []
    
    for i in range(len(dataset)):
        label = dataset.targets[i].item()
        if label in imbalance_classes:
            # Manter apenas uma pequena fração das classes desbalanceadas
            if np.random.rand() < retain_ratio:
                imbalanced_indices.append(i)
        else:
            # Manter todas as outras classes
            indices.append(i)
    
    # Combinar índices
    final_indices = indices + imbalanced_indices
    return Subset(dataset, final_indices)

imbalanced_train_dataset = create_imbalanced_mnist(full_train_dataset, 
                                                 imbalance_classes=[1, 2], retain_ratio=0.05)

print(f"Tamanho do dataset de treinamento original: {len(full_train_dataset)}")
print(f"Tamanho do dataset de treinamento desbalanceado: {len(imbalanced_train_dataset)}")

# Verificar distribuição de classes no dataset desbalanceado
imbalanced_train_labels = [full_train_dataset.targets[i].item() for i in imbalanced_train_dataset.indices]
unique_labels, counts = np.unique(imbalanced_train_labels, return_counts=True)
print("Distribuição de classes no dataset de treinamento desbalanceado:")
for label, count in zip(unique_labels, counts):
    print(f"  Classe {label}: {count} amostras")

# 2. Definir Modelo e Loop de Treinamento
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(10 * 12 * 12, 10)  # 10 canais * 12x12 após conv e pool
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=5, description=""):
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Tempo: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Tempo total de treinamento {description}: {total_time:.2f}s")
    return total_time

def evaluate_model(model, dataloader, description=""):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"\nAvaliação {description}:")
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(all_targets, all_preds, zero_division=0))
    
    return accuracy, all_preds, all_targets

# Test DataLoader otimizado
test_dataloader = DataLoader(
    full_test_dataset, 
    batch_size=128, 
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\n" + "="*80)
print("COMPARAÇÃO DE CONFIGURAÇÕES DO DATALOADER")
print("="*80)

# --- Cenário 1: Treinamento Sem Balanceamento (DataLoader Básico) ---
print("\n--- Treinamento Sem Balanceamento (DataLoader Básico) ---")
model_unbalanced_basic = SimpleCNN().to(device)
optimizer_unbalanced_basic = optim.Adam(model_unbalanced_basic.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

dataloader_unbalanced_basic = DataLoader(
    imbalanced_train_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=0,  # Sem workers
    pin_memory=False
)

time_unbalanced_basic = train_model(
    model_unbalanced_basic, 
    dataloader_unbalanced_basic, 
    criterion, 
    optimizer_unbalanced_basic,
    description="(Sem Balanceamento, DataLoader Básico)"
)

acc_unbalanced_basic, _, _ = evaluate_model(
    model_unbalanced_basic, 
    test_dataloader, 
    "Sem Balanceamento (DataLoader Básico)"
)

# --- Cenário 2: Treinamento Sem Balanceamento (DataLoader Otimizado) ---
print("\n--- Treinamento Sem Balanceamento (DataLoader Otimizado) ---")
model_unbalanced_opt = SimpleCNN().to(device)
optimizer_unbalanced_opt = optim.Adam(model_unbalanced_opt.parameters(), lr=0.001)

dataloader_unbalanced_opt = DataLoader(
    imbalanced_train_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

time_unbalanced_opt = train_model(
    model_unbalanced_opt, 
    dataloader_unbalanced_opt, 
    criterion, 
    optimizer_unbalanced_opt,
    description="(Sem Balanceamento, DataLoader Otimizado)"
)

acc_unbalanced_opt, _, _ = evaluate_model(
    model_unbalanced_opt, 
    test_dataloader, 
    "Sem Balanceamento (DataLoader Otimizado)"
)

# --- Cenário 3: Treinamento Com Balanceamento (WeightedRandomSampler + DataLoader Otimizado) ---
print("\n--- Treinamento Com Balanceamento (WeightedRandomSampler + DataLoader Otimizado) ---")

# Calcular pesos para WeightedRandomSampler
subset_labels = np.array([full_train_dataset.targets[i].item() for i in imbalanced_train_dataset.indices])
class_counts = np.bincount(subset_labels, minlength=10)  # MNIST tem 10 classes

# Evitar divisão por zero para classes que podem não existir no subset
class_weights_raw = 1. / np.where(class_counts == 0, 1e-10, class_counts)
sample_weights = class_weights_raw[subset_labels]
samples_weight_tensor = torch.from_numpy(sample_weights).double()

sampler_balanced = WeightedRandomSampler(
    samples_weight_tensor, 
    len(samples_weight_tensor), 
    replacement=True
)

dataloader_balanced = DataLoader(
    imbalanced_train_dataset, 
    batch_size=64, 
    sampler=sampler_balanced,  # Não pode usar shuffle=True com sampler
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

model_balanced = SimpleCNN().to(device)
optimizer_balanced = optim.Adam(model_balanced.parameters(), lr=0.001)

time_balanced = train_model(
    model_balanced, 
    dataloader_balanced, 
    criterion, 
    optimizer_balanced,
    description="(Com Balanceamento, DataLoader Otimizado)"
)

acc_balanced, preds_balanced, targets_balanced = evaluate_model(
    model_balanced, 
    test_dataloader, 
    "Com Balanceamento (WeightedRandomSampler + DataLoader Otimizado)"
)

# --- Análise Final ---
print("\n" + "="*80)
print("RESUMO COMPARATIVO")
print("="*80)

print(f"Tempo de Treinamento:")
print(f"  Básico (sem workers):        {time_unbalanced_basic:.2f}s")
print(f"  Otimizado (sem balanceamento): {time_unbalanced_opt:.2f}s")
print(f"  Otimizado (com balanceamento): {time_balanced:.2f}s")

print(f"\nSpeedup do DataLoader Otimizado: {time_unbalanced_basic/time_unbalanced_opt:.2f}x")

print(f"\nAcurácia:")
print(f"  Sem Balanceamento (Básico):    {acc_unbalanced_basic:.2f}%")
print(f"  Sem Balanceamento (Otimizado): {acc_unbalanced_opt:.2f}%")
print(f"  Com Balanceamento (Otimizado): {acc_balanced:.2f}%")

# Plotar matriz de confusão para o modelo balanceado
cm = confusion_matrix(targets_balanced, preds_balanced)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Modelo com Balanceamento e DataLoader Otimizado')
plt.tight_layout()
plt.show()

print(f"\nClasses minoritárias (1 e 2) - Desempenho:")
for class_idx in [1, 2]:
    if class_idx < len(cm):
        precision = cm[class_idx, class_idx] / cm[:, class_idx].sum() if cm[:, class_idx].sum() > 0 else 0
        recall = cm[class_idx, class_idx] / cm[class_idx, :].sum() if cm[class_idx, :].sum() > 0 else 0
        print(f"  Classe {class_idx}: Precisão={precision:.3f}, Recall={recall:.3f}")
```

Este capítulo melhorado agora fornece uma visão abrangente do carregamento de dados em PyTorch, incluindo otimizações críticas de performance que podem ter um impacto significativo no tempo de treinamento, especialmente para datasets grandes ou operações de I/O intensivas.
