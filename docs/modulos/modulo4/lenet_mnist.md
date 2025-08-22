---
sidebar_position: 4
title: "Estudo de Caso Clássico: A Arquitetura LeNet-5 e o Conjunto de Dados MNIST"
description: "A Primeira CNN de Sucesso: História, Arquitetura e Implementação Prática"
tags: [lenet, mnist, yann lecun, arquitetura histórica, implementação, pytorch, treinamento]
---

# 4. Estudo de Caso Clássico: A Arquitetura LeNet-5 e o Conjunto de Dados MNIST

Nenhum estudo de caso ilustra melhor a arquitetura e a eficácia das CNNs do que a LeNet-5 de Yann LeCun. A LeNet-5 não é apenas uma arquitetura histórica; é o modelo que demonstrou a viabilidade e o potencial das CNNs, pavimentando o caminho para o campo do deep learning moderno.

## 4.1. Contexto Histórico: A Contribuição de Yann LeCun

A LeNet-5, desenvolvida por Yann LeCun, Leon Bottou, Yoshua Bengio e Patrick Haffner em 1998, foi uma das primeiras e mais bem-sucedidas Redes Neurais Convolucionais. Seu projeto foi o ápice de uma década de pesquisa e foi criado especificamente para o reconhecimento de dígitos manuscritos em cheques bancários e códigos postais. A rede não apenas provou ser uma solução prática e robusta para um problema do mundo real, mas também estabeleceu a importância histórica da arquitetura CNN no desenvolvimento do deep learning.

A LeNet-5 foi treinada e validada no dataset MNIST, um conjunto de 60.000 imagens de treinamento de dígitos manuscritos e 10.000 imagens de teste.²⁴ O MNIST é considerado o "Olá, Mundo!" dos datasets de deep learning, sendo amplamente utilizado para explicar e validar novas teorias.¹ A LeNet-5 alcançou uma notável taxa de erro de apenas 0,95% no conjunto de teste do MNIST com apenas 60.000 amostras, um resultado excepcional na época e uma prova contundente da eficácia da arquitetura.

## 4.2. A Arquitetura da LeNet-5: Análise Camada por Camada

A LeNet-5 consiste em sete camadas, incluindo duas camadas convolucionais, duas camadas de subamostragem (pooling), duas camadas totalmente conectadas e uma camada de saída. Uma forma de visualizar a arquitetura é dividi-la em duas partes principais: um "codificador convolucional" (responsável pela extração de características) e um "bloco denso" (responsável pela classificação).

O fluxo de dados através da rede é o seguinte:

1. **Entrada**: A rede recebe uma imagem em escala de cinza de 32x32 pixels com 1 canal.
2. **Camada Convolucional C1**: A primeira camada convolucional utiliza 6 filtros de 5x5. A aplicação desses filtros resulta em 6 mapas de características de 28x28.
3. **Camada de Pooling S2**: Uma camada de pooling médio de 2x2 é aplicada, reduzindo a dimensionalidade dos mapas de características para 14x14.
4. **Camada Convolucional C3**: A segunda camada convolucional utiliza 16 filtros de 5x5. A saída desta camada é um volume com 16 mapas de características de 10x10.
5. **Camada de Pooling S4**: Uma segunda camada de pooling médio de 2x2 é aplicada, diminuindo a dimensão dos mapas para 5x5.
6. **Achatamento (Flattening)**: A saída 3D da camada S4 (16 mapas de características de 5x5) é achatada em um vetor unidimensional de 400 elementos.
7. **Camada Totalmente Conectada F5**: O vetor achatado é conectado a uma camada densa com 120 saídas, que utiliza a função de ativação sigmoide.
8. **Camada Totalmente Conectada F6**: Uma segunda camada densa com 84 saídas, também com ativação sigmoide.
9. **Camada de Saída**: A camada final é uma camada totalmente conectada com 10 saídas, correspondendo às 10 classes de dígitos (0-9). A função de ativação Softmax é usada para gerar uma distribuição de probabilidade sobre as classes.

A visualização do fluxo de dados camada por camada torna tangível a forma como a LeNet-5 transforma a imagem de entrada, com pixels brutos, em uma representação de alto nível e, finalmente, em uma previsão de probabilidade.

### Tabela: Arquitetura da LeNet-5 no Dataset MNIST

| Camada | Tipo | Entrada | Kernel | Canais de Saída | Saída |
|---------|------|---------|--------|-----------------|--------|
| 1 | Convolucional (C1) | Imagem (32x32x1) | 5x5 | 6 | Mapa de características (28x28x6) |
| 2 | Pooling (S2) | Mapa de características (28x28x6) | 2x2 | 6 | Mapa de características (14x14x6) |
| 3 | Convolucional (C3) | Mapa de características (14x14x6) | 5x5 | 16 | Mapa de características (10x10x16) |
| 4 | Pooling (S4) | Mapa de características (10x10x16) | 2x2 | 16 | Mapa de características (5x5x16) |
| 5 | Achatamento | Mapa de características (5x5x16) | N/A | N/A | Vetor (400) |
| 6 | FC (F5) | Vetor (400) | N/A | N/A | Vetor (120) |
| 7 | FC (F6) | Vetor (120) | N/A | N/A | Vetor (84) |
| 8 | Saída | Vetor (84) | N/A | N/A | Vetor de probabilidades (10) |

## 4.3. Implementação e Treinamento da LeNet-5 com PyTorch

A implementação da LeNet-5 com bibliotecas como o PyTorch é uma forma prática de entender sua arquitetura. A biblioteca torch.nn é usada para criar os blocos de construção da rede, incluindo as camadas convolucionais (nn.Conv2d), as camadas de pooling (nn.AvgPool2d) e as camadas totalmente conectadas (nn.Linear). A arquitetura da LeNet-5, composta por duas camadas convolucionais, duas camadas de pooling e três camadas totalmente conectadas, pode ser definida em uma classe LeNet que herda de nn.Module.

```python
# Implementação da LeNet-5
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # Calculate the input size for the first linear layer
        # based on the output shape of the last pooling layer
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), # The input size should be 16 * 4 * 4 based on the MNIST input size of 28x28
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

A classe LeNet possui o método `__init__`, que define a arquitetura, e o método `forward`, que especifica o fluxo de dados através da rede. No forward, a entrada x passa primeiro pelo extrator de características e, em seguida, é achatada (torch.flatten) para se adequar à camada totalmente conectada, antes de passar pelo classificador final. As dimensões da entrada nn.Linear(400, 120) são calculadas a partir da saída do último bloco de pooling (16 canais x 5 de altura x 5 de largura = 400).

### Treinamento do Modelo com PyTorch

O treinamento de uma rede neural em PyTorch envolve várias etapas: preparação dos dados, definição do modelo, da função de perda e do otimizador, e a execução do loop de treinamento.

Para demonstrar a superioridade da LeNet (uma CNN) sobre um MLP, podemos treinar ambos os modelos no mesmo dataset MNIST. A CNN é consistentemente superior em todas as métricas de desempenho para problemas de visão computacional.

### Implementação do MLP para Comparação

```python
import torch.nn as nn

# Implementação do MLP
# Implementação do MLP para comparação
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # A entrada para o MLP precisa ser um vetor achatado (28x28 = 784)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

### Código de Treinamento e Comparação

```python
# Código de Treinamento e Comparação
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Preparação dos Dados
transform_lenet = transforms.Compose([transforms.ToTensor()])
transform_mlp = transforms.Compose([transforms.ToTensor()])

train_dataset_lenet = datasets.MNIST(root='./data', train=True, download=True, transform=transform_lenet)
test_dataset_lenet = datasets.MNIST(root='./data', train=False, download=True, transform=transform_lenet)
train_dataset_mlp = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mlp)
test_dataset_mlp = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mlp)

batch_size = 64
train_loader_lenet = DataLoader(train_dataset_lenet, batch_size=batch_size, shuffle=True)
test_loader_lenet = DataLoader(test_dataset_lenet, batch_size=batch_size, shuffle=False)
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)
test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=batch_size, shuffle=False)

# 2. Funções de Treinamento e Teste
def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5, patience=3):
    model.train()
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Evaluate on test set for early stopping
        accuracy = evaluate_model(model, test_loader, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            # Optional: Save the best model state
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            print(f"Early stopping patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve == patience:
            print("Early stopping triggered!")
            break

    print(f"Finished Training. Best accuracy: {best_accuracy:.2f}%")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Acurácia no conjunto de teste: {accuracy:.2f}%')
    return accuracy

# 3. Comparação de Desempenho
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando {device} para o treinamento")

# LeNet Training
print("\n--- Treinando LeNet ---")
lenet = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=0.001)
train_model(lenet, train_loader_lenet, test_loader_lenet, optimizer, criterion, device, num_epochs=100)
lenet_accuracy = evaluate_model(lenet, test_loader_lenet, device)

# MLP Training
print("\n--- Treinando MLP ---")
mlp = MLP().to(device)
criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.001)
train_model(mlp, train_loader_mlp, test_loader_mlp, optimizer_mlp, criterion_mlp, device, num_epochs=100)
mlp_accuracy = evaluate_model(mlp, test_loader_mlp, device)

print(f"\nResultados Finais: LeNet = {lenet_accuracy:.2f}%, MLP = {mlp_accuracy:.2f}%")
```

Após o treinamento, espera-se que a CNN (LeNet) alcance uma acurácia superior à do MLP, pois a CNN foi projetada especificamente para explorar a estrutura espacial das imagens, ao contrário do MLP que as processa como vetores planos, perdendo a correlação entre os pixels vizinhos. A CNN também tem um número de parâmetros drasticamente reduzido devido ao compartilhamento de pesos, o que a torna mais eficiente e com menor risco de overfitting do que o MLP para tarefas de visão computacional.