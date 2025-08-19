### **Módulo 4: PyTorch: A Base para Modelos de Deep Learning**

-----

### **4.3 Implementação com PyTorch (2h30min)**


```

#### **Autograd: Diferenciação Automática Prática** (50min)

###### **O que é requires\_grad e quando usar** (15min)

O **Autograd** é o motor de diferenciação automática do PyTorch. Ele constrói um "grafo computacional" para rastrear as operações sobre os tensors, permitindo o cálculo eficiente dos gradientes via backpropagation.

Para que um tensor participe deste rastreamento, ele deve ter a propriedade `requires_grad=True`.

```python
# Tensor que não rastreia gradientes por padrão
x = torch.tensor([2.0], requires_grad=False)
y = x**2 # y também não rastreia
print(f"Tensor y.requires_grad: {y.requires_grad}")

# Tensor que rastreia gradientes
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
z = w * x + b # z agora rastreia, pois um de seus inputs (w, b) rastreia
print(f"Tensor z.requires_grad: {z.requires_grad}")
```

Em um modelo de deep learning, somente os **parâmetros do modelo** (pesos e vieses) devem ter `requires_grad=True`. Os dados de entrada e as labels não precisam, pois não serão otimizados diretamente.

###### **Método .backward() e acúmulo de gradientes** (15min)

O método `.backward()` é o gatilho para a backpropagation. Ele calcula os gradientes da saída de um grafo computacional em relação a todos os seus *nós folha* que possuem `requires_grad=True`. Para que `.backward()` funcione, a saída deve ser um **escalar** ou, no caso de um tensor, você deve fornecer um vetor de mesmo tamanho (`grad_tensors`) para a multiplicação da cadeia.

O PyTorch, por padrão, **acumula gradientes** na propriedade `.grad` dos tensors. Isso significa que, se você chamar `.backward()` várias vezes, os novos gradientes serão *somados* aos gradientes existentes.

```python
import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# Primeira chamada a backward()
z1 = 2 * x + 3 * y
z1.backward()
print(f"Gradiente de x após a 1ª chamada: {x.grad}") # Esperado: 2.0
print(f"Gradiente de y após a 1ª chamada: {y.grad}") # Esperado: 3.0

# Segunda chamada a backward() (sem zerar os gradientes)
z2 = 4 * x + 5 * y
z2.backward()
print(f"\nGradiente de x após a 2ª chamada: {x.grad}") # Esperado: 2.0 + 4.0 = 6.0
print(f"Gradiente de y após a 2ª chamada: {y.grad}") # Esperado: 3.0 + 5.0 = 8.0
```

Esta acumulação é um problema no treinamento de deep learning, pois os gradientes do batch atual seriam contaminados pelos gradientes do batch anterior. A solução é **zerar os gradientes** explicitamente em cada iteração de treinamento. Usamos `optimizer.zero_grad()` para isso, que zera os gradientes de todos os parâmetros do modelo.

###### **Prática intensiva: Gradientes em funções simples** (15min)

Vamos calcular gradientes manualmente e verificar com o PyTorch.

**Exemplo 1: Função Quadrática Simples**

Seja a função $f(x) = x^2$. O gradiente (derivada) é $\\frac{df}{dx} = 2x$.
Para $x=3$, o gradiente é 6.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2

y.backward()
print(f"Gradiente de y em relação a x: {x.grad}")
```

**Exemplo 2: Função com Múltiplas Variáveis**

Seja a função $f(x, y) = 2x^2 + 3y^3$.
Os gradientes parciais são $\\frac{\\partial f}{\\partial x} = 4x$ e $\\frac{\\partial f}{\\partial y} = 9y^2$.
Para $x=2$ e $y=4$, os gradientes são $4(2) = 8$ e $9(4^2) = 144$.

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = 2 * x**2 + 3 * y**3

z.backward()
print(f"Gradiente de z em relação a x: {x.grad}")
print(f"Gradiente de z em relação a y: {y.grad}")
```

###### **torch.no\_grad() para validação** (5min)

Durante a fase de validação ou teste, não precisamos calcular gradientes, pois não estamos otimizando o modelo. O contexto **`with torch.no_grad():`** desabilita o mecanismo de rastreamento de gradientes, o que economiza memória e acelera a computação.

```python
model = ... # suponha que temos um modelo aqui
model.eval() # coloca o modelo em modo de avaliação

with torch.no_grad():
    # Código de validação aqui
    # Todas as operações dentro deste bloco não rastreiam gradientes
```

É uma boa prática usar `torch.no_grad()` durante a inferência e avaliação para garantir que os gradientes não sejam calculados e para economizar recursos.

-----

#### **4.4 Estrutura nn.Module e Otimizadores** (55min)

###### **Construção de MLPs com PyTorch** (20min)

A classe **`torch.nn.Module`** é a classe base para todos os módulos de redes neurais em PyTorch. Ela gerencia o estado do módulo e permite que você defina camadas e implemente a lógica de forward pass.

Para construir uma rede neural, você deve:

1.  Herdar de `nn.Module`.
2.  Definir as camadas (e.g., `nn.Linear`, `nn.Conv2d`) no construtor `__init__()`.
3.  Implementar o método `forward()`, que define a sequência de operações que os dados realizarão.

PyTorch rastreia automaticamente os parâmetros de todos os `nn.Module` definidos como atributos.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Definindo as camadas aqui
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # Função de ativação
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Implementando o forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Exemplo de uso
input_size = 28 * 28 # Para imagens MNIST 28x28
hidden_size = 128
output_size = 10 # 10 classes (dígitos de 0 a 9)

model = MLP(input_size, hidden_size, output_size)
print(model)
```

###### **Funções de Perda (Loss Functions)** (15min)

As funções de perda medem a diferença entre a saída do modelo e os valores de verdade.

  * **`nn.CrossEntropyLoss` (Classificação):**
    Esta função é utilizada para problemas de classificação multiclasse. Ela combina a função de ativação **softmax** e a **Negative Log-Likelihood (NLL)** em uma única etapa. A fórmula matemática é:

    $\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})

    Onde $N$ é o número de amostras, $C$ o número de classes, $y_{i,c}$ é 1 se a amostra $i$ pertence à classe $c$ (e 0 caso contrário), e $\hat{y}_{i,c}$ é a probabilidade prevista da amostra $i$ pertencer à classe $c$.
    É crucial que esta função receba como entrada os **logits** (saídas brutas da última camada) e não as probabilidades normalizadas. As labels devem ser **índices inteiros** da classe correta.

  * **`nn.MSELoss` (Regressão):**
    Calcula o Erro Quadrático Médio. É ideal para problemas de regressão. A fórmula é:

    $\text{Loss} = \frac{1}{N} \sum\_{i=1}^{N} (y_i - \hat{y}\_i)^2$
    
    Onde $y_i$ é o valor real e $\hat{y}\_i$ é o valor previsto.

<!-- end list -->

```python
# Exemplo de uso de CrossEntropyLoss
loss_function_ce = nn.CrossEntropyLoss()
logits = torch.randn(1, 10) # Logits de um batch de 1 e 10 classes
labels = torch.tensor([3]) # Label da classe 3
loss_ce = loss_function_ce(logits, labels)
print(f"Loss com CrossEntropyLoss: {loss_ce}")

# Exemplo de uso de MSELoss
loss_function_mse = nn.MSELoss()
preds = torch.randn(5, 1) # 5 predições de um problema de regressão
targets = torch.randn(5, 1) # 5 valores alvos
loss_mse = loss_function_mse(preds, targets)
print(f"Loss com MSELoss: {loss_mse}")
```

###### **Otimizadores: SGD e Adam** (20min)

O otimizador é responsável por atualizar os parâmetros do modelo com base nos gradientes calculados na backpropagation.

  * **`torch.optim.SGD` (Stochastic Gradient Descent):**
    É a implementação mais básica. A regra de atualização de pesos é:

    $w_{t+1} = w_t - \eta \cdot \nabla J(w_t)$, onde $w_t$ são os pesos no tempo $t$, $\eta$ é o **learning rate** (`lr`) e $\nabla J(w_t)$ é o gradiente da função de perda.
    Com **momentum** e **weight decay (regularização L2)**, a regra de atualização se torna:

    $v\_t = \mu \cdot v_{t-1} + \nabla J(w_t) + \lambda \cdot w_t \\
    w_{t+1} = w_t - \eta \cdot v_t$, onde $\mu$ é o momentum e $\lambda$ é o weight decay.

  * **`torch.optim.Adam` (Adaptive Moment Estimation):**
    É um otimizador adaptativo que ajusta a taxa de aprendizado para cada parâmetro individualmente. Geralmente, funciona bem com configurações padrão.

**Uso prático dos otimizadores:**

1.  Instancie o otimizador, passando os parâmetros do modelo (`model.parameters()`) e o `learning rate`.
2.  Antes do forward pass de cada batch, chame `optimizer.zero_grad()` para zerar os gradientes acumulados.
3.  Após a backpropagation (`loss.backward()`), chame `optimizer.step()` para atualizar os pesos.

<!-- end list -->

```python
import torch.optim as optim

# Otimizador SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Otimizador Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# Otimizadores são usados no training loop da seguinte forma:
# optimizer.zero_grad()
# ... forward pass, loss calculation ...
# loss.backward()
# optimizer.step()
```

-----

### **4.5 Training Loop Completo: Integrando Tudo (1h45min)**

#### **Anatomia do Training Loop** (1h5min)

###### **Estrutura fundamental: forward, loss, backward, step** (25min)

Um loop de treinamento completo segue quatro passos essenciais para cada batch de dados:

1.  **Forward Pass**: Propagar um batch de dados através da rede neural para obter as predições.
2.  **Loss Calculation**: Comparar as predições com os alvos (verdades) usando uma função de perda.
3.  **Backward Pass**: Calcular os gradientes da perda em relação a todos os parâmetros do modelo usando `loss.backward()`.
4.  **Optimizer Step**: Atualizar os pesos do modelo usando o otimizador com base nos gradientes calculados.

Essa sequência se repete para cada batch dentro de cada **epoch** (uma passada completa pelo dataset).

###### **Dataset e DataLoader para MNIST** (15min)

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

###### **Modo train vs eval: quando e por que** (10min)

O PyTorch permite que certas camadas se comportem de forma diferente durante o treino e a inferência. Por exemplo, camadas como **`Dropout`** (que desativa neurônios aleatoriamente) e **`Batch Normalization`** (que normaliza a saída de uma camada) têm um comportamento diferente.

  * `model.train()`: Habilita o modo de treino.
  * `model.eval()`: Habilita o modo de avaliação/inferência.

É uma boa prática sempre chamar `model.train()` antes do loop de treinamento e `model.eval()` antes do loop de validação/teste para garantir o comportamento correto do modelo.

###### **Implementação guiada: Primeiro training loop funcional** (15min)

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

#### **Monitoramento e Debugging** (40min)

###### **Interpretação de loss curves e hiperparâmetros** (15min)

A análise da **curva de perda** (loss curve) é a principal forma de diagnosticar o treinamento.

  * **Learning Rate (LR) muito alto:** A perda pode oscilar erraticamente ou divergir para infinito. O modelo "pula" o mínimo da função de perda.
  * **Learning Rate (LR) muito baixo:** A convergência é muito lenta. A perda diminui gradualmente, mas pode levar muito tempo para alcançar um bom resultado.
  * **Batch size pequeno:** A curva de perda tende a ser mais ruidosa (errática), mas pode ajudar na generalização.
  * **Epochs insuficientes (Underfitting):** A perda de treino ainda está decrescendo. O modelo não aprendeu o suficiente e precisa de mais treino.
  * **Epochs excessivas (Overfitting):** O gap entre a perda de treino e a perda de validação começa a aumentar. O modelo está memorizando os dados de treino e não generaliza bem.

###### **TensorBoard básico para visualização** (15min)

O **TensorBoard** é uma ferramenta de visualização para experimentos de machine learning. Ele permite rastrear métricas como perda e acurácia ao longo do tempo.

```python
from torch.utils.tensorboard import SummaryWriter

# Cria um SummaryWriter para registrar os dados
writer = SummaryWriter('runs/mnist_mlp_experiment_1')

# Exemplo de uso no training loop
# ...
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... forward, backward, step
        if batch_idx % 100 == 0:
            loss = loss_function(output, target)
            # Loga a perda no TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
# ...
writer.close()
```

Para visualizar, basta executar `tensorboard --logdir=runs` no terminal.

###### **Problemas comuns e como resolvê-los** (10min)

  * **Gradientes não sendo calculados:**

      * **Sintomas:** `tensor.grad` é `None` ou o modelo não aprende.
      * **Solução:** Verifique se o tensor tem `requires_grad=True`, se `loss.backward()` foi chamado, ou se você está em um contexto `with torch.no_grad()`.

  * **Gradientes explodindo:**

      * **Sintomas:** Perda se torna `NaN` ou infinita.
      * **Solução:** Reduza o learning rate. Use `nn.utils.clip_grad_norm_()` para "cortar" os gradientes.

  * **Gradientes desaparecendo:**

      * **Sintomas:** Os gradientes se tornam muito pequenos, a perda não muda.
      * **Solução:** Use funções de ativação como `ReLU`, `LeakyReLU`. Use otimizadores adaptativos como Adam. Utilize skip connections em redes muito profundas.

  * **Loss não mudando:**

      * **Sintomas:** A perda permanece constante após várias épocas.
      * **Solução:** Verifique se o learning rate é muito baixo ou se há um bug no código (ex: `optimizer.step()` não foi chamado). Certifique-se de que os dados de entrada e as labels estão corretos.

  * **Erros de memória (`CUDA out of memory`):**

      * **Sintomas:** O programa falha com uma exceção de memória da GPU.
      * **Solução:** Reduza o `batch_size`. Use `del` para remover variáveis grandes que não são mais necessárias.