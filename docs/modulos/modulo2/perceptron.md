---
sidebar_position: 3
title: "O Perceptron: Modelo Matemático e Intuição"
description: "Perceptron"
tags: [perceptron, mlp]
---

## Fundamentos de Redes Neurais Artificiais

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A5yhlyKKzm7VU3qtE3Cr9rbYYNsTXYBx?usp=sharing)

**O Perceptron: Modelo Matemático e Intuição**

O **Perceptron** é reconhecido como o tipo mais simples de rede neural e um algoritmo de aprendizado supervisionado projetado para classificadores binários. Sua concepção, proposta por Frank Rosenblatt em 1957, marcou um marco inicial na área de redes neurais. A intuição por trás do Perceptron deriva de um modelo simplificado de um neurônio biológico, que recebe múltiplas entradas, as processa e produz uma única saída.

O **modelo matemático** de um Perceptron é essencialmente uma função de limiar (threshold function) que mapeia um vetor de entrada de valores reais para uma saída binária (tipicamente 0 ou 1). Os componentes chave incluem:

* **Entradas (x):** Um vetor de características de entrada, onde cada `x_i` representa um dado de entrada.
* **Pesos (w):** Um vetor de pesos reais, onde cada `w_i` é multiplicado pela entrada correspondente `x_i`. Os pesos determinam a influência de cada entrada na saída.
* **Bias (b):** Um termo de bias (`w_0` ou `b`) é adicionado à soma ponderada. Ele atua como um ajuste na fronteira de decisão, permitindo que o neurônio ative mesmo quando todas as entradas são zero, ou que não ative mesmo com entradas positivas.
* **Soma Ponderada (Net Sum):** O neurônio calcula a soma ponderada de suas entradas, somando os produtos das entradas pelos seus respectivos pesos e adicionando o bias:

```
z = b + Σ(i=1 to n) w_i * x_i
```

Em notação vetorial, isso é `z = w · x + b`, onde `w` é o vetor de pesos e `x` é o vetor de entradas.

* **Função de Ativação (Step Function):** A soma ponderada `z` é então passada por uma função de ativação, para inserir uma não-linearidade.

![Perceptron. Extraído de https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf#page=21.00](/img/perceptron_mit.png)

A **fronteira de decisão** de um Perceptron de camada única é linear. Isso significa que ele tenta separar as classes de dados por uma única linha (em 2D) ou um hiperplano (em dimensões superiores). O algoritmo de treinamento do Perceptron ajusta os pesos e o bias para encontrar essa linha ótima que separa as duas classes.

Apesar de sua importância histórica, os Perceptrons de camada única possuem uma **limitação fundamental**: são capazes de aprender e classificar apenas padrões linearmente separáveis. Isso significa que se os dados de entrada não puderem ser divididos por uma única linha (ou hiperplano em dimensões superiores), como no famoso problema XOR, um Perceptron simples não conseguirá classificá-los corretamente.

A limitação do Perceptron à linearidade é severa e foi a causa direta para a necessidade de redes mais complexas, como as Perceptrons Multicamadas (MLPs), que incorporam camadas ocultas e, crucialmente, funções de ativação não-lineares. A "Revolução do Deep Learning" só foi possível com a superação dessas limitações através da arquitetura de redes multicamadas e da introdução de não-linearidades. A discussão do Perceptron e suas limitações é, portanto, fundamental para justificar a evolução para as Redes Multicamadas (MLPs) e a introdução das funções de ativação não-lineares, que são os próximos tópicos, mostrando aos alunos o "porquê" da complexidade crescente das redes neurais.

**2.1.2. Funções de Ativação: ReLU, Sigmoid e Tanh**

As funções de ativação são componentes essenciais nas redes neurais, introduzindo a não-linearidade necessária para que os modelos possam aprender mapeamentos complexos entre entradas e saídas. Elas são aplicadas após a soma ponderada das entradas de um neurônio, transformando o sinal antes de passá-lo para a próxima camada. Sem funções de ativação não-lineares, uma rede neural, independentemente do número de camadas, se comportaria como um modelo linear simples, incapaz de aprender relações complexas nos dados. As três funções de ativação mais comuns são Sigmoid, Tanh e ReLU, cada uma com suas propriedades e aplicações específicas.

A **Função Sigmoid**, também conhecida como função logística, transforma um valor real `x` para o intervalo (0, 1). Sua definição matemática é:

```
F(x) = 1/(1 + e^(-x))
```

Possui um formato em "S" e foi amplamente utilizada em redes neurais iniciais. No entanto, a Sigmoid é suscetível ao problema do **gradiente evanescente (vanishing gradient problem)**, onde as derivadas se tornam muito pequenas para valores de entrada extremos, dificultando o aprendizado em redes profundas. Sua derivada, `F'(x) = F(x)(1-F(x))`, atinge um máximo de 0.25 em `x=0` e se aproxima de zero à medida que `x` se afasta de zero. Isso significa que, para valores de entrada muito grandes ou muito pequenos, o gradiente é quase zero, o que impede a atualização eficaz dos pesos nas camadas anteriores durante a backpropagation.

A **Função Tanh (Tangente Hiperbólica)** transforma um valor real `x` para o intervalo (-1, 1). Sua definição matemática é:

```
F(x) = tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
```

Assim como a Sigmoid, a Tanh possui um formato em "S", mas é simétrica em relação à origem e sua saída é centrada em zero. Sua derivada é `F'(x) = 1 - F(x)^2`, atingindo um máximo de 1.0 em `x=0` e se aproximando de zero para valores de `x` extremos. Embora a Tanh também sofra do problema do gradiente evanescente, ela é geralmente preferível à Sigmoid em camadas ocultas, pois sua saída centrada em zero ajuda a acelerar a convergência.

A **Função ReLU (Rectified Linear Unit)** é definida como `F(x) = max(0, x)`, o que significa que ela retorna o próprio valor de entrada se for positivo, e zero caso contrário. A principal vantagem da ReLU é sua menor suscetibilidade ao problema do gradiente evanescente em comparação com Sigmoid e Tanh. Isso ocorre porque sua derivada é 1 para entradas positivas e 0 para negativas, evitando a saturação da derivada e permitindo que os gradientes fluam de forma mais eficaz através de redes profundas. Além disso, a ReLU é significativamente mais fácil e rápida de calcular do que Sigmoid ou Tanh, contribuindo para a eficiência computacional do treinamento. Devido a essas vantagens, a ReLU tornou-se a escolha padrão para a maioria das redes neurais profundas. Existem variantes da ReLU, como a pReLU (Parameterized ReLU), que adiciona um termo linear para entradas negativas, permitindo que alguma informação passe mesmo para entradas negativas.

A escolha da função de ativação não é arbitrária; ela tem um impacto direto na capacidade da rede de aprender e na estabilidade do treinamento. A transição de Sigmoid/Tanh para ReLU é um marco na história do Deep Learning, demonstrando como a resolução de um problema teórico (gradiente evanescente) levou a uma inovação prática que revolucionou o treinamento de redes neurais profundas, permitindo a construção de modelos com muito mais camadas. A função de ativação é, portanto, um componente arquitetônico chave que afeta diretamente a capacidade de uma rede neural de aprender e de ser treinada de forma estável.

A tabela a seguir apresenta um comparativo das funções de ativação:

**Tabela: Comparativo de Funções de Ativação**

| Característica | Sigmoid | Tanh | ReLU |
| :--- | :--- | :--- | :--- |
| **Definição Matemática** | `F(x) = 1/(1 + e^(-x))` | `F(x) = (e^x - e^(-x))/(e^x + e^(-x))` | `F(x) = max(0, x)` |
| **Intervalo de Saída** | (0, 1) | (-1, 1) | [0, ∞) |
| **Gradiente Evanescente** | Alto | Médio | Baixo |
| **Velocidade Computacional** | Lenta | Lenta | Rápida |
| **Uso Comum** | Camada de saída (prob.) | Menos comum | Camadas ocultas |

Este cálculo ocorre sequencialmente, da camada de entrada em direção à camada de saída. Cada camada da rede realiza uma transformação nos dados que recebe da camada anterior, aplicando operações lineares (multiplicação por pesos e adição de bias) e funções de ativação não-lineares. Para ilustrar com um exemplo de Perceptron Multicamadas (MLP) com uma única camada oculta:

1. Um **vetor de entrada (x)** é recebido pela rede.
2. Na **camada oculta**, a entrada é multiplicada pelos pesos (`W^(1)`) e somada ao bias (`b^(1)`) para formar a entrada para a função de ativação, `z^(1) = W^(1)x + b^(1)`. Esse valor `z^(1)` é então transformado por uma função de ativação φ para produzir a saída da camada oculta, `h = φ(z^(1))`.
3. Na **camada de saída**, a saída da camada oculta `h` é novamente multiplicada pelos pesos (`W^(2)`) e somada ao bias (`b^(2)`) para obter a saída final da rede, `o = W^(2)h + b^(2)`.
4. A saída `o` é comparada com o rótulo verdadeiro `y` usando uma **função de perda** `L = l(o, y)`. Esta função quantifica o quão "errada" a previsão do modelo é em relação ao valor real.
5. Se houver termos de regularização (por exemplo, regularização L2), eles são calculados e adicionados à perda para formar a **função objetivo** `J = L + s`, que é o valor final a ser minimizado.

Durante a Forward Propagation, um **grafo computacional** é construído implicitamente. Este grafo é uma representação das operações matemáticas realizadas e de como os dados fluem através da rede. Em frameworks como PyTorch, a construção desse grafo é **dinâmica**, ocorrendo em tempo de execução. Essa característica oferece grande flexibilidade, permitindo que modelos incorporem fluxos de controle condicionais ou loops, cujo comportamento pode variar com base nas entradas.

A **Backward Propagation (Backprop - Passagem Reversa)** é o método fundamental para calcular os gradientes da função objetivo em relação a todos os parâmetros (pesos e biases) da rede neural. Este processo calcula e armazena os gradientes das variáveis intermediárias e dos parâmetros em ordem inversa, da camada de saída para a camada de entrada, aplicando a **regra da cadeia** do cálculo. A regra da cadeia permite decompor a derivada de uma função composta em um produto de derivadas parciais. Para o exemplo do MLP:

1. Os gradientes da função objetivo `J` são calculados em relação à perda `L` e ao termo de regularização `s`.
2. Em seguida, o gradiente de `J` em relação à saída `o` é calculado, utilizando a derivada da função de perda (`∂L/∂o`), resultando em `∂J/∂o = ∂L/∂o`.
3. Os gradientes em relação aos parâmetros da camada de saída (`W^(2), b^(2)`) são então determinados. Por exemplo, `∂J/∂W^(2) = ∂J/∂o h^T + λ W^(2)` (considerando regularização L2).
4. O gradiente de `J` é propagado de volta para a camada oculta, calculando `∂J/∂h = (W^(2))^T ∂J/∂o`.
5. O gradiente em relação à entrada da função de ativação (`z^(1)`) é obtido através da multiplicação elemento a elemento de `∂J/∂h` pela derivada da função de ativação (`φ'(z^(1))`), ou seja, `∂J/∂z^(1) = ∂J/∂h ⊙ φ'(z^(1))`.
6. Finalmente, os gradientes em relação aos parâmetros da camada oculta (`W^(1), b^(1)`) são calculados. Por exemplo, `∂J/∂W^(1) = ∂J/∂z^(1) x^T + λ W^(1)`.

Em PyTorch, os gradientes são armazenados no atributo `.grad` dos tensores que foram marcados com `requires_grad=True`. É crucial notar que, por padrão, PyTorch **acumula gradientes**. Isso significa que os gradientes de múltiplas passagens de backpropagation são somados. Para garantir que os gradientes de uma nova iteração não se misturem com os anteriores, é indispensável chamar `optimizer.zero_grad()` ou `tensor.grad.zero_()` antes de cada nova passagem de backpropagation.

A Forward Propagation e a Backward Propagation são interdependentes durante o treinamento de uma rede neural. Todas as variáveis intermediárias calculadas na passagem direta devem ser mantidas na memória para que os gradientes possam ser calculados na passagem reversa. Essa necessidade de retenção de estados intermediários implica que o treinamento de modelos de Deep Learning exige significativamente mais memória e armazenamento do que a simples inferência. No contexto de Sensoriamento Remoto, onde as imagens podem ser de altíssima resolução e os modelos de Deep Learning são cada vez mais profundos e largos, essa exigência de memória pode levar a uma explosão no número de variáveis intermediárias. Essa limitação computacional torna a otimização de memória (por exemplo, o uso de `torch.no_grad()` durante a validação ou técnicas como *mixed precision training* em módulos futuros) um aspecto prático crítico para treinar modelos grandes em GPUs com memória limitada. A compreensão dessa limitação é essencial para que os alunos entendam por que certas técnicas de otimização de treinamento são necessárias.

**Exemplo de Código 2.1.3: Forward e Backward Propagation com Autograd**
Este exemplo ilustra o conceito de forward e backward propagation usando o sistema `Autograd` do PyTorch para calcular gradientes automaticamente.

```python
import torch

# 1. Definir um tensor de entrada com requires_grad=True
# Isso informa ao PyTorch para rastrear as operações com este tensor para cálculo de gradientes
x = torch.tensor(2.0, requires_grad=True)
print(f"Tensor de entrada x: {x}")

# 2. Forward Propagation: Definir uma função simples
# y = x^2 + 3x + 5
y = x**2 + 3*x + 5
print(f"Saída y (após forward pass): {y}")

# 3. Backward Propagation: Calcular os gradientes
# Chamamos.backward() na saída escalar (y) para iniciar a backpropagation.
# O Autograd calculará dy/dx.
# dy/dx = 2x + 3
# Para x=2, dy/dx = 2*2 + 3 = 7
y.backward()

# Os gradientes são armazenados no atributo.grad do tensor original
print(f"Gradiente de y em relação a x (dy/dx): {x.grad}")

# Exemplo com múltiplos tensores e acumulação de gradientes
print("\n--- Exemplo com acumulação de gradientes ---")
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

# Primeira computação
c = a * b
c.backward() # d(c)/da = b = 4, d(c)/db = a = 3
print(f"Gradiente de a após primeira backward: {a.grad}")
print(f"Gradiente de b após primeira backward: {b.grad}")

# Segunda computação (sem zerar gradientes)
d = a + b
d.backward() # d(d)/da = 1, d(d)/db = 1
# Os novos gradientes serão SOMADOS aos existentes
print(f"Gradiente de a após segunda backward (acumulado): {a.grad}") # 4 + 1 = 5
print(f"Gradiente de b após segunda backward (acumulado): {b.grad}") # 3 + 1 = 4

# Para evitar acumulação, zere os gradientes antes de cada nova backward pass
print("\n--- Exemplo com zeramento de gradientes ---")
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

# Primeira computação
c = a * b
c.backward()
print(f"Gradiente de a (primeira): {a.grad}")
print(f"Gradiente de b (primeira): {b.grad}")

# Zerar gradientes
a.grad.zero_()
b.grad.zero_()
print(f"Gradiente de a após zerar: {a.grad}")
print(f"Gradiente de b após zerar: {b.grad}")

# Segunda computação (com gradientes zerados)
d = a + b
d.backward()
print(f"Gradiente de a (segunda, zerado antes): {a.grad}") # Deve ser 1
print(f"Gradiente de b (segunda, zerado antes): {b.grad}") # Deve ser 1

# Usando torch.no_grad() para desativar o rastreamento de gradientes
print("\n--- Exemplo com torch.no_grad() ---")
x_no_grad = torch.tensor(5.0, requires_grad=True)
with torch.no_grad():
    y_no_grad = x_no_grad * 2
print(f"y_no_grad.requires_grad: {y_no_grad.requires_grad}") # Deve ser False
# Tentar chamar backward() em y_no_grad resultaria em erro, pois não há grafo para rastrear
# y_no_grad.backward() # Isso geraria um erro

# Usando.detach() para criar uma cópia que não rastreia gradientes
print("\n--- Exemplo com.detach() ---")
x_detach = torch.tensor(10.0, requires_grad=True)
y_intermediate = x_detach * 3
z_detached = y_intermediate.detach() # z_detached não rastreia histórico de y_intermediate
output_final = z_detached * 2

# Chamar backward() em output_final
output_final.backward()
# O gradiente de x_detach será 0, pois z_detached "quebrou" o grafo
print(f"Gradiente de x_detach (após detach): {x_detach.grad}") # Deve ser None ou 0 se já foi calculado antes
# Para ver o gradiente de y_intermediate em relação a x_detach, precisaríamos de outra backward pass
# ou calcular manualmente.
```

**2.1.4. Otimizadores Essenciais: SGD, Adam e a Taxa de Aprendizagem (Learning Rate)**

Os **otimizadores** são algoritmos que ajustam os parâmetros (pesos e biases) de uma rede neural para minimizar a função de perda, utilizando os gradientes calculados pela Backward Propagation. A **taxa de aprendizagem (learning rate - LR)** é um hiperparâmetro fundamental que determina o tamanho do passo que o otimizador dá na direção do gradiente descendente a cada iteração.

A **importância da taxa de aprendizagem** é inegável, sendo considerada "indiscutivelmente o parâmetro mais importante" no treinamento de modelos. Seu impacto é significativo: uma LR muito grande pode levar à divergência da otimização, fazendo com que o modelo "salte" sobre o mínimo da função de perda. Por outro lado, uma LR muito pequena resulta em um treinamento excessivamente lento e pode fazer com que o modelo fique preso em mínimos locais, levando a um resultado subótimo.

Para gerenciar a taxa de aprendizagem de forma eficaz, diversas **estratégias de ajuste** são empregadas:

* **Decaimento da LR (Learning Rate Decay):** A taxa de aprendizagem deve diminuir ao longo do treinamento para permitir uma convergência mais fina e estável à medida que o modelo se aproxima do mínimo da função de perda. Isso evita que o modelo "salte" sobre o mínimo e permite que ele se estabeleça em uma solução mais precisa.
* **Warmup:** Esta estratégia envolve aumentar gradualmente a LR no início do treinamento, em vez de começar com um valor alto. Isso ajuda a evitar a divergência inicial, especialmente com parâmetros aleatoriamente inicializados, e permite um progresso mais rápido do que uma LR consistentemente pequena desde o início. O warmup é particularmente útil para modelos mais avançados que podem ter problemas de estabilidade no início do treinamento.
* **LR Scheduling (Agendamento da Taxa de Aprendizagem):** Refere-se a estratégias mais complexas para ajustar a LR ao longo do tempo, como decaimento piecewise (redução em etapas quando o progresso plateau), ou schedulers baseados em funções como cosseno, que são populares em visão computacional.

Dois dos otimizadores mais comuns são o Gradiente Descendente Estocástico (SGD) e o Adam.

O **Gradiente Descendente Estocástico (SGD)** é um otimizador fundamental que atualiza os parâmetros do modelo usando os gradientes calculados a partir de um pequeno subconjunto de dados (um minibatch) em cada iteração, em vez de usar o conjunto de dados completo. Isso o torna computacionalmente mais eficiente para grandes datasets. A natureza estocástica dos gradientes do minibatch introduz ruído no processo de otimização, o que pode ajudar o modelo a escapar de mínimos locais, mas também pode causar oscilações. Embora o SGD possa apresentar oscilações no processo de otimização devido à natureza estocástica dos gradientes do minibatch, ele é frequentemente combinado com técnicas como o **momentum** para suavizar essas oscilações e acelerar a convergência. O momentum adiciona uma fração do vetor de atualização anterior à atualização atual, ajudando a manter a direção do movimento e a superar vales rasos.

O **Adam (Adaptive Moment Estimation)** é um otimizador adaptativo que ajusta as taxas de aprendizagem para cada parâmetro individualmente, com base nas estimativas dos primeiros e segundos momentos dos gradientes. Ele é conhecido por sua capacidade de "auto-ajuste" durante o treinamento e, em muitos casos, funciona bem com seus hiperparâmetros padrão. O Adam geralmente converge mais rapidamente e é menos sensível à escolha inicial da taxa de aprendizagem em comparação com o SGD. Ele calcula médias móveis exponenciais dos gradientes (primeiro momento) e dos quadrados dos gradientes (segundo momento), e usa essas estimativas para adaptar a taxa de aprendizagem para cada parâmetro. Embora a taxa de aprendizagem para Adam também possa ser ajustada para melhorias, a faixa de valores ótimos é tipicamente menor do que para outros algoritmos.

A escolha e o ajuste de otimizadores são cruciais para a convergência eficiente de modelos de Deep Learning. Diferentes otimizadores e estratégias de agendamento da taxa de aprendizagem podem levar a diferentes resultados em termos de precisão e generalização do modelo, mesmo com o mesmo erro de treinamento. A compreensão de como esses otimizadores funcionam e como a taxa de aprendizagem influencia o processo de treinamento é vital para otimizar o desempenho de modelos, especialmente em aplicações de Sensoriamento Remoto com grandes e complexos conjuntos de dados.

A tabela a seguir compara os otimizadores SGD e Adam:

**Tabela: Comparativo de Otimizadores (SGD vs. Adam)**

| Característica | SGD (Stochastic Gradient Descent) | Adam (Adaptive Moment Estimation) |
| :--- | :--- | :--- |
| **Princípio** | Atualiza pesos na direção oposta ao gradiente do minibatch. | Adapta taxas de aprendizagem para cada parâmetro com base em médias de gradientes (momentos). |
| **Taxa de Aprendizagem** | Única para todos os parâmetros. Necessita ajuste cuidadoso e scheduling. | Adaptativa por parâmetro. Menos sensível à escolha inicial da LR global. |
| **Velocidade de Convergência** | Pode ser mais lento, mas com momentum pode ser eficiente. | Geralmente mais rápido e robusto na convergência. |
| **Memória** | Menor uso de memória, pois armazena apenas gradientes e momentum. | Maior uso de memória, pois armazena estimativas de 1º e 2º momentos para cada parâmetro. |
| **Generalização** | Pode levar a mínimos mais "planos" e melhor generalização em alguns casos. | Pode convergir para mínimos mais "pontudos", às vezes com pior generalização. |
| **Robustez** | Menos robusto a hiperparâmetros inadequados. | Mais robusto a hiperparâmetros, frequentemente funciona bem com defaults. |
| **Uso Comum** | Ainda usado, especialmente com momentum e scheduling avançado. | Muito popular e amplamente utilizado como otimizador padrão. |

Esta tabela oferece um comparativo conciso entre SGD e Adam, destacando suas características e implicações para o treinamento de modelos.

**Exemplo de Código 2.1.4: Configuração de Otimizadores em PyTorch**
Este exemplo demonstra como configurar os otimizadores SGD e Adam em PyTorch, incluindo a definição da taxa de aprendizagem.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definir um modelo simples (apenas para ter parâmetros para otimizar)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1) # 10 entradas, 1 saída
    
    def forward(self, x):
        return self.linear(x)

model_sgd = SimpleModel()
model_adam = SimpleModel()

# 2. Configurar o otimizador SGD (Stochastic Gradient Descent)
# lr (learning rate): Taxa de aprendizagem, um dos hiperparâmetros mais importantes.
# momentum: Ajuda a acelerar o SGD em direções relevantes e a suavizar oscilações.
learning_rate_sgd = 0.01
momentum_sgd = 0.9
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate_sgd, momentum=momentum_sgd)
print(f"Otimizador SGD configurado com LR={learning_rate_sgd}, Momentum={momentum_sgd}")

# 3. Configurar o otimizador Adam (Adaptive Moment Estimation)
# lr (learning rate): Taxa de aprendizagem. Adam é menos sensível à escolha inicial da LR.
# betas: Coeficientes para as médias móveis exponenciais dos gradientes. (beta1, beta2)
# eps: Pequena constante para evitar divisão por zero.
learning_rate_adam = 0.001
betas_adam = (0.9, 0.999) # Valores padrão comuns
eps_adam = 1e-8 # Valor padrão comum
optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate_adam, betas=betas_adam, eps=eps_adam)
print(f"Otimizador Adam configurado com LR={learning_rate_adam}, Betas={betas_adam}, Eps={eps_adam}")

# Exemplo de uso (simulado)
# Criar dados de entrada e rótulos fictícios
dummy_input = torch.randn(1, 10) # Batch size 1, 10 features
dummy_target = torch.randn(1, 1) # Batch size 1, 1 output

# Simular um passo de treinamento com SGD
print("\nSimulando um passo de treinamento com SGD:")
optimizer_sgd.zero_grad() # Zera os gradientes
output_sgd = model_sgd(dummy_input) # Forward pass
loss_sgd = nn.MSELoss()(output_sgd, dummy_target) # Calcula a perda
loss_sgd.backward() # Backward pass
optimizer_sgd.step() # Atualiza os pesos
print(f"Pesos do modelo SGD após 1 passo: {model_sgd.linear.weight.data[:5]}")

# Simular um passo de treinamento com Adam
print("\nSimulando um passo de treinamento com Adam:")
optimizer_adam.zero_grad() # Zera os gradientes
output_adam = model_adam(dummy_input) # Forward pass
loss_adam = nn.MSELoss()(output_adam, dummy_target) # Calcula a perda
loss_adam.backward() # Backward pass
optimizer_adam.step() # Atualiza os pesos
print(f"Pesos do modelo Adam após 1 passo: {model_adam.linear.weight.data[:5]}")
```

**2.2.3. Estrutura `nn.Module`: A Base para Redes Neurais em PyTorch**

No PyTorch, as redes neurais são construídas a partir de blocos fundamentais chamados **módulos**, que são instâncias da classe `torch.nn.Module`. Cada camada de uma rede neural, bem como a rede neural completa em si, é um `nn.Module`. Essa estrutura hierárquica e aninhada simplifica a construção e o gerenciamento de arquiteturas de rede complexas.

Para definir uma rede neural ou uma camada personalizada, é necessário **subclassificar `nn.Module`**. Dentro dessa subclasse, o método `__init__` é utilizado para inicializar as camadas e outros submódulos que compõem a rede. É imperativo chamar `super().__init__()` no início do `__init__` para garantir que a inicialização necessária da classe base `nn.Module` seja realizada, o que inclui a configuração do atributo `_modules` (um `OrderedDict` que armazena os submódulos). A falha em chamar `super().__init__()` ou em registrar os submódulos corretamente pode resultar em parâmetros do modelo não sendo rastreados pelo otimizador.

O método **`forward`** é o coração de qualquer `nn.Module`. Ele define as operações que serão realizadas nos dados de entrada para produzir a saída do módulo. Embora o `forward` seja onde a lógica computacional é implementada, o módulo deve ser invocado diretamente (e.g., `model(input_data)`) em vez de chamar `model.forward()` explicitamente. Isso ocorre porque a invocação direta (`__call__`) do módulo executa operações adicionais importantes, como o registro de *hooks* (funções que podem ser executadas antes ou depois da passagem direta) e o gerenciamento do modo de treinamento/avaliação (`model.train()` / `model.eval()`).

A modularidade do `nn.Module` permite a **construção de redes complexas** através da composição. Por exemplo:

* **`nn.Sequential`**: É um contêiner ordenado de módulos, onde os dados são passados sequencialmente através de cada módulo na ordem definida. É útil para construir redes com um fluxo de dados linear e é uma maneira concisa de empilhar camadas.
* **Blocos Paralelos**: É possível combinar saídas de múltiplos submódulos em paralelo, concatenando-as e passando-as para camadas subsequentes, permitindo arquiteturas mais intrincadas.
* **Composição de Módulos**: Um `nn.Module` pode conter outros `nn.Module`s como atributos, criando uma hierarquia de módulos. Isso permite a construção de arquiteturas complexas de forma organizada e reutilizável.

Muitas camadas dentro de uma rede neural são **parametrizadas**, ou seja, possuem pesos e biases associados que são otimizados durante o treinamento. Ao subclassificar `nn.Module`, o PyTorch rastreia automaticamente todos os campos definidos dentro do objeto do modelo que são instâncias de `nn.Parameter` ou outros `nn.Module`s, tornando todos os parâmetros acessíveis através dos métodos `model.parameters()` ou `model.named_parameters()`. Essa funcionalidade é essencial para que os otimizadores possam acessar e atualizar os parâmetros da rede.

Em suma, `nn.Module` fornece uma estrutura robusta e extensível para definir os blocos de construção das redes neurais, garantindo que os parâmetros sejam gerenciados corretamente e que a passagem direta possa ser executada com funcionalidades adicionais, como hooks, facilitando o desenvolvimento de modelos de Deep Learning.

**Exemplo de Código 2.2.3: Construindo Redes com `nn.Module` e `nn.Sequential`**
Este código demonstra como definir uma rede neural simples usando `nn.Module` e como usar `nn.Sequential` para empilhar camadas.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Construção de um modelo simples com herança de nn.Module
# Esta é a forma mais flexível para construir redes complexas
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # A chamada a super().__init__() é obrigatória
        super().__init__()
        # Definir as camadas lineares
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Implementar a lógica da passagem direta (forward pass)
        x = self.fc1(x)
        x = F.relu(x) # Usando a função de ativação ReLU
        x = self.fc2(x)
        return x

# 2. Construção de um modelo com nn.Sequential
# Útil para redes com uma arquitetura de camadas lineares
model_sequential = nn.Sequential(
    nn.Linear(10, 5), # Camada 1: 10 entradas, 5 saídas
    nn.ReLU(),
    nn.Linear(5, 1) # Camada 2: 5 entradas, 1 saída
)

# Criar uma instância do modelo MLP
input_size = 784 # Ex: Imagem 28x28 achatada
hidden_size = 128
output_size = 10 # Ex: 10 classes para classificação
model_mlp = SimpleMLP(input_size, hidden_size, output_size)

# Imprimir as estruturas dos modelos
print("--- Modelo construído com nn.Module ---")
print(model_mlp)
print("\n--- Modelo construído com nn.Sequential ---")
print(model_sequential)

# Demonstração do forward pass
dummy_input_mlp = torch.randn(64, input_size) # Batch size de 64
output_mlp = model_mlp(dummy_input_mlp)
print(f"\nShape da saída do MLP: {output_mlp.shape}")

dummy_input_sequential = torch.randn(32, 10) # Batch size de 32
output_sequential = model_sequential(dummy_input_sequential)
print(f"Shape da saída do Sequential: {output_sequential.shape}")

# Acessar os parâmetros do modelo
print("\nParâmetros do modelo MLP:")
for name, param in model_mlp.named_parameters():
    print(f"  - {name}: {param.shape}")

print("\nParâmetros do modelo Sequential:")
for name, param in model_sequential.named_parameters():
    print(f"  - {name}: {param.shape}")
```

**2.2.4. `Dataset` e `DataLoader`: Gerenciamento Eficiente de Dados**

No PyTorch, as classes `Dataset` e `DataLoader` são componentes cruciais para o gerenciamento eficiente e otimizado do fluxo de dados em pipelines de Deep Learning. Elas trabalham em conjunto para fornecer dados aos modelos de forma estruturada.

A classe **`Dataset`** é uma representação abstrata dos dados. Seu propósito principal é definir como amostras de dados individuais são acessadas e pré-processadas. Ela atua como uma interface para os dados brutos, seja para imagens, texto ou dados numéricos. Para criar um `Dataset` personalizado, é necessário herdar de `torch.utils.data.Dataset` e implementar dois métodos essenciais:

* `__len__(self)`: Deve retornar o número total de amostras no dataset. Este método é importante para que o `DataLoader` e outros componentes saibam o tamanho do dataset.
* `__getitem__(self, idx)`: É responsável por carregar e pré-processar uma única amostra de dados dado seu índice (`idx`). Este método deve retornar a amostra de dados e seu rótulo/alvo correspondente. É aqui que transformações como normalização, redimensionamento ou aumento de dados podem ser aplicadas.

A classe **`DataLoader`** envolve um `Dataset` e fornece um iterador eficiente sobre o dataset. Suas funções principais são:

* **Divisão em Batches:** Agrupa múltiplas amostras em lotes (batches), o que é fundamental para o treinamento eficiente em GPUs, pois permite o processamento paralelo.
* **Embaralhamento (Shuffle):** Pode embaralhar aleatoriamente os dados no início de cada época, o que ajuda a evitar que o modelo aprenda a ordem das amostras e melhora a generalização. Geralmente, `shuffle=True` é usado para o conjunto de treinamento e `shuffle=False` para validação/teste.
* **Transformações On-the-Fly:** Pode aplicar transformações aos dados (como normalização, redimensionamento ou aumento de dados) à medida que são carregados, sem a necessidade de pré-processar todo o dataset de uma vez.
* **Carregamento Paralelo:** Pode utilizar múltiplos processos de trabalho (`num_workers`) para carregar dados em paralelo, acelerando significativamente o processo de carregamento de dados e evitando gargalos de I/O.
* **`drop_last`**: Se o número total de exemplos não for perfeitamente divisível pelo `batch_size`, o último batch será menor. Definir `drop_last=True` faz com que esse batch incompleto seja descartado, garantindo que todos os batches tenham o mesmo tamanho. Isso pode ser útil para consistência no treinamento ou quando o Batch Normalization é usado.

Existem dois tipos principais de `Dataset`:

* **Map-style datasets:** Permitem acesso aleatório às amostras por índice (e.g., arrays NumPy, arquivos em disco). São geralmente preferidos por facilitarem o embaralhamento e o carregamento paralelo.
* **Iterable-style datasets:** Permitem apenas acesso sequencial (e.g., geradores Python, dados transmitidos por rede). São úteis quando leituras aleatórias são caras ou impossíveis, ou quando o dataset é muito grande para caber na memória e precisa ser transmitido.

A eficiência do `DataLoader` é particularmente relevante para grandes datasets, como os encontrados em Sensoriamento Remoto. Embora técnicas como a geração de dados "on-the-fly" ou estratégias de leitura otimizada de disco possam ser exploradas, a sobrecarga do interpretador Python e o caching do sistema operacional podem limitar os ganhos de desempenho para datasets menores. No entanto, para datasets que excedem a memória RAM disponível, a capacidade do `DataLoader` de carregar dados em lotes e em paralelo torna-se indispensável para o treinamento de modelos de Deep Learning. A reprodutibilidade dos resultados pode ser garantida usando `torch.manual_seed()` para controlar a geração de dados aleatórios ou o embaralhamento.

**Exemplo de Código 2.2.4: `Dataset` e `DataLoader` Personalizados**
Este código demonstra a criação de um `Dataset` personalizado e o uso de `DataLoader` para carregar dados em batches.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Para transformações de dados

# Definir um Dataset personalizado
class CustomImageDataset(Dataset):
    """
    Um Dataset simulado para imagens.
    Na prática, o __init__ carregaria metadados e o __getitem__
    carregaria e processaria a imagem do disco.
    """
    def __init__(self, data_size=100, transform=None):
        # Simular a criação de dados (na prática, isso seria carregar do disco)
        self.data = [torch.randn(3, 64, 64) for _ in range(data_size)]
        self.labels = [torch.randint(0, 10, (1,)).item() for _ in range(data_size)]
        self.transform = transform

    def __len__(self):
        # Retorna o número total de amostras
        return len(self.data)

    def __getitem__(self, idx):
        # Retorna uma amostra de dados e seu rótulo
        image = self.data[idx]
        label = self.labels[idx]
        
        # Aplicar transformações se houver
        if self.transform:
            image = self.transform(image)

        return image, label

# 1. Configurar transformações
# Exemplo de uma transformação simples
# Normaliza o tensor de entrada para que a média seja 0.5 e o desvio padrão 0.5
transform_data = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Criar instâncias do Dataset
train_dataset = CustomImageDataset(data_size=1000, transform=transform_data)
val_dataset = CustomImageDataset(data_size=200, transform=transform_data)

print(f"Dataset de treinamento criado com {len(train_dataset)} amostras.")
print(f"Dataset de validação criado com {len(val_dataset)} amostras.")

# 3. Criar instâncias do DataLoader
batch_size = 64
num_workers = 4 # Use mais workers para acelerar o carregamento em sistemas multi-core
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f"\nDataLoader de treinamento criado com batch_size={batch_size}, shuffle=True, num_workers={num_workers}.")
print(f"DataLoader de validação criado com batch_size={batch_size}, shuffle=False, num_workers={num_workers}.")

# 4. Iterar sobre o DataLoader
# No loop de treinamento, você iteraria assim:
print("\nIterando sobre o primeiro batch do DataLoader de treinamento:")
for images, labels in train_loader:
    print(f"Shape do batch de imagens: {images.shape}")
    print(f"Shape do batch de rótulos: {labels.shape}")
    print(f"Valores de rótulos no batch: {labels}")
    # Verificar a normalização
    print(f"Média das imagens no batch (canal 0): {images[:, 0].mean():.4f}")
    print(f"Desvio padrão das imagens no batch (canal 0): {images[:, 0].std():.4f}")
    break # Parar após o primeiro batch para a demonstração
```

#### Primeira Implementação Completa: Um MLP do Zero

Esta seção descreve os passos para a primeira implementação completa de uma rede neural em PyTorch, focando em um Multi-Layer Perceptron (MLP), e aborda o ciclo de treinamento e as ferramentas de visualização.

**Implementação de um Multi-Layer Perceptron (MLP)**

Um **Multi-Layer Perceptron (MLP)** é uma rede neural feedforward que consiste em múltiplas camadas de neurônios, cada uma totalmente conectada à camada anterior e à camada seguinte. Diferente do Perceptron de camada única, os MLPs utilizam funções de ativação não-lineares nas camadas ocultas, o que lhes permite aprender e distinguir padrões que não são linearmente separáveis.

A arquitetura de um MLP geralmente inclui uma **camada de entrada**, uma ou mais **camadas ocultas** e uma **camada de saída**. A camada de entrada recebe os dados brutos, onde cada neurônio representa uma característica. As camadas ocultas realizam a maior parte das computações, com cada neurônio recebendo entradas de todos os neurônios da camada anterior, multiplicando-as por pesos correspondentes e somando um bias, antes de aplicar uma função de ativação não-linear (como ReLU). A camada de saída produz as previsões finais, com o número de neurônios dependendo da tarefa (e.g., classificação binária, multiclasse, regressão).

Na **implementação de um MLP do zero** em PyTorch, os pesos e biases de cada camada são inicializados. A inicialização dos parâmetros é um aspecto crucial para a estabilidade numérica do treinamento. Escolhas inadequadas podem levar a problemas de **gradientes evanescentes (vanishing gradients)** ou **gradientes explosivos (exploding gradients)**.

* **Gradientes Evanescentes:** Ocorrem quando os gradientes se tornam muito pequenos durante a backpropagation, impedindo que os pesos das camadas iniciais sejam atualizados efetivamente. Funções de ativação como Sigmoid e Tanh são mais propensas a esse problema, pois suas derivadas se aproximam de zero para valores de entrada extremos.
* **Gradientes Explosivos:** Ocorrem quando os gradientes se tornam excessivamente grandes, resultando em atualizações de peso muito grandes que desestabilizam o modelo, impedindo a convergência do otimizador.

A **inicialização aleatória** dos pesos é fundamental para **quebrar a simetria** entre os neurônios de uma mesma camada. Se os pesos fossem inicializados com valores idênticos, todos os neurônios de uma camada se comportariam da mesma forma, recebendo os mesmos gradientes e nunca aprendendo representações distintas, o que limitaria severamente a capacidade expressiva da rede. Técnicas como a **Inicialização Xavier** buscam manter a variância das ativações e dos gradientes fixas durante as propagações direta e reversa, contribuindo para um treinamento mais estável. Para uma camada totalmente conectada sem não-linearidades, com `n_in` entradas e `n_out` saídas, a inicialização Xavier amostra pesos de uma distribuição Gaussiana com média zero e variância `σ² = 2/(n_in + n_out)`.

A passagem direta (`forward`) de um MLP envolve remodelar a entrada (e.g., achatar uma imagem 2D em um vetor 1D), aplicar as transformações lineares (multiplicação de matrizes com pesos e adição de bias) e as funções de ativação em cada camada. A função de perda (e.g., `nn.CrossEntropyLoss` para classificação) é então aplicada para quantificar o erro da previsão.

**Exemplo de Código 2.3.1: Implementação de um MLP Simples**
Este código demonstra a implementação de um MLP simples usando `nn.Module` do PyTorch, com uma camada oculta e função de ativação ReLU.

```python
import torch
import torch.nn as nn

# Definir a arquitetura da rede MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Camada de entrada -> Camada oculta
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Camada oculta -> Camada de saída
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Função de ativação ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # A passagem direta (forward pass) do MLP
        # Entrada: (batch_size, input_size)
        out = self.fc1(x)
        # Saída da primeira camada: (batch_size, hidden_size)
        out = self.relu(out)
        # Saída da ReLU: (batch_size, hidden_size)
        out = self.fc2(out)
        # Saída final: (batch_size, output_size)
        return out

# Parâmetros de exemplo
input_size = 28 * 28 # Imagens 28x28 achatadas (e.g., MNIST)
hidden_size = 512
output_size = 10 # 10 classes

# Criar uma instância do modelo MLP
model = MLP(input_size, hidden_size, output_size)
print("Arquitetura do MLP:")
print(model)

# Criar um tensor de entrada de exemplo
# Batch size de 64, imagem achatada de 28x28
dummy_input = torch.randn(64, input_size)
# Executar a passagem direta
output = model(dummy_input)

print(f"\nShape do tensor de entrada: {dummy_input.shape}")
print(f"Shape do tensor de saída: {output.shape}")

# Inspecionar os parâmetros (pesos e biases) do modelo
print("\nPesos e Biases do modelo:")
for name, param in model.named_parameters():
    print(f"  - {name}: {param.shape}")
    # Exibir alguns valores para ilustração
    print(f"    - Exemplo de valores: {param.data.flatten()[:5]}")
```

**2.3.2. O Loop de Treinamento Passo a Passo**

O **loop de treinamento** em PyTorch é o processo iterativo pelo qual um modelo ajusta seus parâmetros para minimizar a função de perda, com base nos dados de treinamento. Um ciclo de treinamento típico para uma única época (epoch) envolve os seguintes passos:

1. **Modo de Treinamento (`model.train()`):** Antes de iniciar a época de treinamento, o modelo é colocado no modo de treinamento. Isso ativa comportamentos específicos de camadas como Dropout (para regularização) e Batch Normalization (que usa estatísticas do batch para normalização).
2. **Iteração sobre o `DataLoader`:** O loop começa iterando sobre os batches de dados fornecidos pelo `DataLoader`. Para cada iteração, um batch de `inputs` (dados de treinamento) e `labels` (rótulos verdadeiros) é recuperado.
3. **Zerar os Gradientes:** Antes de cada nova etapa de otimização, é crucial zerar os gradientes acumulados do batch anterior, chamando `optimizer.zero_grad()`. Isso evita que os gradientes de diferentes batches se somem, levando a atualizações de peso incorretas, pois o PyTorch acumula gradientes por padrão.
4. **Passagem Direta (Forward Pass):** Os `inputs` são passados através do modelo (`outputs = model(inputs)`) para obter as previsões (`outputs`).
5. **Cálculo da Perda:** A função de perda (`loss_fn`) é aplicada para comparar as `outputs` (previsões) com os `labels` (rótulos verdadeiros), resultando em um valor de `loss` para o batch atual (`loss = loss_fn(outputs, labels)`).
6. **Passagem Reversa (Backward Pass):** O método `loss.backward()` é invocado. Isso aciona o motor Autograd do PyTorch, que calcula os gradientes da perda em relação a todos os parâmetros aprendíveis do modelo. Esses gradientes indicam a direção e a magnitude do ajuste necessário para cada parâmetro.
7. **Passo do Otimizador:** O otimizador (`optimizer.step()`) utiliza os gradientes calculados para ajustar os pesos e biases do modelo de acordo com o algoritmo de otimização escolhido (e.g., SGD, Adam).
8. **Relatório de Perda (Intra-Época):** A perda do batch atual é acumulada, e periodicamente (e.g., a cada 1000 batches), a perda média é calculada e reportada, fornecendo feedback em tempo real sobre o progresso do treinamento.

Além do loop de treinamento por batch, atividades importantes são realizadas **por época**:

* **Modo de Avaliação (`model.eval()`):** Após cada época de treinamento, o modelo é colocado no modo de avaliação. Isso desabilita o Dropout e faz com que o Batch Normalization use as estatísticas da população (média e variância acumuladas durante o treinamento) em vez das estatísticas do batch atual, garantindo resultados consistentes e determinísticos durante a inferência.
* **Execução de Validação:** Uma passagem de validação é realizada em um conjunto de dados separado (`validation_loader`), sem cálculo de gradientes (`with torch.no_grad()`), para avaliar o desempenho do modelo em dados não vistos e monitorar o overfitting.
* **Log e Visualização:** As perdas de treinamento e validação são registradas (e.g., no TensorBoard) para visualização e comparação.
* **Salvamento do Modelo (Checkpointing):** O estado do modelo (seus parâmetros) é salvo periodicamente, especialmente se o desempenho na validação melhorar, para permitir a recuperação do melhor modelo treinado.

Este processo iterativo e detalhado garante que o modelo seja treinado de forma eficaz, seu desempenho seja monitorado e as melhores versões sejam preservadas.

**Exemplo de Código 2.3.2: Loop de Treinamento Passo a Passo**
Este código implementa um loop de treinamento completo para o MLP definido anteriormente, incluindo a passagem direta, cálculo de perda, passagem reversa e atualização dos pesos.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Definir o modelo (reutilizando a classe MLP do exemplo anterior)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 28 * 28
hidden_size = 512
output_size = 10
model = MLP(input_size, hidden_size, output_size)

# 2. Definir a função de perda e o otimizador
# CrossEntropyLoss é a função de perda ideal para problemas de multiclasse
criterion = nn.CrossEntropyLoss()
# Otimizador Adam, uma escolha popular e robusta
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Criar dados simulados de treinamento e validação
# Criar tensores aleatórios para simular o dataset
num_samples = 1000
train_features = torch.randn(num_samples, input_size)
train_labels = torch.randint(0, output_size, (num_samples,))

val_features = torch.randn(200, input_size)
val_labels = torch.randint(0, output_size, (200,))

# Criar os Datasets e DataLoaders
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 4. Iniciar o loop de treinamento
num_epochs = 5
for epoch in range(num_epochs):
    # Entrar no modo de treinamento
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. Zerar os gradientes do otimizador para evitar acúmulo
        optimizer.zero_grad()
        
        # 2. Passagem Direta (Forward Pass)
        outputs = model(inputs)
        
        # 3. Calcular a perda
        loss = criterion(outputs, labels)
        
        # 4. Passagem Reversa (Backward Pass)
        loss.backward()
        
        # 5. Passo do Otimizador
        optimizer.step()
        
        running_loss += loss.item()

    # Log da perda por época
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # 5. Modo de avaliação e validação
    model.eval() # Entrar no modo de avaliação
    correct = 0
    total = 0
    with torch.no_grad(): # Desativar o cálculo de gradientes
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on validation set: {accuracy:.2f}%')

print("Treinamento concluído.")
```

**2.3.3. Uso do `DataLoader` no Treinamento**

O `DataLoader` é uma abstração fundamental no PyTorch que facilita o processo de alimentação de dados para o modelo durante o loop de treinamento. Ele atua como um invólucro em torno de um `Dataset`, fornecendo um iterador que entrega os dados em lotes (batches) de forma eficiente.

Durante o treinamento, o `DataLoader` é iterado para obter os `inputs` e `labels` de cada minibatch. Por exemplo, em um loop de treinamento, a estrutura básica é `for images, targets in train_dataloader:`, onde `images` e `targets` já são tensores contendo um lote de amostras.

Parâmetros importantes do `DataLoader` incluem:

* **`batch_size`**: Define o número de amostras por lote. Um `batch_size` maior pode acelerar o treinamento em GPUs, mas exige mais memória.
* **`shuffle`**: Quando `True` (geralmente para o conjunto de treinamento), os dados são embaralhados a cada época para garantir que o modelo não aprenda a ordem das amostras e para melhorar a generalização.
* **`drop_last`**: Se o número total de exemplos não for perfeitamente divisível pelo `batch_size`, o último batch será menor. Definir `drop_last=True` faz com que esse batch incompleto seja descartado, garantindo que todos os batches tenham o mesmo tamanho. Isso pode ser útil para consistência no treinamento ou quando o Batch Normalization é usado.
* **`num_workers`**: Permite o carregamento paralelo de dados usando múltiplos processos, o que pode acelerar significativamente o pipeline de entrada de dados, especialmente para datasets grandes ou operações de pré-processamento intensivas.

Para cenários onde os dados não podem ser mantidos inteiramente na memória (comum em Sensoriamento Remoto com imagens de alta resolução), o `DataLoader` pode ser configurado para carregar ou gerar dados "on-the-fly" dentro do método `__getitem__` do `Dataset`. Isso evita o consumo excessivo de memória ao carregar todo o dataset de uma vez. A reprodutibilidade dos resultados pode ser garantida usando `torch.manual_seed()` para controlar a geração de dados aleatórios ou o embaralhamento.

A eficiência do `DataLoader` é um fator crítico para o desempenho do treinamento, especialmente com grandes volumes de dados. Ele otimiza o acesso aos dados, a paralelização e o gerenciamento de memória, permitindo que os modelos de Deep Learning sejam treinados de forma mais rápida e estável.

**Exemplo de Código 2.3.3: Uso do `DataLoader` no Loop de Treinamento**
Este exemplo demonstra como o `DataLoader` é integrado ao loop de treinamento para iterar sobre os dados em batches.

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Reutilizando o modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 28 * 28
hidden_size = 512
output_size = 10
model = MLP(input_size, hidden_size, output_size)

# Dados simulados
num_samples = 1000
train_features = torch.randn(num_samples, input_size)
train_labels = torch.randint(0, output_size, (num_samples,))

# Usando TensorDataset para empacotar features e labels
train_dataset = TensorDataset(train_features, train_labels)

batch_size = 64
# Configurando o DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Parâmetros para o treinamento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de treinamento
print(f"Iniciando o loop de treinamento com batches de tamanho {batch_size}...")
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs e labels já vêm prontos do DataLoader
        
        # Passos de otimização
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Imprimir a cada 10 batches para demonstração
        if (i+1) % 10 == 0:
            print(f"  - Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0
print("Loop de treinamento concluído.")
```

**2.3.4. Configuração e Uso Básico do TensorBoard para Visualização**

O **TensorBoard** é uma ferramenta de visualização de código aberto desenvolvida pelo TensorFlow, mas amplamente utilizada com PyTorch, que permite rastrear e visualizar métricas de aprendizado de máquina, como perda e precisão, visualizar o grafo do modelo, exibir histogramas e imagens, entre outras funcionalidades. É uma ferramenta indispensável para monitorar o progresso do treinamento e depurar modelos de Deep Learning.

A **configuração básica** do TensorBoard em PyTorch envolve a criação de uma instância de `torch.utils.tensorboard.SummaryWriter`. Por padrão, o `SummaryWriter` salva os logs em um diretório `runs/`. É uma prática recomendada usar uma estrutura de pastas hierárquica ou com carimbos de data/hora (`log_dir`) para organizar os logs de diferentes experimentos, facilitando a comparação entre eles.

O **uso básico** para registrar métricas envolve o método `writer.add_scalar(tag, scalar_value, global_step)`. Por exemplo, para registrar a perda de treinamento a cada batch ou a precisão a cada época, o `tag` seria o nome da métrica (e.g., 'Loss/train', 'Accuracy/test'), `scalar_value` seria o valor da métrica, e `global_step` seria um contador que aumenta ao longo do treinamento (e.g., número do batch ou da época). O `writer.add_scalars()` permite registrar múltiplas métricas relacionadas (e.g., perda de treinamento vs. validação) em um único gráfico.

Outras funcionalidades úteis incluem:

* **Visualização do Grafo do Modelo:** `writer.add_graph(model, input_to_model)` permite visualizar a arquitetura da rede neural, ajudando a verificar se o modelo está configurado como esperado.
* **Histogramas:** `writer.add_histogram()` pode ser usado para visualizar a distribuição de tensores (como pesos e ativações) ao longo do tempo, o que é útil para diagnosticar problemas como gradientes evanescentes ou explosivos.
* **Imagens:** `writer.add_image()` permite exibir imagens, o que pode ser usado para visualizar as entradas, as saídas do modelo ou até mesmo os filtros aprendidos.

Após registrar os dados, o método `writer.flush()` deve ser chamado para garantir que todos os eventos pendentes sejam gravados em disco. Para iniciar a interface web do TensorBoard e visualizar os dados, o comando `tensorboard --logdir=runs` (ou o diretório de log especificado) é executado no terminal, e a interface pode ser acessada em `http://localhost:6006/`.

O TensorBoard é uma ferramenta poderosa para avaliar o desempenho do modelo, analisar sua arquitetura e monitorar o processo de treinamento. Ele fornece uma representação visual clara de como a perda e a precisão mudam ao longo das épocas, permitindo a comparação entre diferentes execuções de treinamento e auxiliando na melhoria contínua do modelo.

**Exemplo de Código 2.3.4: Integração do TensorBoard no Treinamento**
Este código demonstra como integrar o TensorBoard ao loop de treinamento para registrar e visualizar métricas.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# 1. Preparar o ambiente para o TensorBoard
# Cria um diretório de log para este experimento
log_dir = os.path.join('runs', 'mlp_experiment_with_tensorboard')
writer = SummaryWriter(log_dir)

# 2. Definir o modelo (reutilizando a classe MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 28 * 28
hidden_size = 512
output_size = 10
model = MLP(input_size, hidden_size, output_size)

# 3. Criar dados simulados
num_samples = 1000
train_features = torch.randn(num_samples, input_size)
train_labels = torch.randint(0, output_size, (num_samples,))
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. Adicionar o grafo do modelo ao TensorBoard
dummy_input = torch.randn(1, input_size)
writer.add_graph(model, dummy_input)

# 5. Definir o loop de treinamento com logging
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
global_step = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        global_step += 1
        
        # Log da perda de treinamento para cada batch
        writer.add_scalar('Loss/train', loss.item(), global_step)

    # Log da perda média por época
    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # (Opcional) Log de histogramas dos pesos e biases para diagnóstico
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Grads/{name}', param.grad, epoch)

writer.close()
print(f"\nTreinamento concluído. Execute 'tensorboard --logdir={log_dir}' no terminal para visualizar.")
```

### Exercício: Treinamento de um MLP no dataset MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hKdzzgJ7N2MmAIsgnziiLfmtgU-z79Zs?usp=sharing)