### Cálculo para Deep Learning (1h10min)

O cálculo diferencial é a espinha dorsal da otimização em Deep Learning, fornecendo as ferramentas para que as redes neurais aprendam e melhorem seu desempenho.

#### Derivadas e gradientes - intuição visual (20min)

- **Derivada**: A derivada de uma função mede a taxa de variação instantânea de uma função em relação à sua entrada. Em termos simples, ela indica o quanto a função muda para uma pequena variação na entrada, e seu sinal indica a direção dessa mudança (se a função está aumentando ou diminuindo).

Matematicamente, a derivada de uma função f(x) em um ponto x é definida como:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Intuição Visual**: Imagine uma função f(x) plotada em um gráfico. A derivada em um ponto específico é a inclinação da linha tangente à curva nesse ponto.
- Se a derivada é positiva, a função está subindo (aumentando).
- Se a derivada é negativa, a função está descendo (diminuindo).
- Se a derivada é zero, a função está em um ponto de máximo, mínimo ou sela (plano).

- **Gradiente**: O gradiente é a generalização da derivada para funções com múltiplas variáveis de entrada (funções multivariadas). O gradiente é um vetor que contém as derivadas parciais da função em relação a cada uma de suas variáveis de entrada.

Para uma função f(x₁,x₂,...,xₙ), o gradiente é denotado por ∇f e é definido como:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$

**Intuição Visual**: Imagine uma paisagem montanhosa (uma função com múltiplas entradas e uma saída, como uma função de custo). O gradiente em um ponto específico dessa paisagem aponta na direção da maior inclinação, ou seja, a direção de maior crescimento da função. Se você estivesse em um ponto da montanha e quisesse subir o mais rápido possível, o gradiente indicaria a direção a seguir.

No Deep Learning, o objetivo é minimizar uma função de custo (ou perda), que quantifica o erro do modelo. Portanto, o interesse recai sobre o negativo do gradiente (−∇f), que aponta na direção da maior descida, permitindo que o modelo se mova em direção a um erro menor. O gradiente funciona como uma "bússola" no espaço de alta dimensionalidade dos parâmetros do modelo, guiando o processo de otimização para o "vale" da função de perda.

#### Regra da cadeia - conceito fundamental (25min)

A **Regra da Cadeia** é uma ferramenta matemática fundamental para calcular derivadas de funções compostas. Em redes neurais, onde temos múltiplas camadas de transformações compostas (a saída de uma camada é a entrada da próxima), a regra da cadeia é essencial para o algoritmo de Backpropagation.

- **Conceito Básico**: Se temos uma função y=f(u) e u=g(x), então a derivada de y em relação a x é dada por:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

- **Generalização para Múltiplas Variáveis**: Se uma função z depende de variáveis u e v, e u e v por sua vez dependem de x, a derivada parcial de z em relação a x é:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$$

- **Relevância para Redes Neurais e Backpropagation**: Em uma rede neural, a função de custo (C) depende da saída da rede, que por sua vez depende das ativações das camadas anteriores, e estas dependem dos pesos e vieses. O Backpropagation é essencialmente uma aplicação sistemática da regra da cadeia para computar o gradiente da função de custo em relação a cada peso e viés na rede.

**Intuição Visual da Propagação**: Imagine uma rede neural como uma série de operações encadeadas. Quando calculamos o erro na saída da rede, queremos saber como esse erro é influenciado por cada peso e viés nas camadas anteriores. A regra da cadeia nos permite "propagar" essa informação de erro (gradiente) para trás através da rede, camada por camada.

Por exemplo, para calcular o erro em um neurônio na camada l (δⱼˡ), o Backpropagation usa a regra da cadeia para relacionar esse erro ao erro na camada seguinte (l+1):

$$\delta_j^l = \sum_k w_{kj}^{l+1} \delta_k^{l+1} \sigma'(z_j^l)$$

Onde δⱼˡ é o erro do j-ésimo neurônio na camada l, wₖⱼˡ⁺¹ é o peso da conexão do j-ésimo neurônio da camada l para o k-ésimo neurônio da camada l+1, δₖˡ⁺¹ é o erro do k-ésimo neurônio na camada l+1, e σ'(zⱼˡ) é a derivada da função de ativação do j-ésimo neurônio na camada l em relação à sua entrada ponderada.

Essa capacidade de propagar eficientemente os gradientes para trás através de múltiplas camadas é o que torna o treinamento de redes neurais profundas viável, permitindo que os pesos e vieses sejam ajustados de forma a minimizar a função de custo.

#### Otimização e gradiente descendente (15min)

O **Gradiente Descendente** é o algoritmo de otimização fundamental empregado para minimizar uma função de custo (ou perda) em modelos de Deep Learning, ajustando iterativamente os parâmetros do modelo (pesos e vieses).

O processo do Gradiente Descendente segue uma sequência iterativa:

1. **Cálculo da Perda**: A função de custo é avaliada com base nos parâmetros atuais do modelo, quantificando o erro das previsões em relação aos valores reais.

2. **Cálculo do Gradiente**: O gradiente da função de custo em relação a cada parâmetro é determinado. Este vetor gradiente indica a direção e a magnitude da inclinação da função de custo no ponto atual do espaço de parâmetros.

3. **Atualização dos Parâmetros**: Os parâmetros do modelo são ajustados em uma pequena quantidade na direção oposta ao gradiente, ou seja, descendo a "inclinação" da função de custo. A fórmula de atualização é tipicamente:

$$\theta_{new} = \theta_{old} - \eta \nabla J(\theta_{old})$$

Onde:
- θₙₑw são os novos valores dos parâmetros (pesos e vieses).
- θₒₗd são os valores atuais dos parâmetros.
- η (eta) é a taxa de aprendizagem (learning rate).
- ∇J(θₒₗd) é o gradiente da função de custo J em relação aos parâmetros θₒₗd.

4. **Iteração**: Os passos são repetidos em ciclos contínuos até que a função de custo não possa ser reduzida significativamente, indicando que o modelo alcançou a convergência.

- **Taxa de Aprendizagem (Learning Rate, η)**: Este é um hiperparâmetro crucial que determina o tamanho do passo dado na direção do gradiente negativo. Uma taxa de aprendizagem muito alta pode levar a oscilações excessivas ou a ultrapassar o mínimo da função de custo, resultando em volatilidade ou divergência do processo de treinamento. Por outro lado, uma taxa muito baixa pode tornar o treinamento extremamente lento e potencialmente fazer com que o algoritmo fique preso em mínimos locais, sem conseguir alcançar o mínimo global.

- **Convergência**: Refere-se ao ponto no processo de treinamento em que iterações adicionais do Gradiente Descendente não resultam em uma redução significativa da perda. Isso indica que o algoritmo encontrou um conjunto de parâmetros que minimiza a função de custo, ou pelo menos um mínimo local aceitável. Em funções de custo convexas, o Gradiente Descendente garante a convergência para o mínimo global se a taxa de aprendizagem for apropriadamente escolhida.

A intuição visual para esses conceitos pode ser imaginada como estar em uma montanha em um dia de neblina e desejar chegar ao vale mais baixo. O gradiente indica a direção mais íngreme para baixo. A taxa de aprendizagem representa o tamanho do seu passo. Se os passos forem muito grandes, há o risco de pular o vale ou de oscilar descontroladamente. Se forem muito pequenos, levará uma quantidade excessiva de tempo para chegar ao fundo.

#### Prática no Colab: Visualização de gradiente descendente (10min)

Nesta seção, os alunos implementarão o algoritmo de Gradiente Descendente para uma função simples e visualizarão seu comportamento no Google Colab.

**Exercício Sugerido:**

1. **Implementação de Gradiente Descendente para uma Função Simples:**
   - Escolha uma função de custo convexa simples, como J(x)=x²+2x+1.
   - Implemente o algoritmo de Gradiente Descendente para encontrar o mínimo dessa função.
   - Armazene os valores de x e J(x) em cada iteração.

2. **Visualização Animada da Convergência:**
   - Utilize matplotlib para criar um gráfico da função de custo.
   - Adicione pontos que representem a posição do algoritmo em cada iteração, criando uma animação que mostre como ele se move em direção ao mínimo.

3. **Experimentação com Taxas de Aprendizagem:**
   - Execute o algoritmo com diferentes valores de taxa de aprendizagem (por exemplo, 0.01, 0.1, 0.5, 1.0).
   - Observe e discuta o impacto da taxa de aprendizagem na velocidade de convergência e na estabilidade do algoritmo (convergência lenta, rápida, oscilação, divergência).

Esta prática concreta consolidará a intuição matemática por trás do Gradiente Descendente e preparará os alunos para compreender otimizadores mais complexos em redes neurais.

