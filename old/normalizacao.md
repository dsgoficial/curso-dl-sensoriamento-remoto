### A Importância da Normalização para Imagens de Satélite

No contexto de Deep Learning, a **normalização** é um passo de pré-processamento essencial para garantir que os dados de entrada de uma rede neural tenham uma distribuição consistente. Isso impede que características com valores de intensidade muito altos dominem o aprendizado do modelo e, mais importante, ajuda os algoritmos de otimização (como o Gradiente Descendente) a convergir mais rapidamente e de forma mais estável.

Para imagens de satélite, este processo se torna ainda mais crítico e complexo, devido a algumas características únicas:

1. **Múltiplos Canais Espectrais:** Enquanto uma imagem RGB tem 3 canais (Red, Green, Blue), uma imagem de satélite pode ter dezenas de canais, cada um capturando dados de uma parte diferente do espectro eletromagnético (por exemplo, infravermelho próximo, ondas curtas de infravermelho).

2. **Valores de Pixel e Bit Depth:** Os valores de pixel de imagens de satélite não se restringem ao intervalo 0-255. Dependendo do sensor e do produto (e.g., Level-1, Level-2), os dados podem ser armazenados com 16 bits, 32 bits ou até mesmo como valores de reflectância em ponto flutuante, com faixas de valores muito maiores.

3. **Variações Físicas:** A intensidade dos pixels pode variar drasticamente entre diferentes cenas devido a condições atmosféricas (neblina, nuvens), ângulo de iluminação solar e a natureza da superfície terrestre.

### Abordagens para a Normalização de Imagens de Satélite

Existem duas abordagens principais para normalizar imagens de satélite: a normalização por canal (a mais comum e recomendada) e a normalização global.

#### 1. Normalização por Canal (Per-Channel Normalization)

Esta é a abordagem mais robusta para imagens multiespectrais. Ela consiste em calcular a média ($\mu$) e o desvio padrão ($\sigma$) para cada canal individualmente, usando as estatísticas de um conjunto de dados de treinamento.

**Princípio:** Cada banda espectral (por exemplo, a banda do azul, a banda do infravermelho próximo) possui propriedades físicas e faixas de valores distintas. Um modelo de Deep Learning precisa que cada canal seja tratado de forma independente para extrair as características relevantes de cada um. A normalização por canal garante que, após o pré-processamento, todos os canais tenham uma média de $0$ e um desvio padrão de $1$, o que coloca todos os canais em um mesmo patamar de "importância" para o modelo, independentemente de suas variações originais de brilho e contraste.

A fórmula para a normalização por canal é:

$$
x'_{i,c} = \frac{x_{i,c} - \mu_c}{\sigma_c}
$$

Onde:

* $x'_{i,c}$ é o valor normalizado do pixel $i$ no canal $c$.

* $x_{i,c}$ é o valor original do pixel $i$ no canal $c$.

* $\mu_c$ é a média do canal $c$ em todo o conjunto de dados.

* $\sigma_c$ é o desvio padrão do canal $c$ em todo o conjunto de dados.

**Aplicação em Sensoriamento Remoto:** A normalização por canal é crucial em tarefas como classificação de uso do solo, onde a diferença de reflectância entre as bandas visíveis e infravermelhas é a característica principal para distinguir vegetação de solo. Ao normalizar cada banda separadamente, a rede neural pode aprender a importância relativa de cada espectro de forma mais eficaz.

#### 2. Normalização Global (Global Normalization)

Esta abordagem calcula uma única média ($\mu_{global}$) e um único desvio padrão ($\sigma_{global}$) para todos os canais e todas as imagens no conjunto de dados, tratando-os como um único grande vetor.

**Princípio:** Embora simples de implementar, esta abordagem pode ser inadequada para dados de sensoriamento remoto. Se a banda do vermelho tem valores médios de pixel muito mais baixos que a banda do infravermelho próximo, a normalização global pode reduzir a variabilidade e o contraste da banda do vermelho, dificultando a extração de características pelo modelo.

### Como Calcular a Média e o Desvio Padrão

Para um grande conjunto de dados de imagens de satélite, não é prático carregar todas as imagens na memória de uma vez para calcular a média e o desvio padrão. O cálculo é feito de forma incremental, iterando sobre o dataset e acumulando as estatísticas. O processo é o seguinte:

1. **Inicialização:** Crie dois arrays de zeros, um para a soma dos valores de pixels (`soma_total`) e outro para a soma dos quadrados dos valores de pixels (`soma_quadrados_total`), com a mesma dimensão do número de canais. Inicialize um contador (`num_pixels_total`) para zero.

2. **Iteração:** Percorra o conjunto de dados de treinamento, imagem por imagem, e para cada imagem:

   * Calcule a soma dos valores de pixels de cada canal na imagem atual.

   * Calcule a soma dos quadrados dos valores de pixels de cada canal na imagem atual.

   * Adicione essas somas às variáveis `soma_total` e `soma_quadrados_total`.

   * Adicione o número de pixels da imagem ao `num_pixels_total`.

3. **Cálculo Final da Média (**$\mu$**):**

   * A média de cada canal ($\mu_c$) é calculada dividindo `soma_total[c]` por `num_pixels_total`.

4. **Cálculo Final do Desvio Padrão (**$\sigma$**):**

   * A variância de cada canal ($\text{var}_c$) é calculada usando a fórmula:

    $$
    \text{var}_c = \frac{\text{soma\_quadrados\_total}[c]}{num\_pixels\_total} - \mu_c^2
    $$

   * O desvio padrão ($\sigma_c$) é a raiz quadrada da variância:

    $$
    \sigma_c = \sqrt{\text{var}_c}
    $$

    Ao final deste processo, você terá um vetor de médias ($\mu_1, \mu_2, ..., \mu_c$) e um vetor de desvios padrão ($\sigma_1, \sigma_2, ..., \sigma_c$), um para cada canal. Esses valores são então salvos e usados para normalizar todas as imagens (treinamento, validação e teste) antes de serem passadas para o modelo.