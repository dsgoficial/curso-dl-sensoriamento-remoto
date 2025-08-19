## Plotagem de Gráficos e Imagens com Matplotlib no Google Colab

A visualização de dados é uma etapa fundamental em qualquer processo de análise, permitindo a compreensão de padrões, tendências e anomalias. No campo da ciência de dados e visão computacional, ferramentas como Matplotlib e OpenCV, em conjunto com NumPy, são indispensáveis para transformar dados brutos em representações visuais significativas e para manipular imagens de forma programática. Este relatório serve como um complemento aprofundado ao Módulo 1, focando nas capacidades do Matplotlib para plotagem de gráficos e exibição de imagens, integrando-o com as poderosas funcionalidades de processamento de imagem do OpenCV e as operações de array do NumPy. Todos os exemplos de código são projetados para serem executáveis no ambiente Google Colab, garantindo praticidade e acessibilidade para o aprendizado e a experimentação.

### Configuração do Ambiente e Fundamentos do Matplotlib

Para iniciar a jornada de visualização de dados no Google Colab, é crucial configurar o ambiente corretamente e compreender a estrutura fundamental do Matplotlib.

#### Configuração Essencial para Google Colab

A primeira linha de código em qualquer notebook Colab que envolva plotagem com Matplotlib deve ser `%matplotlib inline`. Esta "magic command" instrui o ambiente a renderizar os gráficos diretamente no notebook, logo abaixo da célula de código que os gera, em vez de abri-los em janelas separadas. Além disso, a importação das bibliotecas necessárias é um passo inicial indispensável.

`matplotlib.pyplot` é a interface de plotagem principal, `numpy` é essencial para manipulação numérica de dados e imagens, `PIL.Image` (Pillow) é comumente usada para carregar imagens, e `cv2` (OpenCV) é a biblioteca padrão para operações de visão computacional.

```python
# Configuração para exibir gráficos inline no Google Colab
%matplotlib inline

# Importações essenciais
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Para carregar imagens
import cv2  # Para processamento de imagens (se necessário)

print("Ambiente configurado e bibliotecas importadas com sucesso!")
```

#### A Anatomia de um Plot Matplotlib: Figure e Axes

O Matplotlib adota uma estrutura hierárquica para a construção de gráficos, que é fundamental para um controle preciso sobre a visualização. No topo dessa hierarquia está a **Figure**, que pode ser concebida como a "tela" ou "janela" principal onde o gráfico será desenhado. Dentro de uma Figure, um ou mais objetos **Axes** são definidos. Um Axes representa a área de plotagem individual, contendo os elementos visuais do gráfico, como linhas, pontos, barras, eixos X e Y, rótulos e um título.

É importante notar a distinção entre **Axes** (a área de plotagem completa) e **Axis** (os eixos individuais, como o eixo X ou Y, que controlam ticks e rótulos).

A maneira mais recomendada para iniciar a criação de um gráfico é utilizando `plt.subplots()`. Esta função retorna uma tupla contendo a Figure e um ou mais objetos Axes, simplificando a configuração inicial.

```python
fig, ax = plt.subplots(figsize=(8, 6))  # Cria uma Figure e um único Axes
ax.set_title("Título do Gráfico")
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
plt.show()  # Exibe o gráfico
```

#### Abordagem Orientada a Objetos para Maior Controle e Previsibilidade

O Matplotlib oferece duas interfaces principais para a criação de gráficos: a interface baseada em estado (matplotlib.pyplot) e a interface orientada a objetos (OO). A interface pyplot, que se assemelha ao MATLAB, opera em um estado global implícito, onde funções como `plt.plot()` ou `plt.title()` atuam sobre a Figure e o Axes "atuais". Embora conveniente para scripts rápidos, essa abordagem pode levar a comportamentos imprevisíveis em ambientes como o Google Colab, onde as células são executadas sequencialmente e o estado global pode ser modificado de forma não óbvia entre as execuções de células.

Em contraste, a interface orientada a objetos oferece um controle explícito sobre os elementos do gráfico. Ao usar `fig, ax = plt.subplots()` e, subsequentemente, chamar métodos diretamente nos objetos ax (como `ax.plot()`, `ax.set_title()`, `ax.set_xlabel()`), o desenvolvedor garante que as modificações se apliquem ao Axes específico que se pretende. Esta prática resulta em um código mais robusto, legível e previsível, especialmente benéfico em notebooks interativos como o Colab, onde a clareza sobre qual objeto está sendo manipulado é crucial para evitar resultados inesperados.

### Plotagem de Gráficos Fundamentais com Matplotlib

Matplotlib oferece uma vasta gama de tipos de gráficos para diversas necessidades de visualização de dados. Os gráficos de linha, dispersão, barras e histogramas são pilares para a análise exploratória e apresentação de dados.

#### Gráficos de Linha (plt.plot)

A função `plt.plot()` é a ferramenta fundamental para a criação de gráficos de linha, que são ideais para exibir tendências ao longo do tempo ou em relação a uma variável contínua. Ela aceita dados para os eixos X e Y, permitindo uma representação flexível. No uso mais básico, pode-se fornecer apenas os valores Y, e o Matplotlib gerará automaticamente os valores X como uma sequência de inteiros (0, 1, 2...). No entanto, é comum fornecer explicitamente tanto os valores X quanto Y para maior controle.

A personalização é um ponto forte do `plt.plot()`. É possível estilizar as linhas usando um terceiro argumento opcional, uma "string de formato", que combina informações de cor, estilo de linha e marcador (por exemplo, 'r--' para uma linha tracejada vermelha). Alternativamente, argumentos de palavra-chave como `color`, `linestyle` (ls), `linewidth` (lw), `marker`, `markersize` e `label` oferecem um controle mais granular sobre cada aspecto visual da linha.

##### Formatos Comuns para plt.plot

| Formato | Descrição da Cor | Descrição do Estilo de Linha | Descrição do Marcador | Exemplo |
|---------|------------------|------------------------------|----------------------|---------|
| b- | Azul | Linha Sólida | Nenhum | `plt.plot(x, y, 'b-')` |
| r-- | Vermelho | Linha Tracejada | Nenhum | `plt.plot(x, y, 'r--')` |
| g^ | Verde | Nenhum | Triângulos | `plt.plot(x, y, 'g^')` |
| ks: | Preto | Linha Pontilhada | Quadrados | `plt.plot(x, y, 'ks:')` |
| c-. | Ciano | Linha Traço-Ponto | Nenhum | `plt.plot(x, y, 'c-.')` |

A adição de elementos textuais é crucial para a interpretabilidade do gráfico. O título do gráfico é definido com `ax.set_title()`, os rótulos dos eixos X e Y com `ax.set_xlabel()` e `ax.set_ylabel()`, respectivamente. Para identificar múltiplas linhas em um gráfico, a função `ax.legend()` exibe uma legenda, utilizando os rótulos fornecidos no parâmetro label de cada chamada `ax.plot()`. Finalmente, `ax.grid(True)` adiciona uma grade ao fundo do gráfico, auxiliando na leitura dos valores.

```python
# Dados de exemplo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Criação do plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando múltiplas linhas com personalização
ax.plot(x, y1, color='blue', linestyle='-', linewidth=2, label='Seno')
ax.plot(x, y2, 'ro--', markersize=5, label='Cosseno')  # String de formato

# Adicionando título, rótulos e legenda
ax.set_title('Gráfico de Seno e Cosseno', fontsize=16)
ax.set_xlabel('Ângulo (radianos)', fontsize=12)
ax.set_ylabel('Magnitude', fontsize=12)
ax.legend(loc='upper right', fontsize=10)  # Posição da legenda
ax.grid(True, linestyle=':', alpha=0.7)  # Adiciona grade

plt.show()
```

#### Gráficos de Dispersão (plt.scatter)

Os gráficos de dispersão, criados com `plt.scatter()`, são ferramentas excelentes para visualizar a relação entre duas variáveis numéricas, onde cada ponto no gráfico representa uma única observação. Diferentemente de `plt.plot`, que conecta pontos com linhas, `plt.scatter` plota pontos individuais, o que é ideal para identificar agrupamentos, tendências ou valores atípicos em um conjunto de dados.

Uma das grandes vantagens de `plt.scatter` é a sua capacidade de mapear dimensões adicionais de dados para as propriedades visuais dos pontos. Os parâmetros `s` (size) e `c` (color) permitem que uma terceira e quarta variável sejam representadas, respectivamente, pelo tamanho e cor dos marcadores. Isso permite a visualização efetiva de até quatro dimensões de dados (X, Y, Tamanho, Cor) em um único gráfico 2D. Esta capacidade é fundamental na análise exploratória de dados, pois possibilita a identificação de clusters, outliers ou padrões complexos que não seriam evidentes ao observar apenas as duas dimensões principais. A escolha de um `cmap` (colormap) apropriado é crucial aqui, pois a percepção humana de gradientes de cor pode influenciar significativamente a interpretação dos dados.

O parâmetro `data` é outra funcionalidade útil, permitindo que se passe um dicionário ou um objeto similar a um DataFrame e referencie as colunas por nome, o que aumenta a legibilidade do código.

```python
# Dados de exemplo
np.random.seed(42)  # Para reprodutibilidade
num_pontos = 100
x_data = np.random.rand(num_pontos) * 10
y_data = 2 * x_data + np.random.randn(num_pontos) * 5
tamanhos = np.random.rand(num_pontos) * 500  # Tamanho dos pontos
cores = np.random.rand(num_pontos)  # Valores para mapear a cor

# Criação do plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotando o gráfico de dispersão com personalização
scatter = ax.scatter(x_data, y_data, s=tamanhos, c=cores, cmap='viridis', alpha=0.7,
                    edgecolors='w', linewidth=0.5, label='Pontos de Dados')

# Adicionando título e rótulos
ax.set_title('Gráfico de Dispersão Personalizado', fontsize=16)
ax.set_xlabel('Variável X', fontsize=12)
ax.set_ylabel('Variável Y', fontsize=12)

# Adicionando colorbar para o mapeamento de cores
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Valor de Cor', fontsize=12)

ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)
plt.show()
```

#### Gráficos de Barras (plt.bar)

Os gráficos de barras, gerados com `plt.bar()`, são ideais para a representação de dados categóricos, onde o comprimento de cada barra é diretamente proporcional ao valor que ela representa. Os parâmetros mais básicos são `x` para as categorias (rótulos no eixo X) e `height` para os valores correspondentes (altura das barras).

A personalização dos gráficos de barras é bastante flexível. O parâmetro `width` permite controlar a largura das barras (o padrão é 0.8). A cor das barras pode ser definida usando o parâmetro `color`, que aceita uma única cor, uma lista de cores (aplicadas recursivamente a cada barra) ou um dicionário que mapeia nomes de colunas para cores. Uma funcionalidade importante é o parâmetro `bottom`, que possibilita a criação de gráficos de barras empilhadas. Neste caso, a base de uma barra é definida pelo topo da barra anterior, permitindo visualizar a composição de diferentes componentes dentro de cada categoria. O parâmetro `align` controla como as barras são alinhadas em relação aos rótulos do eixo X, com opções como 'center' ou 'edge'. Para gráficos de barras horizontais, a função `plt.barh()` é utilizada, operando de forma similar a `plt.bar()` mas exibindo as barras na horizontal.

Embora os gráficos de barras sejam intuitivos para comparações diretas entre categorias, a capacidade de empilhar barras com o parâmetro `bottom` expande sua utilidade para a visualização de "partes de um todo" ou a composição de diferentes componentes dentro de cada categoria. Isso transforma um gráfico de comparação simples em uma ferramenta para entender a estrutura interna de cada grupo. A escolha entre barras verticais e horizontais também não é meramente estética; barras horizontais são frequentemente mais eficazes para exibir rótulos de categorias longos, pois oferecem mais espaço para o texto.

```python
# Dados de exemplo
categorias = ['A', 'B', 'C', 'D']
valores1 = np.array([20, 35, 30, 25])
valores2 = np.array([15, 20, 10, 30])

# Gráfico de Barras Simples
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categorias, valores1, color='skyblue', width=0.6, label='Série 1')
ax.set_title('Vendas por Categoria', fontsize=16)
ax.set_xlabel('Categoria', fontsize=12)
ax.set_ylabel('Quantidade Vendida', fontsize=12)
ax.legend()
plt.show()

# Gráfico de Barras Empilhadas
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categorias, valores1, color='lightcoral', label='Série 1')
ax.bar(categorias, valores2, bottom=valores1, color='lightgreen', label='Série 2')  # Empilha a Série 2
ax.set_title('Vendas Empilhadas por Categoria', fontsize=16)
ax.set_xlabel('Categoria', fontsize=12)
ax.set_ylabel('Quantidade Total', fontsize=12)
ax.legend()
plt.show()

# Gráfico de Barras Horizontal
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(categorias, valores1, color='gold', label='Série 1')
ax.set_title('Vendas Horizontais por Categoria', fontsize=16)
ax.set_xlabel('Quantidade Vendida', fontsize=12)
ax.set_ylabel('Categoria', fontsize=12)
ax.legend()
plt.show()
```

#### Histogramas (plt.hist)

Histogramas são ferramentas estatísticas cruciais para a visualização da distribuição de dados numéricos. Eles funcionam dividindo o intervalo de valores dos dados em uma série de "bins" (intervalos não sobrepostos) e, em seguida, contando a frequência ou a densidade de valores que caem em cada bin, representando essas contagens como barras.

Para a análise da distribuição de dados unidimensionais (1D), a função `plt.hist()` é a escolha apropriada. Para conjuntos de dados bidimensionais (2D), onde se busca visualizar a frequência conjunta de duas variáveis, `plt.hist2d()` é a função utilizada, que gera um mapa de calor representando a densidade de pontos nas regiões do plano.

Os parâmetros chave para `plt.hist()` incluem:

- **x**: Os dados de entrada para os quais o histograma será construído.
- **bins**: Pode ser um inteiro que especifica o número de bins, uma sequência de valores que define as bordas dos bins, ou a string 'auto' para que o Matplotlib determine automaticamente o número ideal de bins.
- **density**: Se definido como True, o histograma é normalizado para formar uma estimativa de densidade de probabilidade, onde a área total sob o histograma soma 1. Isso é fundamental para comparar a forma de distribuições com diferentes números de pontos de dados, pois a área total sob cada histograma se torna 1, permitindo uma comparação justa das formas das distribuições. Sem `density=True`, um conjunto de dados maior sempre pareceria ter barras mais altas, obscurecendo a verdadeira forma da distribuição.
- **histtype**: Define o estilo visual do histograma, com opções como 'bar' (barras tradicionais), 'barstacked' (barras empilhadas para múltiplos datasets), 'step' (apenas contornos) ou 'stepfilled' (contornos preenchidos).
- **range**: Um tupla que especifica o limite inferior e superior do intervalo dos bins. Valores fora deste intervalo são ignorados.

A personalização visual é possível através do acesso aos "patches" (as barras individuais) que `plt.hist` retorna. Isso permite modificar cores individualmente, por exemplo, colorindo as barras com base em sua altura. Adicionalmente, `PercentFormatter` pode ser empregado para formatar o eixo Y como porcentagens quando `density=True`, facilitando a interpretação das proporções. Para histogramas 2D, a escolha do `cmap` (colormap) e a inclusão de uma colorbar são cruciais para interpretar a "densidade" de pontos em regiões bidimensionais, revelando correlações ou agrupamentos nos dados.

```python
# Dados de exemplo (distribuição normal)
np.random.seed(19680801)
data1 = np.random.normal(loc=0, scale=1, size=10000)
data2 = np.random.normal(loc=5, scale=0.8, size=10000)

# Histograma 1D Simples
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data1, bins=30, color='teal', alpha=0.7, label='Distribuição 1')
ax.set_title('Histograma de Dados 1D', fontsize=16)
ax.set_xlabel('Valor', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
ax.legend()
plt.show()

# Histograma 1D com Densidade e Múltiplos Datasets
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data1, bins=30, density=True, alpha=0.6, color='blue', label='Distribuição 1 (Densidade)')
ax.hist(data2, bins=30, density=True, alpha=0.6, color='red', label='Distribuição 2 (Densidade)')
ax.set_title('Histograma de Densidade para Múltiplos Datasets', fontsize=16)
ax.set_xlabel('Valor', fontsize=12)
ax.set_ylabel('Densidade de Probabilidade', fontsize=12)
ax.legend()
plt.show()

# Histograma 2D
fig, ax = plt.subplots(figsize=(8, 7))
hist_2d = ax.hist2d(data1, data2, bins=40, cmap='Blues')
fig.colorbar(hist_2d[3], ax=ax, label='Contagem de Pontos')  # hist_2d[3] é o objeto mappable
ax.set_title('Histograma 2D', fontsize=16)
ax.set_xlabel('Variável X', fontsize=12)
ax.set_ylabel('Variável Y', fontsize=12)
plt.show()
```

### Visualização e Processamento de Imagens com Matplotlib e OpenCV

A manipulação e visualização de imagens são tarefas centrais em visão computacional e análise de dados visuais. Matplotlib, com sua função imshow, e OpenCV, com suas capacidades de processamento, formam uma dupla poderosa para essas finalidades.

#### Exibindo Imagens (plt.imshow)

A função `plt.imshow()` é a principal ferramenta do Matplotlib para exibir dados como uma imagem em uma grade 2D regular. Ela é versátil, capaz de exibir tanto dados RGB(A) (imagens coloridas com ou sem transparência) quanto dados escalares 2D (imagens em escala de cinza ou mapas de calor), que são mapeados para cores usando um colormap.

O carregamento de dados de imagem é geralmente feito convertendo-os para arrays NumPy. A biblioteca Pillow (PIL.Image) é comumente utilizada para abrir arquivos de imagem, e `np.asarray()` para converter o objeto PIL.Image.Image em um array NumPy. Para exemplos no Google Colab, imagens podem ser carregadas diretamente de URLs usando `requests` e `BytesIO` ou através do upload de arquivos.

É fundamental compreender os formatos dos arrays NumPy que `imshow()` espera para representar diferentes tipos de imagens:

- **(M, N)**: Representa uma imagem em escala de cinza ou dados escalares 2D. Os valores são mapeados para cores usando um colormap (cmap) e normalização.
- **(M, N, 3)**: Representa uma imagem RGB (Vermelho, Verde, Azul). Os valores dos pixels podem ser floats entre 0 e 1 ou inteiros entre 0 e 255.
- **(M, N, 4)**: Representa uma imagem RGBA (Vermelho, Verde, Azul, Alpha), incluindo um canal de transparência. Os valores também podem ser floats (0-1) ou inteiros (0-255).

As duas primeiras dimensões, M e N, definem, respectivamente, o número de linhas (altura) e colunas (largura) da imagem.

##### Formatos de Array para plt.imshow

| Shape do Array X | Descrição | Exemplo de Dados | Casos de Uso |
|------------------|-----------|------------------|--------------|
| (M, N) | Imagem em escala de cinza ou dados escalares 2D. Os valores são mapeados para cores via cmap. | `np.array([[1, 2], [3, 4]])` | Mapas de calor, dados de sensor, imagens médicas, imagens monocromáticas. |
| (M, N, 3) | Imagem RGB (Vermelho, Verde, Azul). Canais de cor explícitos. | `np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]])` | Fotografias coloridas, saídas de modelos de visão computacional. |
| (M, N, 4) | Imagem RGBA (Vermelho, Verde, Azul, Alpha). Inclui canal de transparência. | `np.array([[[255, 0, 0, 255], [0, 255, 0, 128]], [[0, 0, 255, 255], [255, 255, 0, 64]]])` | Imagens com transparência (ex: PNGs), sobreposição de camadas. |

Os parâmetros essenciais para imshow permitem um controle significativo sobre a exibição da imagem:

- **X**: Os dados da imagem, como um array NumPy ou uma imagem PIL.
- **cmap**: O colormap, usado para mapear dados escalares para cores (por exemplo, 'gray' para escala de cinza, 'viridis' para um gradiente de cores perceptualmente uniforme, 'hot' para um mapa de calor). Este parâmetro é ignorado para imagens RGB(A).
- **origin**: Define se o índice `[0, 0]` do array corresponde ao canto superior esquerdo ('upper', padrão para imagens e matrizes) ou inferior esquerdo ('lower', padrão para gráficos cartesianos). Isso afeta a orientação vertical da imagem.
- **aspect**: Controla a proporção do Axes. 'equal' garante que os pixels sejam quadrados, enquanto 'auto' ajusta o aspecto para preencher o Axes, o que pode resultar em pixels não quadrados.
- **interpolation**: O método de interpolação usado para redimensionar a imagem ao exibi-la (por exemplo, 'nearest' para pixels nítidos, 'bilinear' ou 'bicubic' para suavização).
- **vmin, vmax**: Para dados escalares, esses parâmetros definem o intervalo de dados que o colormap cobrirá, sendo úteis para ajustar o contraste da imagem.

Para interpretar o mapeamento de cores em imagens de dados escalares, `plt.colorbar()` é uma adição crucial, fornecendo uma referência visual da escala de valores.

##### Parâmetros Chave para plt.imshow

| Parâmetro | Descrição | Valores Comuns/Efeito | Implicação |
|-----------|-----------|----------------------|------------|
| X | Dados da imagem. | Array NumPy (M,N), (M,N,3), (M,N,4). | Entrada fundamental. |
| cmap | Colormap para dados escalares. | 'gray', 'viridis', 'hot', 'jet'. | Define a paleta de cores para imagens em escala de cinza ou dados numéricos. |
| origin | Posição do índice `[0, 0]`. | 'upper' (padrão para imagens), 'lower' (padrão para gráficos). | Afeta a orientação vertical da imagem. |
| aspect | Proporção do Axes. | 'equal' (pixels quadrados), 'auto' (preenche o Axes). | Garante que os pixels sejam exibidos corretamente ou que a imagem preencha a área. |
| interpolation | Método de interpolação. | 'nearest', 'bilinear', 'bicubic'. | Suaviza ou pixeliza a imagem ao redimensionar. |
| vmin, vmax | Limites de dados para o colormap. | Números floats. | Ajusta o contraste e a faixa de valores visíveis para dados escalares. |

Embora imshow seja comumente associado à exibição de fotografias, sua capacidade de mapear dados escalares (arrays com shape (M, N)) para cores através de cmap, vmin e vmax a transforma em uma ferramenta poderosa para visualizar qualquer matriz numérica como um mapa de calor ou imagem pseudocolorizada. Isso é extremamente útil em campos como ciência de dados para visualizar matrizes de correlação, mapas de ativação de redes neurais ou dados de densidade. Os parâmetros origin e aspect são cruciais para garantir que a interpretação visual (seja como uma matriz matemática ou uma imagem fotográfica) esteja correta.

```python
# Para carregar uma imagem de uma URL no Colab:
import requests
from io import BytesIO

# URL de uma imagem de exemplo (substitua por sua própria imagem se desejar)
image_url = "https://matplotlib.org/_static/stinkbug.png"
response = requests.get(image_url)
img_pil = Image.open(BytesIO(response.content))
img_np = np.array(img_pil)

print(f"Shape da imagem original: {img_np.shape}")

# Exibindo imagem RGB/RGBA
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_np)
ax.set_title('Imagem Original (RGB/RGBA)', fontsize=16)
ax.axis('off')  # Remove os eixos para visualização de imagem
plt.show()

# Exibindo imagem em escala de cinza (canal de luminância)
lum_img = img_np[:, :, 0]  # Pega o primeiro canal (Red) como luminância
print(f"Shape da imagem em escala de cinza: {lum_img.shape}")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(lum_img, cmap='gray', origin='upper', interpolation='bilinear')
fig.colorbar(im, ax=ax, label='Intensidade do Pixel')
ax.set_title('Imagem em Escala de Cinza com Colormap', fontsize=16)
ax.axis('off')
plt.show()

# Ajustando contraste com vmin/vmax
fig, ax = plt.subplots(figsize=(6, 6))
im_contrast = ax.imshow(lum_img, cmap='gray', vmin=50, vmax=200)
fig.colorbar(im_contrast, ax=ax, label='Intensidade do Pixel (Contraste Ajustado)')
ax.set_title('Imagem em Escala de Cinza (Contraste Ajustado)', fontsize=16)
ax.axis('off')
plt.show()
```

#### Manipulação Básica de Imagens para Plotagem (NumPy e OpenCV)

Para preparar dados para plotagem ou para realizar tarefas complexas de visão computacional, NumPy e OpenCV são bibliotecas indispensáveis. Enquanto Matplotlib atua como a camada de visualização, NumPy e OpenCV fornecem as ferramentas robustas para a manipulação e processamento de dados e imagens.

##### Manipulação de Arrays NumPy

NumPy é a base para a computação numérica em Python, e suas funções de manipulação de arrays são cruciais para o pré-processamento de dados de imagem.

- **numpy.reshape()**: Esta função permite alterar a forma (dimensões) de um array sem modificar seus dados subjacentes. É extremamente útil para transformar arrays 1D em 2D (ou vice-versa) ou para reorganizar as dimensões de um array para que se ajustem aos requisitos de uma função específica.
- **numpy.transpose()**: Utilizada para permutar os eixos de um array. Para um array 2D, ela executa a transposição de matriz padrão, trocando linhas por colunas. Para arrays de N dimensões, o parâmetro axes permite especificar a nova ordem dos eixos, oferecendo controle preciso sobre a reorganização dos dados.
- **numpy.stack() vs. numpy.vstack() / numpy.hstack()**:
  - **numpy.stack()**: Esta função une uma sequência de arrays ao longo de um novo eixo, o que significa que ela aumenta a dimensão do array resultante. Todos os arrays de entrada devem ter a mesma forma.
  - **numpy.vstack()**: Empilha arrays verticalmente, linha por linha, ao longo do primeiro eixo. É funcionalmente equivalente a stack(axis=0).
  - **numpy.hstack()**: Empilha arrays horizontalmente, coluna por coluna, ao longo do segundo eixo (ou do primeiro para arrays 1D).

É fundamental compreender que muitas operações NumPy, como reshape e transpose, frequentemente retornam uma view do array original, em vez de uma cópia independente. Isso significa que nenhuma nova memória é alocada, o que é altamente eficiente para grandes conjuntos de dados. No entanto, uma consequência direta é que qualquer modificação feita na view também afetará o array original. Para garantir que uma operação crie uma cópia de dados completamente independente, `array.copy()` deve ser explicitamente chamado. Esta distinção é vital para evitar efeitos colaterais inesperados em pipelines de dados complexos e para otimizar o desempenho em grandes datasets.

```python
# Exemplo de reshape
arr_1d = np.arange(12)
print(f"Array 1D: {arr_1d}, Shape: {arr_1d.shape}")
arr_2d = arr_1d.reshape((3, 4))
print(f"Array 2D (reshape): \n{arr_2d}, Shape: {arr_2d.shape}")

# Exemplo de transpose
arr_transposed = arr_2d.transpose()
print(f"Array 2D Transposed: \n{arr_transposed}, Shape: {arr_transposed.shape}")

# Exemplo de stack (cria nova dimensão)
a = np.array([2, 3, 1])
b = np.array([4, 5, 6])
stacked_arr = np.stack((a, b))
print(f"Array Stacked (nova dimensão): \n{stacked_arr}, Shape: {stacked_arr.shape}")

# Exemplo de vstack (empilha verticalmente)
x = np.array([[2, 3], [1, 4]])
y = np.array([[5, 6], [7, 8]])
vstack_arr = np.vstack((x, y))
print(f"Array vstacked: \n{vstack_arr}, Shape: {vstack_arr.shape}")

# Exemplo de hstack (empilha horizontalmente)
hstack_arr = np.hstack((x, y))
print(f"Array hstacked: \n{hstack_arr}, Shape: {hstack_arr.shape}")
```

##### Funções NumPy para Manipulação de Arrays (Pré-processamento)

| Função NumPy | Propósito | Parâmetros Chave | Comportamento (View/Copy) |
|--------------|-----------|------------------|---------------------------|
| reshape() | Altera as dimensões de um array sem alterar os dados. | shape (tupla de ints, -1 para inferir). | Retorna uma view se possível, senão uma cópia. |
| transpose() | Permuta os eixos de um array. | axes (tupla de ints para ordem dos eixos). | Retorna uma view se possível, senão uma cópia. |
| stack() | Une arrays ao longo de um novo eixo. | axis (onde o novo eixo é inserido). | Retorna uma cópia. |
| vstack() | Empilha arrays verticalmente (linha por linha). | tup (sequência de arrays). | Retorna uma cópia. |
| hstack() | Empilha arrays horizontalmente (coluna por coluna). | tup (sequência de arrays). | Retorna uma cópia. |

##### Funções OpenCV para Processamento de Imagens

OpenCV (cv2) é uma biblioteca de código aberto amplamente utilizada para tarefas de visão computacional e processamento de imagens. Suas funções são otimizadas para desempenho e operam nativamente em arrays NumPy, o que facilita uma integração fluida com o Matplotlib para visualização.

- **cv2.imread() e cv2.imshow()**:
  - **cv2.imread(filepath)**: Carrega uma imagem de um arquivo para um array NumPy. Por padrão, o OpenCV lê imagens no formato BGR (Azul, Verde, Vermelho), que difere do formato RGB (Vermelho, Verde, Azul) comumente usado por Matplotlib e outras bibliotecas.
  - **cv2.imshow(window_name, image_array)**: Esta função é usada para exibir uma imagem em uma janela dedicada do OpenCV. No entanto, é crucial notar que `cv2.imshow()` não funciona diretamente no Google Colab de forma interativa. O Colab é um ambiente baseado em navegador e não possui uma interface gráfica de usuário (GUI) para renderizar janelas do OpenCV. Para visualizar imagens processadas com OpenCV no Colab, é necessário converter a imagem para o formato RGB (se for BGR) e então usar `plt.imshow()` do Matplotlib.

- **cv2.calcHist()**: Calcula o histograma de uma imagem, fornecendo uma representação da distribuição de intensidade dos pixels ou dos canais de cor. Os parâmetros chave incluem `images` (uma lista de arrays de imagem), `channels` (os índices dos canais a serem analisados, e.g., `[0]` para azul, `[1]` para verde, `[2]` para vermelho em BGR), `mask` (uma máscara opcional para calcular o histograma apenas em uma região específica), `histSize` (o número de bins para cada dimensão do histograma) e `ranges` (o intervalo de valores de pixel a serem considerados). O resultado de `cv2.calcHist()` é um array NumPy que pode ser facilmente plotado com `plt.plot()` para visualização.

- **cv2.equalizeHist()**: Esta função é utilizada para melhorar o contraste de uma imagem em escala de cinza. Ela opera esticando o histograma de intensidade da imagem para cobrir toda a faixa dinâmica disponível, resultando em uma imagem com contraste aprimorado. A função aceita uma imagem de entrada (que deve estar em escala de cinza) e retorna a imagem com o histograma equalizado.

- **Rotação de Imagens (cv2.getRotationMatrix2D e cv2.warpAffine)**:
  - **cv2.getRotationMatrix2D(center, angle, scale)**: Calcula uma matriz de transformação 2x3 que pode ser usada para rotação, escala e translação de uma imagem. O `angle` é especificado em graus, com valores positivos indicando rotação anti-horária.
  - **cv2.warpAffine(src, M, dsize)**: Aplica a transformação afim definida pela matriz M (obtida de getRotationMatrix2D ou outras funções) a uma imagem de origem src, produzindo uma imagem de destino com o tamanho especificado por dsize.

A principal consideração ao trabalhar com OpenCV no Google Colab é a sinergia entre as bibliotecas e as limitações do ambiente. Embora o OpenCV seja a ferramenta de escolha para processar imagens (leitura, manipulação, aplicação de filtros), o Matplotlib é a ferramenta ideal para visualizar essas imagens no ambiente baseado em navegador do Colab. A saída de quase todas as funções do OpenCV é um array NumPy, que é o formato de entrada nativo para `plt.imshow()`. Isso estabelece um pipeline de trabalho muito eficiente: `cv2.imread` -> `cv2.processamento` -> `plt.imshow`. A necessidade de converter imagens de BGR para RGB para exibição com Matplotlib e a impossibilidade de usar `cv2.imshow` diretamente no Colab são pontos cruciais que devem ser compreendidos para um fluxo de trabalho eficaz.

```python
# Exemplo de carregamento, histograma e equalização de imagem
# Baixar uma imagem de exemplo (se não tiver uma localmente)
!wget -O sample_image.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Clouds_over_the_Atlantic_Ocean.jpg/640px-Clouds_over_the_Atlantic_Ocean.jpg

# Carregar imagem com OpenCV (lê em BGR por padrão)
img_bgr = cv2.imread('sample_image.jpg')
if img_bgr is None:
    print("Erro: Imagem não encontrada. Verifique o caminho.")
else:
    # Converter de BGR para RGB para exibição com Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Converter para escala de cinza para equalização de histograma
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Equalizar histograma
    img_equalized = cv2.equalizeHist(img_gray)
    
    # Calcular histograma do canal de luminância
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # Plotar imagens e histograma
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title('Escala de Cinza')
    axes[1].axis('off')
    
    axes[2].imshow(img_equalized, cmap='gray')
    axes[2].set_title('Histograma Equalizado')
    axes[2].axis('off')
    plt.show()
    
    # Plotar o histograma calculado
    plt.figure(figsize=(8, 5))
    plt.plot(hist_gray, color='black')
    plt.title('Histograma da Imagem em Escala de Cinza')
    plt.xlabel('Intensidade do Pixel')
    plt.ylabel('Frequência')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    # Exemplo de Rotação de Imagem
    if img_bgr is not None:
        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        angle = 45  # Rotação de 45 graus
        scale = 1.0  # Sem escala
        
        # Obter a matriz de rotação
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Aplicar a rotação
        rotated_img_bgr = cv2.warpAffine(img_bgr, M, (w, h))
        rotated_img_rgb = cv2.cvtColor(rotated_img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(rotated_img_rgb)
        plt.title(f'Imagem Rotacionada {angle} Graus')
        plt.axis('off')
        plt.show()
```

##### Funções OpenCV para Processamento de Imagens (Pré-processamento)

| Função OpenCV | Propósito | Parâmetros Chave | Saída (Formato) | Compatibilidade Colab |
|---------------|-----------|------------------|-----------------|----------------------|
| imread() | Carrega uma imagem de arquivo. | filename, flags (ex: cv2.IMREAD_COLOR). | numpy.ndarray (BGR). | Sim (para leitura). |
| imshow() | Exibe uma imagem em uma janela. | window_name, image_array. | None. | Não (use plt.imshow no Colab). |
| calcHist() | Calcula o histograma de uma imagem. | images, channels, mask, histSize, ranges. | numpy.ndarray (float32). | Sim (o resultado pode ser plotado com Matplotlib). |
| equalizeHist() | Equaliza o histograma de uma imagem em escala de cinza. | src (imagem de entrada), dst (imagem de saída). | numpy.ndarray (escala de cinza). | Sim (o resultado pode ser plotado com Matplotlib). |
| getRotationMatrix2D() | Calcula a matriz de rotação 2D. | center, angle, scale. | numpy.ndarray (matriz 2x3). | Sim. |
| warpAffine() | Aplica uma transformação afim a uma imagem. | src, M (matriz de transformação), dsize (tamanho de saída). | numpy.ndarray. | Sim (o resultado pode ser plotado com Matplotlib). |

#### Interatividade e Salvamento de Plots

O modelo de execução do Google Colab com `%matplotlib inline` implica que os gráficos são renderizados como imagens estáticas no momento em que a célula é executada. Diferentemente de um ambiente Python interativo local que pode abrir uma janela de gráfico dinamicamente atualizável, no Colab, cada célula de plotagem é um "instantâneo" do estado do gráfico. Isso significa que comandos de plotagem em células subsequentes não afetarão gráficos que já foram exibidos em células anteriores. Para modificar um gráfico, todas as alterações e chamadas de plotagem devem ser realizadas dentro da mesma célula onde o gráfico foi criado. Esta característica do ambiente incentiva a encapsulação de um gráfico completo em uma única célula para maior clareza e previsibilidade.

Para salvar os gráficos gerados, as funções `plt.savefig('nome_do_arquivo.png')` ou `fig.savefig('nome_do_arquivo.pdf')` podem ser utilizadas. Os arquivos serão salvos no sistema de arquivos temporário do ambiente Colab e podem ser baixados posteriormente para uso local.

```python
# Exemplo de salvamento de plot
x = np.linspace(0, 5, 100)
y = x**2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label='y = x^2')
ax.set_title('Gráfico para Salvar')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.grid(True)

# Salvar o gráfico como PNG
plt.savefig('meu_primeiro_grafico.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'meu_primeiro_grafico.png'")
plt.show()  # Exibir o gráfico após salvar
```
