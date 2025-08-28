---
sidebar_position: 3
title: "Concorrência e Paralelismo no Processamento de Imagens"
description: "Otimizando pipelines de deep learning com ThreadPoolExecutor e ProcessPoolExecutor"
tags: [paralelismo, concorrência, processamento de imagens, python]
---

# **Concorrência e Paralelismo em Deep Learning: Um Guia Definitivo para ThreadPoolExecutor e ProcessPoolExecutor no Processamento de Imagens**

## **1. Introdução: O Dilema da Otimização em Python e o Papel do GIL**

O desenvolvimento de aplicações de deep learning frequentemente lida com a necessidade de processar grandes volumes de dados, em particular imagens e máscaras de segmentação. A eficiência com que essa fase de pré-processamento é conduzida pode ser um gargalo significativo, limitando a velocidade de treinamento e a produtividade. Otimizar a ingestão e a transformação de dados é, portanto, um componente crítico para qualquer pipeline de aprendizado profundo em escala. Para alcançar essa otimização, é fundamental distinguir entre dois conceitos-chave: concorrência e paralelismo. A concorrência refere-se à capacidade de um sistema gerenciar múltiplas tarefas em um único núcleo de processamento, dando a ilusão de que estão ocorrendo simultaneamente. O paralelismo, por outro lado, é a execução verdadeira de múltiplas tarefas ao mesmo tempo, utilizando vários núcleos de processamento.¹

O módulo `concurrent.futures` foi projetado para abstrair as complexidades do gerenciamento de threads e processos, oferecendo uma interface de alto nível para a execução de chamadas de função de forma assíncrona.²

No contexto do Python, o caminho para o paralelismo não é direto. A razão reside em uma característica intrínseca do interpretador CPython: o Global Interpreter Lock (GIL). O GIL é um mecanismo de bloqueio que permite que apenas um thread execute bytecode Python por vez, mesmo em sistemas equipados com múltiplos núcleos de CPU.³ Seu propósito principal é garantir a consistência dos dados internos do interpretador e simplificar o gerenciamento de memória, tornando a maioria das operações nativas thread-safe.³ Essa limitação, no entanto, impede o paralelismo real para tarefas de uso intensivo de CPU que dependem exclusivamente de código Python.⁴

A distinção entre tipos de tarefa é a base para a escolha entre `ThreadPoolExecutor` e `ProcessPoolExecutor`. As tarefas podem ser classificadas como **I/O-Bound** (vinculadas a E/S) ou **CPU-Bound** (vinculadas à CPU). Tarefas I/O-Bound, como requisições de rede, leitura de arquivos ou acesso a um banco de dados, passam a maior parte do tempo em um estado de espera, aguardando por uma operação externa. Durante esse tempo de espera, o thread libera o GIL, permitindo que outro thread execute o código Python.⁴ Por isso, o multithreading é uma abordagem eficaz para lidar com tarefas de E/S, pois ele consegue mascarar a latência da espera.¹ Já as tarefas CPU-Bound, como cálculos matemáticos complexos ou manipulação intensiva de pixels em uma imagem, dependem de um poder de processamento puro. Para essas tarefas, o GIL representa um obstáculo direto ao paralelismo, uma vez que impede a execução simultânea de threads em diferentes núcleos.⁴ A solução para este problema reside na utilização de processos, pois cada processo é uma instância separada do interpretador Python e, portanto, possui seu próprio GIL, permitindo a execução paralela de código Python.⁴

## **2. A Concorrência em Ação: Otimizando Tarefas I/O-Bound com ThreadPoolExecutor**

O `ThreadPoolExecutor` é uma classe poderosa, disponível no módulo `concurrent.futures`, que utiliza um pool de threads de trabalho para a execução assíncrona de tarefas.⁷ A principal vantagem de um pool de threads é a redução do overhead de criação e destruição de threads para cada tarefa, uma vez que threads ociosas são reutilizadas à medida que novas tarefas são submetidas.⁶ Essa abordagem é ideal para cenários I/O-Bound, como a leitura de um grande conjunto de dados de imagens de um disco, pois as threads podem se revezar na posse do GIL enquanto aguardam a conclusão das operações de E/S.²

A classe `ThreadPoolExecutor` é instanciada com um número máximo de threads de trabalho. A partir do Python 3.8, o valor padrão para o parâmetro `max_workers` foi alterado para `min(32, os.cpu_count() + 4)`, um valor que busca preservar um número mínimo de workers para tarefas de E/S, evitando o uso excessivo de recursos em sistemas com muitos núcleos.⁶ A interface de programação oferece dois métodos principais para submeter tarefas ao pool.

O método `submit(fn, *args, **kwargs)` é utilizado para agendar a execução de uma função com seus argumentos e retorna um objeto `Future`. Este objeto `Future` atua como um manipulador para a execução assíncrona da tarefa, permitindo verificar seu status ou recuperar o resultado quando a execução for concluída.² Já o método `map(fn, *iterables)` aplica uma função a cada item de um ou mais iteráveis de forma concorrente. Ele retorna um iterador que produz os resultados na mesma ordem em que os dados de entrada foram fornecidos, simplificando consideravelmente a sintaxe do código.²

O gerenciador de contexto `with...as...` é a maneira mais recomendada de usar o `ThreadPoolExecutor`, pois garante que o pool de threads seja automaticamente encerrado e que todos os recursos sejam liberados assim que a execução do bloco de código for concluída, mesmo em caso de exceções.¹¹

Abaixo, um exemplo prático que demonstra a aplicação de `ThreadPoolExecutor` para carregar um conjunto de imagens de um diretório de forma concorrente:

```python
import concurrent.futures
import time
from PIL import Image
import os

# Função para simular uma tarefa de E/S (carregamento de imagem do disco)
def carregar_imagem(caminho_imagem):
    try:
        # Abre e fecha a imagem para simular o carregamento de dados do disco
        with Image.open(caminho_imagem) as img:
            print(f"> Imagem carregada: {os.path.basename(caminho_imagem)}")
            # Retorna o tamanho da imagem como resultado
            return img.size
    except Exception as e:
        print(f"Erro ao carregar {caminho_imagem}: {e}")
        return None

if __name__ == "__main__":
    # Suponha que temos uma lista de caminhos de imagens
    caminhos_imagens = [f"path/to/image_{i}.jpg" for i in range(10)]
    
    start_time = time.time()
    
    # Utilizando o ThreadPoolExecutor para carregar as imagens de forma concorrente
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # A função as_completed retorna os futuros à medida que são concluídos
        # permitindo processar os resultados de forma assíncrona
        futures = {executor.submit(carregar_imagem, caminho): caminho for caminho in caminhos_imagens}
        
        for future in concurrent.futures.as_completed(futures):
            caminho_original = futures[future]
            try:
                tamanho_imagem = future.result()
                if tamanho_imagem:
                    print(f"Processado: {os.path.basename(caminho_original)} com tamanho {tamanho_imagem}")
            except Exception as e:
                print(f"Exceção para {os.path.basename(caminho_original)}: {e}")
                
    end_time = time.time()
    print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
```

## **3. O Verdadeiro Paralelismo: Acelerando Tarefas CPU-Bound com ProcessPoolExecutor**

Enquanto o `ThreadPoolExecutor` é a ferramenta de escolha para tarefas de espera, o `ProcessPoolExecutor` é a solução para a execução de tarefas CPU-Bound em paralelo real.² A classe `ProcessPoolExecutor` opera de forma análoga ao `ThreadPoolExecutor`, mas em vez de threads, ela gerencia um pool de processos separados.⁴ Cada processo é uma nova instância do interpretador Python, o que significa que cada um possui seu próprio Global Interpreter Lock. Esta arquitetura permite que as tarefas sejam executadas simultaneamente em diferentes núcleos de CPU, contornando a principal limitação do GIL.⁴ A reutilização de processos do pool também minimiza o custo computacional de iniciar e encerrar novos processos repetidamente, o que é um processo dispendioso.¹²

Uma consequência da arquitetura baseada em processos separados é a forma como os dados são comunicados. Para que uma função e seus argumentos sejam enviados a um processo de trabalho, eles precisam ser serializados e copiados. Da mesma forma, os resultados são serializados e copiados de volta para o processo principal.⁴ Essa exigência de serialização impõe uma restrição fundamental: a função e todos os seus argumentos devem ser "picklable".¹⁰ Isso significa que certas construções, como funções lambda ou funções definidas no interpretador interativo (REPL), não são suportadas pelo `ProcessPoolExecutor`.¹⁰ Outra consideração técnica importante é que o módulo `__main__` deve ser importável pelos subprocessos de trabalho.¹⁰

O método `map()` do `ProcessPoolExecutor` oferece um parâmetro adicional e crucial: `chunksize`. Para grandes iteráveis, o desempenho pode ser significativamente aprimorado ao definir `chunksize` para um valor maior que o padrão 1.¹⁰ Isso ocorre porque o overhead de comunicação inter-processo é alto, pois envolve a serialização e o envio de dados. Ao agrupar tarefas em blocos maiores (chunks), a sobrecarga é reduzida, pois um único pacote de dados é enviado em vez de vários pequenos pacotes. Por outro lado, para o `ThreadPoolExecutor`, que usa memória compartilhada, não há esse overhead de serialização de dados, o que torna o parâmetro `chunksize` irrelevante para o seu desempenho.¹⁰

A seguir, um exemplo que demonstra o uso de `ProcessPoolExecutor` para uma tarefa de pré-processamento intensiva em CPU, como o redimensionamento e a normalização de uma lista de imagens.

```python
import concurrent.futures
import time
from PIL import Image
import numpy as np
import os

# Função que realiza uma tarefa intensiva de CPU: redimensiona e normaliza a imagem
def processar_imagem(caminho_imagem):
    try:
        # Abertura da imagem
        with Image.open(caminho_imagem) as img:
            # Tarefa intensiva de CPU: redimensionamento
            img_redimensionada = img.resize((224, 224))
            
            # Tarefa intensiva de CPU: conversão e normalização
            np_array = np.array(img_redimensionada) / 255.0
            
            print(f"> Imagem processada: {os.path.basename(caminho_imagem)}")
            return np_array.shape
    except Exception as e:
        print(f"Erro ao processar {caminho_imagem}: {e}")
        return None

if __name__ == "__main__":
    # Suponha que temos uma lista de caminhos de imagens
    caminhos_imagens = [f"path/to/image_{i}.jpg" for i in range(10)]
    
    start_time = time.time()
    
    # Utilizando o ProcessPoolExecutor para processar as imagens em paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Usando map para aplicar a função a cada caminho
        resultados = executor.map(processar_imagem, caminhos_imagens)
        
        for resultado in resultados:
            if resultado:
                print(f"Dimensões finais da imagem: {resultado}")
                
    end_time = time.time()
    print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
```

## **4. Análise Comparativa: ThreadPoolExecutor vs. ProcessPoolExecutor**

A escolha entre `ThreadPoolExecutor` e `ProcessPoolExecutor` não é arbitrária. A decisão deve ser baseada na natureza da tarefa a ser executada. A Tabela 1 sintetiza as principais diferenças e os cenários de uso ideais para cada classe.

| Característica | ThreadPoolExecutor | ProcessPoolExecutor |
| :---- | :---- | :---- |
| **Tipo de Worker** | Threads | Processos |
| **Impacto do GIL** | Limitado (apenas uma thread executa código Python por vez) | Contornado (cada processo tem seu próprio GIL) |
| **Cenário Ideal** | Tarefas I/O-Bound (leitura de arquivos, requisições de rede) | Tarefas CPU-Bound (cálculos matemáticos, manipulação de pixels) |
| **Consumo de Memória** | Memória compartilhada entre threads dentro de um único processo | Memória separada para cada processo |
| **Comunicação de Dados** | Direta (memória compartilhada) | Cópia/Serialização (pickling) entre processos |
| **Overhead de Inicialização** | Baixo | Alto |
| **Parâmetro chunksize** | Sem efeito no desempenho de map() | Melhora significativa o desempenho de map() para grandes iteráveis |
| **Restrições Adicionais** | Possibilidade de *deadlocks* se as tarefas esperam por outras Futures | Requer que funções e argumentos sejam "picklable" e que \_\_main\_\_ seja importável |

A principal conclusão que se extrai desta comparação é que a arquitetura do `ThreadPoolExecutor` é inerentemente mais eficiente para tarefas de E/S devido à sua capacidade de compartilhar memória de forma direta, eliminando a sobrecarga de serialização. Em contrapartida, o `ProcessPoolExecutor` brilha em tarefas de computação intensiva, onde o paralelismo real é alcançado, superando a barreira imposta pelo GIL.

Portanto, em um pipeline de processamento de imagens, a regra de ouro é:

* Use o `ThreadPoolExecutor` para tarefas como o carregamento de imagens de um disco (uma operação I/O-Bound) ou para fazer chamadas de rede para APIs de modelos.  
* Use o `ProcessPoolExecutor` para qualquer manipulação de pixel, como redimensionamento, recorte, aplicação de filtros, detecção de características ou normalização de dados, que são todas operações CPU-Bound.¹⁴

## **5. Estudo de Caso Avançado: Construindo um Pipeline Otimizado de Processamento de Imagens e Máscaras**

Para demonstrar a sinergia entre as duas classes, considere o cenário de otimizar um data loader para um modelo de segmentação semântica, onde cada entrada consiste em um par de imagem e máscara. O fluxo de trabalho envolve uma mistura de operações de E/S e de CPU.

1. **Carregamento de Dados (I/O-Bound)**: A primeira etapa é carregar o par de imagem e máscara do disco. Esta é uma operação de E/S clássica. Utilizar o `ThreadPoolExecutor` é a escolha ideal para carregar esses dados de forma concorrente. Enquanto um thread está esperando o disco responder para carregar uma imagem, outro pode ser agendado para carregar a próxima, mantendo a CPU ocupada e maximizando a taxa de transferência de dados.¹  
2. **Pré-processamento de Dados (CPU-Bound)**: Uma vez que as imagens e máscaras estão na memória, elas precisam ser processadas. As transformações comuns incluem redimensionamento para um tamanho fixo, normalização dos valores de pixel ou aplicação de técnicas de aumento de dados como rotação e flip. Cada uma dessas operações é intensiva em CPU e pode ser executada em paralelo de forma eficiente.¹⁴ O `ProcessPoolExecutor` é perfeitamente adequado para esta etapa, permitindo que cada imagem seja processada em um núcleo de CPU diferente.

Uma abordagem completa combinaria as duas etapas: a leitura de um lote de arquivos com `ThreadPoolExecutor` e, em seguida, o processamento desse lote com `ProcessPoolExecutor`.

Uma análise de desempenho comparativa, como a proposta na Tabela 2, ilustraria o ganho substancial de desempenho obtido com essa abordagem híbrida.

| Metodologia | Descrição | Tempo de Execução (em segundos) |
| :---- | :---- | :---- |
| **Abordagem Serial** | Leitura e processamento de cada imagem em um loop for simples. | Tserial |
| **ThreadPoolExecutor** | Leitura concorrente com threads, processamento serial. | Tthreads |
| **ProcessPoolExecutor** | Leitura serial, processamento paralelo com processos. | Tprocessos |
| **Abordagem Híbrida** | Leitura concorrente com threads, processamento paralelo com processos. | Thíbrido |

A expectativa é que o `ThreadPoolExecutor` ofereça um ganho marginal (ou até mesmo uma perda) na etapa de pré-processamento (CPU-Bound) devido à sobrecarga de gerenciamento de threads sem os benefícios do paralelismo. Por outro lado, o `ProcessPoolExecutor` demonstraria um ganho de desempenho significativo em comparação com a abordagem serial ou com threads para a mesma tarefa. O tempo total da abordagem híbrida seria o mais baixo, pois otimiza ambas as naturezas de tarefas.

## **6. Diretrizes e Melhores Práticas**

A utilização eficiente do módulo `concurrent.futures` requer a adoção de algumas práticas recomendadas:

* **Gerenciamento de Recursos com Gerenciadores de Contexto**: A utilização do gerenciador de contexto (with...as...) é a forma mais segura de usar os pools.⁷ Ele garante que, independentemente de como o bloco de código é finalizado (normalmente ou por uma exceção), o método `shutdown()` do executor será chamado, liberando todos os recursos e garantindo que os workers sejam encerrados corretamente.¹²  
* **Dimensionamento de max\_workers**: Para `ThreadPoolExecutor`, um número maior de threads geralmente é benéfico em tarefas I/O-Bound, pois a maioria dos workers estará em estado de espera. Já para o `ProcessPoolExecutor`, a prática ideal é não exceder o número de núcleos de CPU disponíveis. Definir um número de workers muito maior que o número de núcleos pode levar a uma queda de desempenho devido à sobrecarga de troca de contexto (context switching) do sistema operacional, uma vez que o sistema operacional gasta mais tempo gerenciando processos do que executando tarefas.⁶  
* **Tratamento de Exceções**: Quando uma função submetida a um executor levanta uma exceção, o objeto `Future` capturará essa exceção.¹⁶ Para lidar com isso, pode-se usar o método `Future.result()` dentro de um bloco try...except ou o método `Future.exception()` para inspecionar a exceção sem interromper a execução do programa principal.² O uso da função `concurrent.futures.as_completed()` é uma excelente estratégia, pois permite processar os resultados e tratar as exceções à medida que as tarefas são concluídas, sem a necessidade de esperar que todas as tarefas terminem.²  
* **Restrições do ProcessPoolExecutor**: É fundamental lembrar que o `ProcessPoolExecutor` não funcionará corretamente em ambientes interativos como o console Python, e que as funções a serem executadas não podem ser definidas de forma lambda nem aninhadas em outra função, a menos que sejam serializáveis.

## **7. Conclusão**

A otimização do processamento de imagens para pipelines de deep learning vai além do uso de bibliotecas de alto desempenho como NumPy ou PyTorch. A verdadeira eficiência é alcançada ao se empregar a concorrência e o paralelismo de forma estratégica, aproveitando as ferramentas certas para cada tipo de tarefa.

Este guia demonstra que a escolha entre `ThreadPoolExecutor` e `ProcessPoolExecutor` é um reflexo direto da compreensão do comportamento do Global Interpreter Lock (GIL) e da natureza das tarefas a serem executadas. O `ThreadPoolExecutor`, com sua arquitetura baseada em threads e memória compartilhada, é a ferramenta ideal para tarefas I/O-Bound, como o carregamento de dados de imagens. Por outro lado, o `ProcessPoolExecutor`, com sua capacidade de contornar o GIL por meio de processos separados, é a escolha superior para tarefas CPU-Bound, como o pré-processamento e a manipulação intensiva de pixels.

Ao combinar a leitura de arquivos concorrente com o `ThreadPoolExecutor` e o processamento paralelo com o `ProcessPoolExecutor`, é possível construir pipelines de dados robustos e significativamente mais rápidos. O domínio dessas ferramentas de alto nível, disponíveis na biblioteca padrão do Python, capacita o desenvolvedor a escrever código mais limpo e eficiente, permitindo que o foco principal permaneça na arquitetura e no treinamento do modelo de deep learning.

#### **Referências citadas**

1. Python Multithreading vs. Multiprocessing Explained \- Built In, acessado em agosto 28, 2025, [https://builtin.com/data-science/multithreading-multiprocessing](https://builtin.com/data-science/multithreading-multiprocessing)  
2. Introduction to concurrent.futures in Python | by smrati katiyar \- Medium, acessado em agosto 28, 2025, [https://medium.com/@smrati.katiyar/introduction-to-concurrent-futures-in-python-009fe1d4592c](https://medium.com/@smrati.katiyar/introduction-to-concurrent-futures-in-python-009fe1d4592c)  
3. O Paralelismo no Python: Threads vs. Processos \- Parte 1 \- Revelo Community, acessado em agosto 28, 2025, [https://community.revelo.com.br/o-paralelismo-no-python-threads-vs-processos-parte-1/](https://community.revelo.com.br/o-paralelismo-no-python-threads-vs-processos-parte-1/)  
4. What is the difference between ProcessPoolExecutor and ThreadPoolExecutor? \- Stack Overflow, acessado em agosto 28, 2025, [https://stackoverflow.com/questions/51828790/what-is-the-difference-between-processpoolexecutor-and-threadpoolexecutor](https://stackoverflow.com/questions/51828790/what-is-the-difference-between-processpoolexecutor-and-threadpoolexecutor)  
5. Multiprocessamento Python: Um guia para threads e processos | DataCamp, acessado em agosto 28, 2025, [https://www.datacamp.com/pt/tutorial/python-multiprocessing-tutorial](https://www.datacamp.com/pt/tutorial/python-multiprocessing-tutorial)  
6. How to use ThreadPoolExecutor in Python3 ? \- GeeksforGeeks, acessado em agosto 28, 2025, [https://www.geeksforgeeks.org/python/how-to-use-threadpoolexecutor-in-python3/](https://www.geeksforgeeks.org/python/how-to-use-threadpoolexecutor-in-python3/)  
7. Concurrency in Python, Part VI — The concurrent.futures Module \- Medium, acessado em agosto 28, 2025, [https://medium.com/more-python/concurrency-in-python-part-vi-the-concurrent-futures-module-63cd5f05e8bc](https://medium.com/more-python/concurrency-in-python-part-vi-the-concurrent-futures-module-63cd5f05e8bc)  
8. Python ThreadPoolExecutor: 7-Day Crash Course | by Super Fast Python | Medium, acessado em agosto 28, 2025, [https://medium.com/@superfastpython/python-threadpoolexecutor-7-day-crash-course-78d4846d5acc](https://medium.com/@superfastpython/python-threadpoolexecutor-7-day-crash-course-78d4846d5acc)  
9. Como Utilizar O Threadpoolexecutor Em Python Para Otimizar A Execução De Tarefas, acessado em agosto 28, 2025, [https://awari.com.br/como-utilizar-o-threadpoolexecutor-em-python-para-otimizar-a-execucao-de-tarefas/](https://awari.com.br/como-utilizar-o-threadpoolexecutor-em-python-para-otimizar-a-execucao-de-tarefas/)  
10. concurrent.futures — Launching parallel tasks — Python 3.13.7 ..., acessado em agosto 28, 2025, [https://docs.python.org/3/library/concurrent.futures.html](https://docs.python.org/3/library/concurrent.futures.html)  
11. ThreadPoolExecutor Example in Python \- Super Fast Python, acessado em agosto 28, 2025, [https://superfastpython.com/threadpoolexecutor-example/](https://superfastpython.com/threadpoolexecutor-example/)  
12. ProcessPoolExecutor Class in Python \- GeeksforGeeks, acessado em agosto 28, 2025, [https://www.geeksforgeeks.org/python/processpoolexecutor-class-in-python/](https://www.geeksforgeeks.org/python/processpoolexecutor-class-in-python/)  
13. ThreadPoolExecutor vs ProcessPoolExecutor in Python, acessado em agosto 28, 2025, [https://superfastpython.com/threadpoolexecutor-vs-processpoolexecutor/](https://superfastpython.com/threadpoolexecutor-vs-processpoolexecutor/)  
14. Python Script for Batch Resizing Images \- Infotechys.com, acessado em agosto 28, 2025, [https://infotechys.com/python-batch-resize-images/](https://infotechys.com/python-batch-resize-images/)  
15. Using multi-threading to process an image faster on python? \- Stack Overflow, acessado em agosto 28, 2025, [https://stackoverflow.com/questions/8802916/using-multi-threading-to-process-an-image-faster-on-python](https://stackoverflow.com/questions/8802916/using-multi-threading-to-process-an-image-faster-on-python)  
16. How to store concurrent.futures ProcessPoolExecutor HTTP responses and process in real time? \- Stack Overflow, acessado em agosto 28, 2025, [https://stackoverflow.com/questions/70888034/how-to-store-concurrent-futures-processpoolexecutor-http-responses-and-process-i](https://stackoverflow.com/questions/70888034/how-to-store-concurrent-futures-processpoolexecutor-http-responses-and-process-i)