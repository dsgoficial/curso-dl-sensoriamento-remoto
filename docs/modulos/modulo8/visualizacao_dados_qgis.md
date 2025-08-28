---
sidebar_position: 1
title: "Visualização de Dados no QGIS"
description: "Explorando e visualizando dados geoespaciais no QGIS para sensoriamento remoto"
tags: [qgis, visualização, geoespacial, sensoriamento remoto]
---

# Acesso e Download aos Dados

Pasta com os dados do curso: https://drive.google.com/drive/folders/1-yvUNeT4JoeZCsjWhC8vtu2MMhn8ANcD?usp=share_link

# Visualizar os dados no QGIS

Abrir os dados e visualizar. Selecionar as camadas.

# Guia de Instalação do DSGTools via GitHub

## Sobre o DSGTools

O DSGTools é um plugin desenvolvido pelo Exército Brasileiro para o QGIS, focado em facilitar o trabalho com dados geoespaciais conforme especificações da Infraestrutura Nacional de Dados Espaciais (INDE) e padrões militares.

## Pré-requisitos

- QGIS instalado (versão 3.x recomendada)
- Acesso à internet para download
- Permissões de administrador (em alguns casos)

## Método 1: Download Direto do ZIP

### Passo 1: Acessar o Repositório
1. Acesse o repositório oficial: https://github.com/dsgoficial/DsgTools
2. Clique no botão verde **"Code"**
3. Selecione **"Download ZIP"**

### Passo 2: Extrair o Arquivo
1. Após o download, extraia o arquivo ZIP
2. Dentro da pasta extraída de `DsgTools-main`, copie a pasta `DsgTools`

### Passo 3: Localizar a Pasta de Plugins do QGIS

**Windows:**
```
C:\Users\[seu_usuario]\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\
```

**Linux:**
```
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

**macOS:**
```
~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/
```

### Passo 4: Copiar o Plugin
1. Copie a pasta `DsgTools` para a pasta de plugins identificada no passo anterior
2. A estrutura final deve ser: `[pasta_plugins]/DsgTools/`

# Tutorial: Clipar Imagens usando DSGTools no QGIS

## Pré-requisitos

- QGIS 3.x instalado
- Plugin DSGTools instalado e habilitado
- Imagens raster para clipar (formato .tif, .img, etc.)
- Conhecimento básico do QGIS

# Tutorial: Instalação do Plugin QuickMapServices no QGIS

O QuickMapServices é um dos plugins mais populares do QGIS, permitindo adicionar facilmente mapas base de diferentes provedores (OpenStreetMap, Google Maps, Bing, etc.) aos seus projetos.

## Pré-requisitos

- QGIS instalado (versão 3.0 ou superior recomendada)
- Conexão com a internet

## Passo a Passo

### 1. Acessar o Gerenciador de Plugins

1. Abra o QGIS
2. No menu superior, clique em **Plugins**
3. Selecione **Gerenciar e Instalar Plugins** (ou use o atalho `Ctrl+Shift+T`)

### 2. Buscar o Plugin

1. Na janela que abrir, clique na aba **Todos** no menu lateral esquerdo
2. No campo de pesquisa no topo, digite: `QuickMapServices`
3. O plugin aparecerá na lista com o nome "QuickMapServices"

### 3. Instalar o Plugin

1. Clique no plugin **QuickMapServices** para selecioná-lo
2. Clique no botão **Instalar Plugin** no canto inferior direito
3. Aguarde o download e instalação automática

### 4. Verificar a Instalação

Após a instalação, você verá:
- Uma nova opção **Web** no menu superior do QGIS
- Dentro do menu Web, a opção **QuickMapServices**

### 5. Configurar Serviços Adicionais (Opcional)

Para acessar mais provedores de mapas:

1. Vá em **Web > QuickMapServices > Settings**
2. Na aba **More services**, clique em **Get contributed pack**
3. Isso adicionará serviços como Google Maps, Bing Maps, etc.

## Como Usar

Após a instalação:

1. Vá em **Web > QuickMapServices**
2. Escolha um provedor (ex: OpenStreetMap, Google Satellite, etc.)
3. A camada será automaticamente adicionada ao seu projeto

## Dicas Importantes

- **Projeção**: O plugin funciona melhor com projetos em EPSG:3857 (Web Mercator)
- **Internet**: É necessária conexão com internet para carregar os mapas
- **Uso Comercial**: Verifique os termos de uso de cada provedor, especialmente para uso comercial

## Solução de Problemas Comuns

**Plugin não aparece após instalação:**
- Reinicie o QGIS
- Verifique se o plugin está habilitado em Plugins > Gerenciar e Instalar Plugins > Instalados

**Mapas não carregam:**
- Verifique sua conexão com internet
- Certifique-se de que o projeto está na projeção correta
- Alguns provedores podem ter restrições regionais

## Parte 1: Instalação e Preparação do DSGTools

### 1.1 Instalar o DSGTools
1. Abra o QGIS
2. Vá em **Complementos** → **Gerenciar e instalar complementos**
3. Na aba **Todos**, procure por "DSGTools"
4. Clique em **Instalar complemento**
5. Após a instalação, verifique se aparece a barra de ferramentas DSGTools

### 1.2 Preparar o Projeto
1. Crie um novo projeto no QGIS
2. Defina o Sistema de Referência de Coordenadas (SRC) apropriado para sua região
3. Carregue suas imagens raster no projeto

## Parte 2: Criando a Camada de Enquadramento Sistemático

### 2.1 Acessar a Ferramenta de Enquadramento
1. Na barra de ferramentas do DSGTools, localize o ícone de **Enquadramento Sistemático**
2. Ou vá em **DSGTools** → **Ferramentas de Produção** → **Criar Enquadramento Sistemático**

### 2.2 Configurar o Enquadramento
1. Na janela **Criar Enquadramento Sistemático**:
   - **Escala**: Escolha a escala desejada (ex: 1:25.000, 1:50.000)
   - **Tipo de Enquadramento**: Selecione o padrão brasileiro (MI, MIR, etc.)
   - **Extensão**: Defina a área de interesse ou use "Usar extensão da tela"
   - **Sistema de Coordenadas**: Confirme o SRC correto

2. **Configurações Avançadas**:
   - **Prefixo**: Digite um prefixo para nomear as feições (ex: "FOLHA_")
   - **Criar índice espacial**: Marque esta opção para melhor performance

### 2.3 Gerar o Enquadramento
1. Clique em **OK** para gerar o enquadramento
2. Uma nova camada vetorial será criada com a grade sistemática
3. Renomeie a camada para algo identificável como "Enquadramento_Sistemático"

## Parte 3: Configurando as Imagens e Camadas

### 3.1 Organizar as Camadas
1. No **Painel de Camadas**, organize da seguinte forma:
   - Camada de enquadramento (superior)
   - Imagens raster (abaixo)

### 3.2 Configurar Simbologia do Enquadramento
1. Clique com botão direito na camada de enquadramento → **Propriedades**
2. Na aba **Simbologia**:
   - Remova o preenchimento (transparente)
   - Configure contorno visível (cor contrastante)
   - Adicione rótulos com os nomes das folhas

## Parte 4: Processo de Clipping por Feição

### 4.1 Preparar para o Clipping
1. Certifique-se de que a camada de enquadramento está selecionada
2. Ative a ferramenta de seleção: **Selecionar feições por área ou clique único**

### 4.2 Clipar Cada Feição Individualmente

Para cada feição do enquadramento, execute os seguintes passos:

#### Passo 1: Selecionar a Feição
1. Clique na feição desejada da grade de enquadramento
2. A feição ficará destacada (geralmente em amarelo)

#### Passo 2: Acessar a Ferramenta de Clipping
1. Vá em **Raster** → **Extração** → **Clipar raster por camada de máscara**
2. Ou use a **Caixa de Ferramentas de Processamento** (Ctrl+Alt+T) e procure por "Clip"

#### Passo 3: Configurar o Clipping
Na janela **Clipar Raster por Camada de Máscara**:

1. **Camada de entrada**: Selecione sua imagem raster
2. **Camada de máscara**: Selecione a camada de enquadramento
3. **Marcar**: ☑️ **Usar apenas feições selecionadas**
4. **Valor NoData**: -9999 (ou deixe padrão)
5. **Criar máscara alfa**: ☑️ (recomendado)
6. **Manter resolução da camada de origem**: ☑️

#### Passo 4: Definir Arquivo de Saída
1. Em **Clipped (mask)**, clique nos três pontinhos **...**
2. **Salvar para arquivo** → Escolha o local e nome
3. **Formato**: Selecione **GeoTIFF (*.tif)**
4. **Nome do arquivo**: Use uma convenção como `FOLHA_[NOME]_clipped.tif`

#### Passo 5: Executar o Clipping
1. Clique em **Executar**
2. Aguarde o processamento
3. A imagem clipped será adicionada ao projeto automaticamente

### 4.3 Repetir o Processo
1. Limpe a seleção atual: **Selecionar** → **Desselecionar feições de todas as camadas**
2. Selecione a próxima feição
3. Repita os passos 4.2 até clipar todas as feições necessárias