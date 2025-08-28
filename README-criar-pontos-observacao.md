# README - Criar Pontos de Observação

Este documento contém as instruções para configurar o ambiente necessário para executar o projeto de criação de pontos de observação.

## Pré-requisitos do Sistema

### 1. Instalar Ferramentas de Compilação

O projeto requer o compilador C (gcc) para instalar algumas dependências Python. Execute:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

### 2. Instalar Graphviz

O projeto utiliza o pygraphviz que depende do Graphviz. Instale:

```bash
sudo apt-get install graphviz graphviz-dev
```

## Configuração do Ambiente Python

### 1. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências Python

Com o ambiente virtual ativado, instale as dependências:

```bash
pip install -r requirements-criar-pontos-observacao.txt
```

## Dependências Principais

O projeto utiliza as seguintes bibliotecas principais:

- **pandas** (2.3.1) - Manipulação de dados
- **numpy** (2.2.6) - Computação numérica
- **matplotlib** (3.10.5) - Visualização de dados
- **networkx** (3.4.2) - Análise de redes
- **pygraphviz** (1.14) - Visualização de grafos
- **scipy** (1.15.3) - Computação científica
- **shapely** (2.1.1) - Geometria computacional
- **sympy** (1.14.0) - Matemática simbólica

## Solução de Problemas

### Erro de Compilação do pygraphviz

Se encontrar erro ao instalar o `pygraphviz`, certifique-se de que:

1. O `build-essential` está instalado
2. O `python3-dev` está instalado
3. O `graphviz` e `graphviz-dev` estão instalados

### Verificar Instalação

Para verificar se tudo foi instalado corretamente:

```bash
python -c "import pygraphviz; print('pygraphviz instalado com sucesso')"
python -c "import pandas; print('pandas instalado com sucesso')"
python -c "import numpy; print('numpy instalado com sucesso')"
```

## Estrutura do Projeto

```
RoverModel/
├── venv/                           # Ambiente virtual Python
├── requirements-criar-pontos-observacao.txt  # Dependências Python
└── README-criar-pontos-observacao.md        # Este arquivo
```

## Notas

- Este projeto foi testado no Ubuntu 22.04 LTS (WSL2)
- Python 3.10+ é recomendado
- Todas as dependências estão fixadas em versões específicas para garantir compatibilidade
