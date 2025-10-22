# Planejador de Missões Multi-Robô para Inspeção Autônoma

## 1. Visão Geral

Este projeto implementa um sistema completo para o planejamento e execução de missões de inspeção para múltiplos robôs autônomos. A solução utiliza um modelo de grafos para representar o ambiente, otimiza a alocação de tarefas entre os robôs e gera rotas eficientes para a execução.

O sistema é dividido em três componentes principais:

1.  **Geração do Grafo do Ambiente**: Cria um mapa navegável a partir de dados de obstáculos, gerando segmentos e pontos de passagem seguros.
2.  **Planejamento de Missão**: Recebe um conjunto de alvos de inspeção, distribui as tarefas entre os robôs disponíveis e calcula a rota ótima para cada um.
3.  **Servidor de Missão e Monitoramento**: Um servidor Flask com WebSocket que recebe solicitações de missão via API REST, envia os waypoints para os robôs e monitora sua execução em tempo real.

### Principais Funcionalidades

*   **Modelagem com Grafos**: Utiliza `NetworkX` para representar o ambiente, as missões e as capacidades dos robôs.
*   **Geração de Caminhos Seguros**: Cria um grafo de navegação a partir de AABBs (Axis-Aligned Bounding Boxes) que representam obstáculos.
*   **Alocação de Tarefas Balanceada**: Distribui os pontos de inspeção entre os robôs, considerando a posição inicial e as restrições de cada um para minimizar o tempo total.
*   **Otimização de Rota (TSP)**: Calcula a ordem de visitação mais eficiente para os pontos de cada robô usando o algoritmo do vizinho mais próximo.
*   **Execução e Simulação**: Envia as missões geradas para robôs em simulação (SITL) ou reais.
*   **Servidor Centralizado**: Gerencia e monitora as missões através de endpoints HTTP e eventos WebSocket.

## 2. Arquitetura do Projeto

O fluxo de trabalho do sistema é o seguinte:

1.  **`CriarPontosObservacao`**: Script inicial que processa uma planilha de obstáculos (`.xlsx`), gera os AABBs e cria um grafo de navegação (`graph.json`) e os pontos de observação (`observation_points.json`).
2.  **`PlanejadorHeterogeneoIntegrado.py`**: Orquestra o planejamento. Ele carrega o grafo do ambiente, os pontos de inspeção e as posições dos robôs. Em seguida, executa a clusterização de tarefas e o planejamento de rotas (TSP), gerando um plano de missão detalhado em formato JSON.
3.  **`SITL_planningsimulation.py`**: Exemplo de como consumir o plano de missão gerado, convertê-lo em waypoints GPS e enviá-lo para simuladores de robôs.
4.  **`mission_server.py`**: O servidor principal que expõe a funcionalidade de planejamento através de uma API REST. Ele integra os módulos de planejamento e monitoramento, permitindo que uma interface de usuário (frontend) inicie e acompanhe as missões.

### Módulos Principais

*   **`multigraphplanner.py`**: Contém a classe `MultiGraphPlanner` com lógicas para busca de caminho (A*), composição de autômatos e planejamento.
*   **`segmentutils.py`**: Classe utilitária para manipulação geométrica, criação de segmentos, geração de grafos a partir de AABBs e salvamento/carregamento de dados.
*   **`tspOptimization.py`**: Implementa o planejamento de alto nível, incluindo a clusterização de tarefas (`FixedTaskPlanner`) e a otimização da ordem de visita (TSP).
*   **`PlanejadorHeterogeneoIntegrado.py`**: Contém a função `run_planner` que integra todos os passos do planejamento offline.
*   **`mission_server.py`**: Servidor Flask/SocketIO para interação com o frontend e os robôs.

## 3. Como Executar

Existem duas maneiras principais de usar o projeto: executando o pipeline de planejamento offline ou interagindo com o servidor de missão.

### 3.1. Pré-requisitos

Primeiro, configure o ambiente. É recomendado o uso de um ambiente virtual Python.

1.  **Instalar dependências do sistema (Debian/Ubuntu):**
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential python3-dev graphviz graphviz-dev
    ```

2.  **Criar e ativar o ambiente virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Este `requirements` contém as bibliotecas principais como `pandas`, `networkx`, `shapely`, etc., usadas em todo o projeto).*

### 3.2. Executando o Planejamento Offline

Para gerar um plano de missão a partir do zero, você pode adaptar e executar o `PlanejadorHeterogeneoIntegrado.py`.

1.  **Gere o Grafo e os Pontos de Observação**:
    *   Execute o script `CriarPontosObservacao.py` (não fornecido no contexto, mas referenciado) para gerar `graph.json` e `observation_points.json` a partir de uma planilha de obstáculos.

2.  **Execute o Planejador**:
    *   Configure os parâmetros dentro de `PlanejadorHeterogeneoIntegrado.py` (ou um script similar), como os arquivos de entrada, a lista de missões e as posições dos robôs.
    *   Execute o script. Ele salvará o plano de missão detalhado em um arquivo JSON.

3.  **Simule a Missão**:
    *   Use o `SITL_planningsimulation.py` para carregar o JSON da missão, converter os caminhos para coordenadas GPS e enviá-los aos robôs simulados.

### 3.3. Executando o Servidor de Missão

O servidor é a forma recomendada para integrar o planejador com uma interface de usuário.

1.  **Inicie o servidor:**
    ```bash
    # Certifique-se de que o ambiente virtual está ativado
    python mission_server.py
    ```

2.  **Interaja via API REST**:
    O servidor estará rodando em `http://127.0.0.1:5001`. Você pode usar ferramentas como `curl` ou Postman para interagir com os endpoints:
    *   `POST /execute-mission`: Envia uma configuração de missão (robôs, equipamentos) e recebe de volta os waypoints planejados.
    *   `POST /stop-mission`: Para a missão em andamento.
    *   `GET /mission-status`: Verifica o status da missão atual.
    *   `GET /health`: Verifica a saúde do servidor.

3.  **Monitore via WebSocket**:
    Conecte-se ao servidor via WebSocket para receber atualizações em tempo real sobre a posição dos robôs e o status da missão.

## 4. Estruturas de Dados Chave

*   **Grafo do Ambiente (`graph.json`)**: Um dicionário JSON representando um grafo `networkx`, onde os nós possuem um atributo `pos` com coordenadas `(x, y)` e as arestas possuem `weight` (distância).
*   **Pontos de Observação (`observation_points.json`)**: Um dicionário que mapeia cada obstáculo a uma lista de pontos de observação ao seu redor, cada um com um `label` único e coordenadas GPS `[lat, lon]`.
*   **Plano de Missão (`missao.json`)**: A saída do planejador. É um dicionário que mapeia cada robô a uma lista de missões, e cada missão a uma lista de tarefas (pontos de inspeção). Cada tarefa contém o caminho detalhado em nós do grafo e em coordenadas GPS.

## 5. Conclusão Teórica

O `MultiGraphPlanner` resolve um problema complexo de planejamento para múltiplos agentes (multi-agent pathfinding and task allocation). Ele combina técnicas de:

*   **Busca Heurística (A\*)**: Para encontrar caminhos ótimos de baixo nível no grafo do ambiente.
*   **Clusterização (K-Means implícito)**: Para dividir as tarefas de forma balanceada entre os robôs.
*   **Problema do Caixeiro Viajante (TSP)**: Para otimizar a sequência de visitação das tarefas atribuídas a cada robô.
*   **Composição de Autômatos**: A estrutura do `multigraphplanner.py` também inclui métodos para composição paralela de autômatos, permitindo a modelagem de comportamentos complexos e sincronizados, embora o pipeline principal use a abordagem de clusterização e TSP.

Este sistema é aplicável a cenários de robótica autônoma, logística, manufatura e operações inteligentes onde tarefas distribuídas precisam ser otimizadas.