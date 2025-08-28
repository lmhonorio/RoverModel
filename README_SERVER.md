# Servidor de Planejamento de Missões

Este servidor Flask recebe dados da interface web de planejamento de missões e executa o algoritmo de otimização de rotas para rovers.

## 🚀 Instalação e Execução

### Método 1: Usando o script de inicialização (Recomendado)
```bash
cd RoverModel
python start_server.py
```

### Método 2: Execução manual
```bash
cd RoverModel
pip install -r requirements.txt
python mission_server.py
```

## 📡 Endpoints da API

### `GET /health`
Verifica se o servidor está funcionando.

**Resposta:**
```json
{
  "status": "healthy",
  "message": "Servidor de planejamento de missões ativo",
  "version": "1.0.0"
}
```

### `POST /execute-mission`
Executa o planejamento de missão com os dados recebidos da interface.

**Corpo da requisição:**
```json
{
  "name": "Nome da Missão",
  "description": "Descrição da missão",
  "rovers": [
    {
      "identifier": "rover_id",
      "name": "Nome do Rover",
      "id": "internal_id"
    }
  ],
  "equipments": [
    {
      "id": "eq_id",
      "name": "Nome do Equipamento",
      "type": "tipo_equipamento",
      "equipmentId": "id_equipamento",
      "lat": -3.123,
      "lng": -41.764
    }
  ],
  "substation": "substation_id"
}
```

**Resposta de sucesso:**
```json
{
  "success": true,
  "message": "Planejamento executado com sucesso",
  "data": {
    "substation": "substation_id",
    "robots_used": 2,
    "total_missions": 15,
    "routes": {
      "R1": {
        "rover_name": "Rover Alpha",
        "rover_id": "R1",
        "route": ["start", "point1", "point2", "..."],
        "total_points": 8,
        "estimated_time": 120
      }
    }
  }
}
```

### `GET /config`
Retorna as configurações atuais do servidor.

## 🔧 Configuração

As configurações padrão estão definidas em `DEFAULT_CONFIG` no arquivo `mission_server.py`:

- **graph_file**: Caminho para o arquivo JSON do grafo
- **observation_points_file**: Caminho para os pontos de observação
- **missions**: Lista de missões padrão
- **mission_execution_time**: Tempo estimado por missão (segundos)
- **robot_positions**: Posições iniciais dos robôs

## 📁 Arquivos Necessários

O servidor espera encontrar os seguintes arquivos:
- `./jsons/graph9_new.json` - Grafo do ambiente
- `./jsons/obp_6.json` - Pontos de observação por obstáculo

## 🔗 Integração com a Interface Web

A interface web deve fazer uma requisição POST para `http://localhost:5000/execute-mission` com os dados dos rovers e equipamentos selecionados.

### Exemplo de uso no JavaScript:
```javascript
const response = await fetch('http://localhost:5000/execute-mission', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    rovers: selectedRovers,
    equipments: selectedEquipments,
    substation: selectedSubstation
  })
});

const result = await response.json();
```

## 🐛 Troubleshooting

### Erro "Módulo não encontrado"
Certifique-se de que todos os módulos Python necessários estão no diretório RoverModel:
- `old/multigraphplanner.py`
- `segmentutils.py`
- `plotutils.py`
- `tspOptimization.py`
- `aabbutils.py`

### Erro "Arquivo não encontrado"
Verifique se os arquivos JSON necessários existem nos caminhos corretos:
- `./jsons/graph9_new.json`
- `./jsons/obp_6.json`

### Erro de CORS
O servidor tem CORS habilitado por padrão. Se ainda houver problemas, verifique se a URL da interface web está correta.

## 📊 Logs

O servidor imprime logs detalhados no console, incluindo:
- Dados recebidos via POST
- Processo de planejamento
- Rotas calculadas
- Erros e exceções

## ⚡ Performance

O servidor é otimizado para:
- Requisições assíncronas
- Processamento em threads separadas
- Logs detalhados para debugging
- Tratamento robusto de erros
