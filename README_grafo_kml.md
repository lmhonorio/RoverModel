# 🗺️ Conversores Graph JSON para KML

Este conjunto de scripts converte o arquivo `graph_equipment.json` em arquivos KML para visualização dos caminhos possíveis no Google Earth.

## 📁 Arquivos Criados

### Scripts de Conversão:
- `graph_to_kml.py` - Conversor completo com todos os caminhos
- `graph_to_kml_simples.py` - Conversor simplificado e otimizado

### Arquivos KML Gerados:
- `grafo_caminhos.kml` - Versão completa com todos os caminhos (3.3MB)
- `grafo_caminhos_simples.kml` - Versão simplificada (560KB)

## 🚀 Como Usar

### Conversão Completa (todos os caminhos):
```bash
cd /home/viki/RoverModel
python3 graph_to_kml.py
```

### Conversão Simplificada (amostra):
```bash
cd /home/viki/RoverModel
python3 graph_to_kml_simples.py
```

## 📊 Dados Processados

- **Total de nós:** 3,593 pontos de observação
- **Total de arestas:** 5,462 caminhos possíveis
- **Versão simplificada:** 500 nós + 1,000 arestas (amostra)

## 🎨 Visualização no Google Earth

### Elementos do KML:
- **🟢 Pontos verdes:** Pontos de observação (nós do grafo)
- **🔵 Linhas azuis:** Caminhos possíveis (arestas do grafo)

### Informações Exibidas:
- **Pontos:** ID do ponto, coordenadas originais e GPS
- **Caminhos:** Origem, destino e distância em metros

## 📋 Estrutura do KML

### Versão Completa (`grafo_caminhos.kml`):
- Todos os 3,593 pontos de observação
- Todos os 5,462 caminhos possíveis
- Organizado em pastas por tipo
- Arquivo grande (3.3MB) - pode ser lento no Google Earth

### Versão Simplificada (`grafo_caminhos_simples.kml`):
- Amostra de 500 pontos de observação
- Amostra de 1,000 caminhos possíveis
- Arquivo menor (560KB) - carregamento mais rápido
- Ideal para visualização geral e análise

## 🔧 Características Técnicas

- **Coordenadas:** GPS precisas baseadas no ArduPilot
- **Transformação:** Mesma conversão do Gazebo2CSV.py
- **Compatibilidade:** Google Earth, Google Maps, QGIS
- **Performance:** Versão simplificada otimizada para carregamento rápido

## 📍 Localização

Todos os caminhos estão localizados na região de Parnaíba III:
- **Latitude de referência:** -3.123199°
- **Longitude de referência:** -41.764537°

## 🎯 Como Visualizar

1. **Abra o Google Earth**
2. **File > Open**
3. **Selecione um dos arquivos KML:**
   - `grafo_caminhos_simples.kml` (recomendado para início)
   - `grafo_caminhos.kml` (completo, pode ser lento)
4. **Explore os caminhos organizados em pastas**
5. **Use as informações detalhadas para análise de rotas**

## 📈 Análise dos Caminhos

### O que você pode ver:
- **Rede de caminhos:** Como os pontos se conectam
- **Distâncias:** Comprimento de cada segmento
- **Conectividade:** Quais pontos são mais conectados
- **Rotas possíveis:** Caminhos alternativos entre pontos

### Para Planejamento:
- **Pontos estratégicos:** Nós com muitas conexões
- **Caminhos críticos:** Arestas importantes para conectividade
- **Áreas isoladas:** Pontos com poucas conexões
- **Rotas otimizadas:** Caminhos mais curtos entre pontos

## ⚠️ Notas Importantes

- **Performance:** Use a versão simplificada primeiro
- **Navegação:** Os caminhos são organizados em pastas no Google Earth
- **Coordenadas:** Baseadas no sistema de coordenadas corrigido do Gazebo
- **Escala:** As distâncias são em metros reais

## 🎊 Resultado

Os arquivos KML permitem visualizar toda a rede de caminhos possíveis do rover na região de Parnaíba III, facilitando o planejamento de missões e análise de conectividade entre pontos de observação!
