# 🗺️ Conversores CSV para KML

Este conjunto de scripts converte o arquivo `todos_pontos_gps.csv` em arquivos KML para visualização no Google Earth.

## 📁 Arquivos Criados

### Scripts de Conversão:
- `csv_to_kml.py` - Conversor completo com pastas organizadas
- `csv_to_kml_simples.py` - Conversor simplificado e otimizado

### Arquivos KML Gerados:
- `objetos_gazebo.kml` - Versão completa com pastas organizadas (255KB)
- `objetos_gazebo_simples.kml` - Versão simplificada (125KB)

## 🚀 Como Usar

### Conversão Completa (com pastas):
```bash
cd /home/viki/RoverModel/Gazebo
python3 csv_to_kml.py
```

### Conversão Simplificada:
```bash
cd /home/viki/RoverModel/Gazebo
python3 csv_to_kml_simples.py
```

## 📊 Dados Processados

- **Total de objetos:** 444
- **🔴 Reatores:** 19 objetos
- **🔵 TPC (Transformadores):** 9 objetos  
- **🟢 Outros equipamentos:** 416 objetos

## 🎨 Visualização no Google Earth

### Cores dos Objetos:
- **🔴 Vermelho:** Reatores nucleares
- **🔵 Azul:** TPC (Transformadores de Potência)
- **🟢 Verde:** Outros equipamentos

### Informações Exibidas:
- Nome do objeto
- ID único
- Dimensões (largura x altura)
- Coordenadas GPS precisas

## 📋 Estrutura do KML

### Versão Completa (`objetos_gazebo.kml`):
- Organizada em pastas por tipo de objeto
- Estilos personalizados para cada categoria
- Polígonos para objetos com dimensões definidas
- Metadados detalhados

### Versão Simplificada (`objetos_gazebo_simples.kml`):
- Estrutura mais simples e rápida
- Apenas pontos com cores diferenciadas
- Arquivo menor para carregamento mais rápido
- Ideal para visualização geral

## 🔧 Características Técnicas

- **Coordenadas:** GPS precisas baseadas no ArduPilot
- **Altitude:** Altura real dos objetos no mundo Gazebo
- **Dimensões:** Largura e altura extraídas do CSV
- **Compatibilidade:** Google Earth, Google Maps, QGIS

## 📍 Localização

Todos os objetos estão localizados na região de Parnaíba III:
- **Latitude de referência:** -3.123199°
- **Longitude de referência:** -41.764537°

## 🎯 Próximos Passos

1. **Abra o Google Earth**
2. **File > Open**
3. **Selecione um dos arquivos KML**
4. **Explore os objetos organizados por tipo**
5. **Use as informações detalhadas para análise**

## ⚠️ Notas Importantes

- Os arquivos KML são salvos automaticamente na pasta `Gazebo/`
- As coordenadas GPS foram corrigidas para considerar a rotação do mundo
- Os objetos são organizados por tipo para facilitar a navegação
- Ambos os formatos são compatíveis com ferramentas GIS padrão
