#!/usr/bin/env python3
"""
Conversor Graph JSON para KML Simplificado
Versão otimizada que mostra apenas uma amostra dos caminhos para melhor performance
"""

import json
import os
import math
from typing import Dict, List, Tuple
from xml.dom.minidom import Document

class SimpleGraphToKMLConverter:
    def __init__(self, json_file: str = "../../jsons/graph_equipment.json"):
        """Inicializa o conversor simplificado"""
        self.json_file = json_file
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(self.project_dir, json_file)
        
        print("🚀 Conversor Graph JSON para KML Simplificado")
        print(f"📁 Diretório: {self.project_dir}")
    
    def load_graph_data(self) -> Dict:
        """Carrega os dados do arquivo JSON"""
        try:
            if not os.path.exists(self.json_path):
                print(f"❌ Arquivo não encontrado: {self.json_path}")
                return {}
            
            with open(self.json_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            print(f"✅ JSON carregado:")
            print(f"   📊 Nós: {len(graph_data.get('nodes', {}))}")
            print(f"   🔗 Arestas: {len(graph_data.get('edges', []))}")
            return graph_data
            
        except Exception as e:
            print(f"❌ Erro ao carregar JSON: {e}")
            return {}
    
    def convert_coordinates_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """Converte coordenadas do grafo para GPS"""
        lat_ref = -3.123199
        lon_ref = -41.764537
        radius_of_earth = 6378100.0
        
        # SEM ROTAÇÃO: Usa as coordenadas diretamente, consistente com Gazebo2CSV.py
        # As coordenadas já estão na orientação correta
        
        # Converte metros para graus usando constantes do ArduPilot
        # Latitude: variação Norte-Sul (Y direto)
        lat = y / (radius_of_earth * math.pi / 180.0) + lat_ref
        # Longitude: variação Leste-Oeste (X direto)
        lon = x / (radius_of_earth * math.cos(math.radians(lat_ref)) * math.pi / 180.0) + lon_ref
        
        return lat, lon
    
    def create_simple_kml(self, graph_data: Dict) -> str:
        """Cria KML simplificado"""
        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])
        
        # Limita o número de elementos para melhor performance
        max_nodes = 500
        max_edges = 1000
        
        # Seleciona uma amostra dos nós
        node_items = list(nodes.items())[:max_nodes]
        
        # Seleciona uma amostra das arestas
        sample_edges = edges[:max_edges]
        
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Grafo de Caminhos - Parnaíba III (Amostra)</name>
    <description>Caminhos possíveis entre pontos de observação (amostra para melhor performance)</description>
    
    <!-- Estilos -->
    <Style id="node_style">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>0.6</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
    
    <Style id="edge_style">
      <LineStyle>
        <color>ff0000ff</color>
        <width>1</width>
      </LineStyle>
    </Style>
    
    <!-- Pontos de Observação -->
    <Folder>
      <name>Pontos de Observação (Amostra)</name>
'''
        
        # Adiciona nós
        print(f"📍 Adicionando {len(node_items)} nós ao KML...")
        for node_id, node_data in node_items:
            pos = node_data.get('pos', [])
            if len(pos) >= 2:
                x, y = pos[0], pos[1]
                lat, lon = self.convert_coordinates_to_gps(x, y)
                
                kml_content += f'''      <Placemark>
        <name>{node_id}</name>
        <description>ID: {node_id}
Coordenadas: ({x:.2f}, {y:.2f})
GPS: ({lat:.8f}, {lon:.8f})</description>
        <styleUrl>#node_style</styleUrl>
        <Point>
          <coordinates>{lon},{lat},0</coordinates>
        </Point>
      </Placemark>
'''
        
        kml_content += '''    </Folder>
    
    <!-- Caminhos Possíveis -->
    <Folder>
      <name>Caminhos Possíveis (Amostra)</name>
'''
        
        # Adiciona arestas
        print(f"🔗 Adicionando {len(sample_edges)} arestas ao KML...")
        for edge in sample_edges:
            if len(edge) >= 3:
                node1_id, node2_id, distance = edge[0], edge[1], edge[2]
                
                if node1_id in nodes and node2_id in nodes:
                    pos1 = nodes[node1_id].get('pos', [])
                    pos2 = nodes[node2_id].get('pos', [])
                    
                    if len(pos1) >= 2 and len(pos2) >= 2:
                        lat1, lon1 = self.convert_coordinates_to_gps(pos1[0], pos1[1])
                        lat2, lon2 = self.convert_coordinates_to_gps(pos2[0], pos2[1])
                        
                        kml_content += f'''      <Placemark>
        <name>{node1_id} → {node2_id}</name>
        <description>De: {node1_id}
Para: {node2_id}
Distância: {distance:.2f}m</description>
        <styleUrl>#edge_style</styleUrl>
        <LineString>
          <coordinates>{lon1},{lat1},0 {lon2},{lat2},0</coordinates>
        </LineString>
      </Placemark>
'''
        
        kml_content += '''    </Folder>
  </Document>
</kml>'''
        
        return kml_content
    
    def save_kml(self, kml_content: str, output_file: str = "grafo_caminhos_simples.kml"):
        """Salva o KML em arquivo"""
        try:
            output_path = os.path.join(self.project_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            
            print(f"💾 KML salvo: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar KML: {e}")
            return None
    
    def convert(self, output_file: str = "grafo_caminhos_simples.kml"):
        """Executa a conversão"""
        print("\n🔄 Convertendo Graph JSON para KML...")
        
        # Carrega dados
        graph_data = self.load_graph_data()
        if not graph_data:
            return None
        
        # Cria KML
        print("📝 Criando KML...")
        kml_content = self.create_simple_kml(graph_data)
        
        # Salva arquivo
        print("💾 Salvando arquivo...")
        output_path = self.save_kml(kml_content, output_file)
        
        if output_path:
            print(f"\n✅ Conversão concluída!")
            print(f"📄 Arquivo: {output_path}")
            
            nodes_count = len(graph_data.get('nodes', {}))
            edges_count = len(graph_data.get('edges', []))
            
            print(f"📊 Estatísticas:")
            print(f"   📍 Total de nós: {nodes_count}")
            print(f"   🔗 Total de arestas: {edges_count}")
            print(f"   📍 Nós incluídos: 500 (amostra)")
            print(f"   🔗 Arestas incluídas: 1000 (amostra)")
            
            return output_path
        else:
            print("❌ Falha na conversão!")
            return None

def main():
    """Função principal"""
    print("🚀 CONVERSOR GRAPH JSON PARA KML SIMPLIFICADO")
    print("=" * 50)
    
    # Cria conversor
    converter = SimpleGraphToKMLConverter("../../jsons/graph_equipment.json")
    
    # Executa conversão
    output_file = converter.convert("grafo_caminhos_simples.kml")
    
    if output_file:
        print(f"\n🎯 Para visualizar:")
        print(f"   1. Abra o Google Earth")
        print(f"   2. File > Open")
        print(f"   3. Selecione: {output_file}")
        print(f"   4. Explore os caminhos possíveis!")
        print(f"\n📋 Elementos:")
        print(f"   🟢 Pontos verdes: Pontos de observação (amostra)")
        print(f"   🔵 Linhas azuis: Caminhos possíveis (amostra)")
        print(f"\n💡 Nota: Esta versão mostra apenas uma amostra para melhor performance.")
        print(f"   Para ver todos os caminhos, use o arquivo completo: grafo_caminhos.kml")
    else:
        print("\n❌ Conversão falhou!")

if __name__ == "__main__":
    main()
