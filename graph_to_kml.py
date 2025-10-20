#!/usr/bin/env python3
"""
Conversor Graph JSON para KML
Converte o arquivo graph_equipment.json em um arquivo KML para visualização dos caminhos no Google Earth
"""

import json
import os
import math
from typing import Dict, List, Tuple
from xml.dom.minidom import Document

class GraphToKMLConverter:
    def __init__(self, json_file: str = "jsons/graph_equipment.json"):
        """Inicializa o conversor de grafo para KML"""
        self.json_file = json_file
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.json_path = os.path.join(self.project_dir, json_file)
        
        print("🚀 Inicializando conversor Graph JSON para KML...")
        print(f"📁 Diretório: {self.project_dir}")
        print(f"📄 Arquivo JSON: {self.json_path}")
    
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
        """Converte coordenadas do grafo para GPS usando a mesma transformação do Gazebo2CSV.py"""
        # Coordenadas de referência do Gazebo2CSV.py
        lat_ref = -3.123199
        lon_ref = -41.764537
        radius_of_earth = 6378100.0  # metros
        
        # Aplica a mesma transformação de rotação do Gazebo2CSV.py
        x_rotated = y  # X original vira Y após rotação de -90°
        y_rotated = -x  # Y original vira -X após rotação de -90°
        
        # Converte metros para graus usando constantes do ArduPilot
        lat = y_rotated / (radius_of_earth * math.pi / 180.0) + lat_ref
        lon = x_rotated / (radius_of_earth * math.cos(math.radians(lat_ref)) * math.pi / 180.0) + lon_ref
        
        return lat, lon
    
    def create_kml_document(self, graph_data: Dict) -> Document:
        """Cria o documento KML principal"""
        doc = Document()
        kml = doc.createElement("kml")
        kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
        doc.appendChild(kml)
        
        document = doc.createElement("Document")
        kml.appendChild(document)
        
        # Propriedades do documento
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode("Grafo de Caminhos - Parnaíba III"))
        document.appendChild(name)
        
        description = doc.createElement("description")
        description.appendChild(doc.createTextNode("Caminhos possíveis entre pontos de observação extraídos do grafo"))
        document.appendChild(description)
        
        # Estilos
        self.create_styles(document)
        
        # Pastas para organizar
        folders = self.create_folders(document)
        
        # Adiciona nós e arestas
        self.add_nodes_and_edges(graph_data, folders)
        
        return doc
    
    def create_styles(self, document):
        """Cria estilos para diferentes elementos"""
        doc = document.ownerDocument
        
        # Estilo para nós (pontos de observação)
        node_style = doc.createElement("Style")
        node_style.setAttribute("id", "node_style")
        icon_style = doc.createElement("IconStyle")
        color = doc.createElement("color")
        color.appendChild(doc.createTextNode("ff00ff00"))  # Verde
        scale = doc.createElement("scale")
        scale.appendChild(doc.createTextNode("0.8"))
        icon = doc.createElement("Icon")
        href = doc.createElement("href")
        href.appendChild(doc.createTextNode("http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"))
        icon.appendChild(href)
        icon_style.appendChild(color)
        icon_style.appendChild(scale)
        icon_style.appendChild(icon)
        node_style.appendChild(icon_style)
        document.appendChild(node_style)
        
        # Estilo para arestas (caminhos)
        edge_style = doc.createElement("Style")
        edge_style.setAttribute("id", "edge_style")
        line_style = doc.createElement("LineStyle")
        color = doc.createElement("color")
        color.appendChild(doc.createTextNode("ff0000ff"))  # Azul
        width = doc.createElement("width")
        width.appendChild(doc.createTextNode("2"))
        line_style.appendChild(color)
        line_style.appendChild(width)
        edge_style.appendChild(line_style)
        document.appendChild(edge_style)
    
    def create_folders(self, document):
        """Cria pastas para organizar os elementos"""
        doc = document.ownerDocument
        folders = {}
        
        # Pasta para nós
        nodes_folder = doc.createElement("Folder")
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode("Pontos de Observação"))
        nodes_folder.appendChild(name)
        document.appendChild(nodes_folder)
        folders['nodes'] = nodes_folder
        
        # Pasta para arestas
        edges_folder = doc.createElement("Folder")
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode("Caminhos Possíveis"))
        edges_folder.appendChild(name)
        document.appendChild(edges_folder)
        folders['edges'] = edges_folder
        
        return folders
    
    def add_nodes_and_edges(self, graph_data: Dict, folders: Dict):
        """Adiciona nós e arestas ao KML"""
        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])
        
        # Adiciona nós
        print("📍 Adicionando nós ao KML...")
        for node_id, node_data in nodes.items():
            pos = node_data.get('pos', [])
            if len(pos) >= 2:
                x, y = pos[0], pos[1]
                lat, lon = self.convert_coordinates_to_gps(x, y)
                
                # Cria Placemark para o nó
                placemark = folders['nodes'].ownerDocument.createElement("Placemark")
                
                name = folders['nodes'].ownerDocument.createElement("name")
                name.appendChild(folders['nodes'].ownerDocument.createTextNode(node_id))
                placemark.appendChild(name)
                
                description = folders['nodes'].ownerDocument.createElement("description")
                desc_text = f"ID: {node_id}\nCoordenadas originais: ({x:.2f}, {y:.2f})\nGPS: ({lat:.8f}, {lon:.8f})"
                description.appendChild(folders['nodes'].ownerDocument.createTextNode(desc_text))
                placemark.appendChild(description)
                
                style_url = folders['nodes'].ownerDocument.createElement("styleUrl")
                style_url.appendChild(folders['nodes'].ownerDocument.createTextNode("#node_style"))
                placemark.appendChild(style_url)
                
                point = folders['nodes'].ownerDocument.createElement("Point")
                coordinates = folders['nodes'].ownerDocument.createElement("coordinates")
                coordinates.appendChild(folders['nodes'].ownerDocument.createTextNode(f"{lon},{lat},0"))
                point.appendChild(coordinates)
                placemark.appendChild(point)
                
                folders['nodes'].appendChild(placemark)
        
        # Adiciona arestas
        print("🔗 Adicionando arestas ao KML...")
        for edge in edges:
            if len(edge) >= 3:
                node1_id, node2_id, distance = edge[0], edge[1], edge[2]
                
                if node1_id in nodes and node2_id in nodes:
                    pos1 = nodes[node1_id].get('pos', [])
                    pos2 = nodes[node2_id].get('pos', [])
                    
                    if len(pos1) >= 2 and len(pos2) >= 2:
                        # Converte coordenadas para GPS
                        lat1, lon1 = self.convert_coordinates_to_gps(pos1[0], pos1[1])
                        lat2, lon2 = self.convert_coordinates_to_gps(pos2[0], pos2[1])
                        
                        # Cria Placemark para a aresta
                        placemark = folders['edges'].ownerDocument.createElement("Placemark")
                        
                        name = folders['edges'].ownerDocument.createElement("name")
                        name.appendChild(folders['edges'].ownerDocument.createTextNode(f"{node1_id} → {node2_id}"))
                        placemark.appendChild(name)
                        
                        description = folders['edges'].ownerDocument.createElement("description")
                        desc_text = f"De: {node1_id}\nPara: {node2_id}\nDistância: {distance:.2f}m"
                        description.appendChild(folders['edges'].ownerDocument.createTextNode(desc_text))
                        placemark.appendChild(description)
                        
                        style_url = folders['edges'].ownerDocument.createElement("styleUrl")
                        style_url.appendChild(folders['edges'].ownerDocument.createTextNode("#edge_style"))
                        placemark.appendChild(style_url)
                        
                        # Cria linha
                        line_string = folders['edges'].ownerDocument.createElement("LineString")
                        coordinates = folders['edges'].ownerDocument.createElement("coordinates")
                        coord_text = f"{lon1},{lat1},0 {lon2},{lat2},0"
                        coordinates.appendChild(folders['edges'].ownerDocument.createTextNode(coord_text))
                        line_string.appendChild(coordinates)
                        placemark.appendChild(line_string)
                        
                        folders['edges'].appendChild(placemark)
    
    def save_kml(self, doc: Document, output_file: str = "grafo_caminhos.kml"):
        """Salva o KML em arquivo"""
        try:
            # Cria string XML formatada
            pretty_xml = doc.toprettyxml(indent="  ")
            
            # Remove linhas vazias extras
            lines = [line for line in pretty_xml.split('\n') if line.strip()]
            formatted_xml = '\n'.join(lines)
            
            # Salva arquivo
            output_path = os.path.join(self.project_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_xml)
            
            print(f"💾 KML salvo: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar KML: {e}")
            return None
    
    def convert(self, output_file: str = "grafo_caminhos.kml"):
        """Executa a conversão completa"""
        print("\n🔄 Iniciando conversão Graph JSON para KML...")
        
        # Carrega dados
        graph_data = self.load_graph_data()
        if not graph_data:
            return None
        
        # Cria KML
        print("📝 Criando documento KML...")
        doc = self.create_kml_document(graph_data)
        
        # Salva arquivo
        print("💾 Salvando arquivo KML...")
        output_path = self.save_kml(doc, output_file)
        
        if output_path:
            print(f"\n✅ Conversão concluída com sucesso!")
            print(f"📄 Arquivo KML: {output_path}")
            
            nodes_count = len(graph_data.get('nodes', {}))
            edges_count = len(graph_data.get('edges', []))
            
            print(f"📊 Estatísticas:")
            print(f"   📍 Pontos de observação: {nodes_count}")
            print(f"   🔗 Caminhos possíveis: {edges_count}")
            
            return output_path
        else:
            print("❌ Falha na conversão!")
            return None

def main():
    """Função principal"""
    print("🚀 CONVERSOR GRAPH JSON PARA KML")
    print("=" * 50)
    
    # Cria conversor
    converter = GraphToKMLConverter("jsons/graph_equipment.json")
    
    # Executa conversão
    output_file = converter.convert("grafo_caminhos.kml")
    
    if output_file:
        print(f"\n🎯 Para visualizar:")
        print(f"   1. Abra o Google Earth")
        print(f"   2. File > Open")
        print(f"   3. Selecione: {output_file}")
        print(f"   4. Explore os caminhos possíveis!")
        print(f"\n📋 Elementos:")
        print(f"   🟢 Pontos verdes: Pontos de observação")
        print(f"   🔵 Linhas azuis: Caminhos possíveis")
    else:
        print("\n❌ Conversão falhou!")

if __name__ == "__main__":
    main()
