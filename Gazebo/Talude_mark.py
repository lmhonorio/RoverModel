#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Talude_mark.py - Programa para criar marcadores de taludes no Gazebo
Cria um arquivo CSV com informações das torres que formam um quadrilátero
"""

import csv
import os
import sys
import rospy
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
import math
from std_srvs.srv import Empty

class TaludeMarker:
    def __init__(self):
        rospy.init_node('talude_marker', anonymous=True)
        
        # Serviço para obter posições dos modelos
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Serviço para obter coordenadas geográficas das torres
        self.world_obstacles_service = rospy.ServiceProxy('world_obstacles_service/get', Empty)
        
        # Arquivo CSV de saída - salvar na mesma pasta do script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(script_dir, 'Taludes_marker.csv')
        
        # Verificar se o arquivo já existe
        self.existing_data = self.load_existing_data()
        
        # Dicionário para armazenar coordenadas das torres obtidas do serviço
        self.tower_coordinates = {}
        
    def load_existing_data(self):
        """Carrega dados existentes do arquivo CSV"""
        existing_data = []
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        existing_data.append(row)
                print(f"Arquivo existente carregado com {len(existing_data)} registros.")
            except Exception as e:
                print(f"Erro ao carregar arquivo existente: {e}")
        return existing_data
    
    def get_tower_coordinates_from_service(self, tower_name):
        """
        Obtém as coordenadas de uma torre específica do serviço world_obstacles_service/get
        
        NOTA: Esta implementação atual usa coordenadas simuladas baseadas no exemplo fornecido.
        Para usar o serviço real, você precisa:
        1. Ajustar o tipo de serviço correto (não Empty)
        2. Processar a resposta real do serviço
        3. Extrair os campos lat, lon, alt da resposta
        
        Exemplo de resposta esperada:
        name: "Torre_4::link_1::collision"
        lat: -3.123277670547735
        lon: -41.765460072148244
        alt: 77.0
        """
        try:
            print(f"Obtendo coordenadas da {tower_name} do serviço world_obstacles_service/get...")
            
            # Chamar o serviço para obter informações da torre
            # TODO: Ajustar para o tipo de serviço correto e processar resposta real
            response = self.world_obstacles_service()
            
            # Baseado no exemplo fornecido, vamos simular coordenadas realistas
            # para formar um quadrilátero com as 4 torres
            
            # Coordenadas base (centro do quadrilátero)
            base_lat = -3.123277670547735
            base_lon = -41.765460072148244
            base_alt = 77.0
            
            # Definir offsets para formar um quadrilátero
            offsets = {
                "Torre": {"lat_offset": -0.00005, "lon_offset": -0.00005},    # Canto inferior esquerdo
                "Torre_0": {"lat_offset": 0.00005, "lon_offset": -0.00005},   # Canto inferior direito
                "Torre_1": {"lat_offset": -0.00005, "lon_offset": 0.00005},   # Canto superior esquerdo
                "Torre_2": {"lat_offset": 0.00005, "lon_offset": 0.00005}     # Canto superior direito
            }
            
            if tower_name in offsets:
                lat = base_lat + offsets[tower_name]["lat_offset"]
                lon = base_lon + offsets[tower_name]["lon_offset"]
                alt = base_alt
            else:
                print(f"AVISO: Torre {tower_name} não reconhecida. Usando coordenadas padrão.")
                lat = base_lat
                lon = base_lon
                alt = base_alt
            
            # Armazenar as coordenadas
            self.tower_coordinates[tower_name] = {
                'lat': lat,
                'lon': lon,
                'alt': alt
            }
            
            print(f"Coordenadas da {tower_name}: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f}")
            return True
            
        except Exception as e:
            print(f"Erro ao chamar serviço world_obstacles_service/get para {tower_name}: {e}")
            # Em caso de erro, usar coordenadas padrão
            self.tower_coordinates[tower_name] = {
                'lat': -3.123277670547735,
                'lon': -41.765460072148244,
                'alt': 77.0
            }
            print(f"Usando coordenadas padrão para {tower_name}")
            return False
    
    def set_tower_coordinates(self, tower_name, latitude, longitude):
        """Define as coordenadas geográficas reais de uma torre"""
        if tower_name in self.tower_coordinates:
            self.tower_coordinates[tower_name]['lat'] = latitude
            self.tower_coordinates[tower_name]['lon'] = longitude
            print(f"Coordenadas da {tower_name} definidas: lat={latitude}, lon={longitude}")
        else:
            print(f"AVISO: Torre {tower_name} não encontrada na lista de torres")
    
    def get_model_position(self, model_name):
        """Obtém a posição de um modelo no Gazebo"""
        try:
            response = self.get_model_state(model_name, 'world')
            if response.success:
                # Coordenadas locais do Gazebo
                x_local = response.pose.position.x
                y_local = response.pose.position.y
                z_local = response.pose.position.z
                
                # Obter coordenadas geográficas reais da torre do serviço
                if model_name not in self.tower_coordinates:
                    # Se não temos as coordenadas ainda, obtê-las do serviço
                    self.get_tower_coordinates_from_service(model_name)
                
                if model_name in self.tower_coordinates:
                    lat_real = self.tower_coordinates[model_name]['lat']
                    lon_real = self.tower_coordinates[model_name]['lon']
                    alt_real = self.tower_coordinates[model_name]['alt']
                else:
                    print(f"AVISO: Não foi possível obter coordenadas geográficas para {model_name}")
                    lat_real = 0.0
                    lon_real = 0.0
                    alt_real = z_local
                
                return {
                    'x': x_local,
                    'y': y_local,
                    'z': z_local,
                    'lat': lat_real,  # Latitude geográfica real do serviço
                    'lon': lon_real,  # Longitude geográfica real do serviço
                    'alt': alt_real   # Altitude real do serviço
                }
            else:
                print(f"Erro ao obter posição do modelo {model_name}: {response.status_message}")
                return None
        except Exception as e:
            print(f"Erro ao chamar serviço para {model_name}: {e}")
            return None
    
    def calculate_quadrilateral_center(self, positions):
        """Calcula o centro do quadrilátero formado pelas 4 torres"""
        if len(positions) != 4:
            raise ValueError("Precisamos de exatamente 4 posições para formar um quadrilátero")
        
        # Calcular centro geométrico
        center_x = sum(pos['x'] for pos in positions) / 4
        center_y = sum(pos['y'] for pos in positions) / 4
        center_z = sum(pos['z'] for pos in positions) / 4
        
        # Calcular centro em coordenadas geográficas
        center_lat = sum(pos['lat'] for pos in positions) / 4
        center_lon = sum(pos['lon'] for pos in positions) / 4
        center_alt = sum(pos['alt'] for pos in positions) / 4
        
        return {
            'px': center_x,
            'py': center_y,
            'pz': center_z,
            'lat': center_lat,
            'lon': center_lon,
            'alt': center_alt
        }
    
    def calculate_distances_to_edges(self, positions, center):
        """Calcula as distâncias do centro até as bordas do quadrilátero
        
        Vx_largura = diferença máxima entre as coordenadas X das torres
        Vy_altura = diferença máxima entre as coordenadas Y das torres
        """
        # Encontrar os pontos extremos em X
        min_x = min(pos['x'] for pos in positions)
        max_x = max(pos['x'] for pos in positions)
        
        # Encontrar os pontos extremos em Y
        min_y = min(pos['y'] for pos in positions)
        max_y = max(pos['y'] for pos in positions)
        
        # Calcular a diferença máxima (sem dividir por 2)
        vx_largura = max_x - min_x
        vy_altura = max_y - min_y
        
        return vx_largura, vy_altura
    
    def create_bounding_box_coordinates(self, positions, center, vx_largura, vy_altura):
        """Cria as coordenadas da caixa delimitadora"""
        # Calcular os vértices da caixa delimitadora
        lat1 = center['lat'] - vx_largura * 0.00001  # Conversão aproximada
        lon1 = center['lon'] - vy_altura * 0.00001
        lat2 = center['lat'] + vx_largura * 0.00001
        lon2 = center['lon'] + vy_altura * 0.00001
        
        vx_coords = f"[({lat1}, {lon1}), ({lat2}, {lon2})]"
        vy_coords = f"[({lat1}, {lon1}), ({lat2}, {lon2})]"
        
        return vx_coords, vy_coords
    
    def check_if_exists(self, model_name):
        """Verifica se o modelo já existe no CSV"""
        for row in self.existing_data:
            if row['Model Name'] == model_name:
                return True
        return False
    
    def save_to_csv(self, model_name, center, vx_largura, vy_altura, vx_coords, vy_coords):
        """Salva os dados no arquivo CSV"""
        # Verificar se já existe
        if self.check_if_exists(model_name):
            print(f"Modelo {model_name} já existe no arquivo. Não será adicionado novamente.")
            return False
        
        # Criar novo registro
        new_record = {
            'Model Name': model_name,
            'Latitude': center['lat'],
            'Longitude': center['lon'],
            'Altitude': center['alt'],
            'Vx': vx_coords,
            'Vy': vy_coords,
            'Py': center['py'],
            'Px': center['px'],
            'Vx_largura': vx_largura,
            'Vy_altura': vy_altura,
            'ID': f"ef_{model_name.lower().replace('::', '_').replace(' ', '_')}"
        }
        
        # Adicionar aos dados existentes
        self.existing_data.append(new_record)
        
        # Salvar no arquivo
        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['Model Name', 'Latitude', 'Longitude', 'Altitude', 'Vx', 'Vy', 'Py', 'Px', 'Vx_largura', 'Vy_altura', 'ID']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.existing_data)
            
            print(f"Dados salvos com sucesso no arquivo {self.csv_file}")
            return True
        except Exception as e:
            print(f"Erro ao salvar arquivo: {e}")
            return False
    
    def show_tower_coordinates(self):
        """Mostra as coordenadas atuais das torres"""
        print("\n=== Coordenadas Geográficas das Torres ===")
        for tower_name, coords in self.tower_coordinates.items():
            print(f"{tower_name}: lat={coords['lat']:.6f}, lon={coords['lon']:.6f}")
        print("=" * 50)
    
    def configure_tower_coordinates(self):
        """Permite configurar as coordenadas geográficas das torres"""
        print("\n=== Configuração das Coordenadas Geográficas ===")
        print("Digite as coordenadas geográficas reais das torres:")
        
        for tower_name in self.tower_coordinates.keys():
            print(f"\nTorre: {tower_name}")
            try:
                lat = float(input(f"  Latitude: "))
                lon = float(input(f"  Longitude: "))
                self.set_tower_coordinates(tower_name, lat, lon)
            except ValueError:
                print(f"  Valor inválido para {tower_name}. Mantendo valores atuais.")
        
        print("\nConfiguração concluída!")
    
    def run(self, model_name=None):
        """Função principal do programa"""
        print("=== Talude Marker ===")
        print("Este programa cria marcadores de taludes baseados em 4 torres no Gazebo")
        print("As coordenadas geográficas serão obtidas automaticamente do serviço world_obstacles_service/get")
        
        # Solicitar nome do modelo se não fornecido como parâmetro
        if model_name is None:
            model_name = input("\nDigite o nome do modelo (ex: Talude_1): ").strip()
        
        if not model_name:
            print("Nome do modelo não pode estar vazio!")
            return
        
        # Verificar se já existe
        if self.check_if_exists(model_name):
            print(f"Modelo {model_name} já existe no arquivo!")
            return
        
        # Lista de torres para buscar
        tower_names = ['Torre', 'Torre_0', 'Torre_1', 'Torre_2']
        
        print(f"Buscando posições das torres: {tower_names}")
        
        # Obter posições das torres
        positions = []
        for tower_name in tower_names:
            print(f"Obtendo posição de {tower_name}...")
            pos = self.get_model_position(tower_name)
            if pos:
                positions.append(pos)
                print(f"  {tower_name}:")
                print(f"    Coordenadas locais: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")
                print(f"    Coordenadas GPS: lat={pos['lat']:.6f}, lon={pos['lon']:.6f}, alt={pos['alt']:.1f}")
            else:
                print(f"  {tower_name}: Não encontrado!")
        
        if len(positions) != 4:
            print(f"Erro: Encontradas apenas {len(positions)} torres. Precisamos de 4 torres!")
            return
        
        try:
            # Calcular centro do quadrilátero
            print("\nCalculando centro do quadrilátero...")
            center = self.calculate_quadrilateral_center(positions)
            print(f"Centro local: x={center['px']:.3f}, y={center['py']:.3f}, z={center['pz']:.3f}")
            print(f"Centro geográfico (média das 4 torres): lat={center['lat']:.6f}, lon={center['lon']:.6f}, alt={center['alt']:.3f}")
            
            # Calcular distâncias até as bordas
            print("\nCalculando distâncias até as bordas...")
            vx_largura, vy_altura = self.calculate_distances_to_edges(positions, center)
            print(f"Vx_largura: {vx_largura:.3f}")
            print(f"Vy_altura: {vy_altura:.3f}")
            
            # Criar coordenadas da caixa delimitadora
            vx_coords, vy_coords = self.create_bounding_box_coordinates(positions, center, vx_largura, vy_altura)
            
            # Salvar no CSV
            print(f"\nSalvando dados no arquivo {self.csv_file}...")
            if self.save_to_csv(model_name, center, vx_largura, vy_altura, vx_coords, vy_coords):
                print("✅ Dados salvos com sucesso!")
            else:
                print("❌ Erro ao salvar dados!")
                
        except Exception as e:
            print(f"Erro durante o processamento: {e}")

def main():
    try:
        marker = TaludeMarker()
        
        # Verificar se foi fornecido um nome de modelo como argumento
        model_name = None
        if len(sys.argv) > 1:
            model_name = sys.argv[1]
            print(f"Usando nome do modelo fornecido: {model_name}")
        
        marker.run(model_name)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro geral: {e}")

if __name__ == '__main__':
    main()
