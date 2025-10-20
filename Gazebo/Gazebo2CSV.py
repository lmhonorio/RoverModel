#!/usr/bin/env python3
"""
Script para extrair posições de todos os objetos do mundo Gazebo via tópicos ROS
e converter para coordenadas GPS usando o plugin de georeferência
Formato de saída igual ao todos_pontos_gps.csv
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, TransformStamped
from gazebo_msgs.msg import ModelStates, LinkStates
from gazebo_msgs.srv import GetModelState, GetLinkState
from std_msgs.msg import String
import pandas as pd
import math
import time
import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional

class GazeboObjectExtractor:
    def __init__(self, world_type="completo"):
        """Inicializa o extrator de objetos do Gazebo"""
        rospy.init_node('gazebo_object_extractor', anonymous=True)
        
        # Tipo de mundo selecionado
        self.world_type = world_type
        
        # Coordenadas de referência do mundo Gazebo (serão obtidas do tópico geo_coordinates)
        self.lat_ref = None
        self.lon_ref = None
        self.geo_coordinates_received = False
        
        # Buffer para transformações TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Dicionário para armazenar posições dos objetos
        self.objects_data = {}
        
        # Lista de objetos para monitorar
        self.objects_to_monitor = []
        
        # Dimensões dos objetos (baseado no treat_spreadsheet2.py)
        self.DIMENSIONS = {
            "reator": (2*1.534546, 2*3.054527),
            "pr": (2*0.6871033, 2*0.6181564),
            "tpc": (2*0.864502, 2*0.7777786),
            "ip": (2*0.7239075, 2*0.5235214),
            "sech": (2*3.303741, 2*0.5302429),
            "tc": (2*0.8567124, 2*0.7042313),
            "secv": (2*1.088993, 2*0.54982),
            "disjuntor": (2*2.458675, 2*0.7382889),
            "buscsb": (2*0.7805481, 2*0.54982),
            "busip": (2*0.7805519, 2*0.54982),
            "bombeiro": (1, 1),
            "caixa": (1,1),
            "estrutura": (5.48, 2.6),
            "torre": (5.48, 2.6),
            "transformador": (2*1.534546, 2*3.054527),
            "obstaculo": (1,1),
            "cercado": (11.41, 6.8),
        }
        
        print("🚀 Inicializando extrator de objetos do Gazebo...")
        print(f"🌍 Tipo de mundo: {self.world_type}")
        print("🌍 Aguardando coordenadas de referência do tópico geo_coordinates...")
        
        # Subscreve ao tópico de coordenadas geográficas
        rospy.Subscriber('/gazebo/default/geo_coordinates', String, self.geo_coordinates_callback)
    
    def geo_coordinates_callback(self, msg: String):
        """Callback para receber coordenadas geográficas do Gazebo"""
        try:
            # O tópico geo_coordinates geralmente envia dados em formato JSON ou string
            data = msg.data
            
            # Tenta parsear como JSON primeiro
            try:
                geo_data = json.loads(data)
                if 'latitude' in geo_data and 'longitude' in geo_data:
                    self.lat_ref = float(geo_data['latitude'])
                    self.lon_ref = float(geo_data['longitude'])
                elif 'lat' in geo_data and 'lon' in geo_data:
                    self.lat_ref = float(geo_data['lat'])
                    self.lon_ref = float(geo_data['lon'])
                else:
                    print(f"⚠️ Formato JSON não reconhecido: {geo_data}")
                    return
            except json.JSONDecodeError:
                # Se não for JSON, tenta parsear como string simples
                # Formato esperado: "lat,lon" ou "latitude:X longitude:Y"
                if ',' in data:
                    parts = data.split(',')
                    if len(parts) >= 2:
                        self.lat_ref = float(parts[0].strip())
                        self.lon_ref = float(parts[1].strip())
                elif 'latitude' in data.lower() and 'longitude' in data.lower():
                    # Extrai números do texto
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', data)
                    if len(numbers) >= 2:
                        self.lat_ref = float(numbers[0])
                        self.lon_ref = float(numbers[1])
                else:
                    print(f"⚠️ Formato de coordenadas não reconhecido: {data}")
                    return
            
            self.geo_coordinates_received = True
            print(f"✅ Coordenadas de referência recebidas: lat={self.lat_ref}, lon={self.lon_ref}")
            
        except Exception as e:
            print(f"❌ Erro ao processar coordenadas geográficas: {e}")
    
    def wait_for_geo_coordinates(self, timeout: float = 10.0) -> bool:
        """Aguarda as coordenadas geográficas serem recebidas"""
        print("⏳ Aguardando coordenadas geográficas do Gazebo...")
        start_time = time.time()
        
        while not self.geo_coordinates_received and time.time() - start_time < timeout:
            rospy.sleep(0.1)
        
        if not self.geo_coordinates_received:
            print("⚠️ Timeout ao aguardar coordenadas geográficas, usando valores padrão")
            self.lat_ref = -3.123199
            self.lon_ref = -41.764537
            return False
        
        return True
    
    def get_model_list_from_states(self) -> List[str]:
        """Obtém lista de modelos a partir do tópico ModelStates"""
        try:
            # Aguarda uma mensagem do tópico model_states
            msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=10.0)
            return list(msg.name)
        except rospy.ROSException as e:
            print(f"❌ Erro ao obter lista de modelos: {e}")
            return []
    
    def get_link_list_from_states(self) -> List[str]:
        """Obtém lista de links a partir do tópico LinkStates"""
        try:
            # Aguarda uma mensagem do tópico link_states
            msg = rospy.wait_for_message('/gazebo/link_states', LinkStates, timeout=10.0)
            return list(msg.name)
        except rospy.ROSException as e:
            print(f"❌ Erro ao obter lista de links: {e}")
            return []
    
    def extract_links_from_gazebo(self) -> Dict[str, Dict]:
        """Extrai TODOS os links dos modelos via tópico LinkStates"""
        print("🔍 Extraindo TODOS os links via tópico LinkStates...")
        
        try:
            # Obtém dados dos links
            msg = rospy.wait_for_message('/gazebo/link_states', LinkStates, timeout=10.0)
            
            links_data = {}
            
            for i, link_name in enumerate(msg.name):
                # Filtra links do sistema e rovers
                if any(excluded in link_name.lower() for excluded in ['ground_plane', 'sun', 'rover']):
                    continue
                
                pose = msg.pose[i]
                
                # Extrai posição x, y, z
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Converte para GPS
                latitude, longitude = self.gazebo_to_gps_coords(x, y)
                
                # Obtém dimensões do objeto
                width, height = self.get_object_dimensions(link_name)
                
                # Gera coordenadas dos vértices
                vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                
                # Gera ID único
                object_id = f"ef_{link_name.lower().replace(' ', '_').replace('::', '_')}"
                
                # Armazena os dados
                links_data[link_name] = {
                    'Model Name': link_name,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Altitude': int(z + 77),  # z + 77 como solicitado
                    'Vx': vx_coords,
                    'Vy': vy_coords,
                    'Py': y,  # Coordenada Y cartesiana
                    'Px': x,  # Coordenada X cartesiana
                    'Vx_largura': width,
                    'Vy_altura': height,
                    'ID': object_id
                }
                
                print(f"  ✅ {link_name}: ({x:.2f}, {y:.2f})")
            
            return links_data
            
        except rospy.ROSException as e:
            print(f"❌ Erro ao obter links: {e}")
            return {}
    
    def gazebo_to_gps_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Converte coordenadas Gazebo para GPS usando coordenadas de referência do tópico geo_coordinates"""
        if self.lat_ref is None or self.lon_ref is None:
            print("⚠️ Coordenadas de referência não disponíveis, usando valores padrão")
            self.lat_ref = -3.123199
            self.lon_ref = -41.764537
        
        # Raio da Terra usado pelo ArduPilot
        radius_of_earth = 6378100.0  # metros
        
        # SEM ROTAÇÃO: Usa as coordenadas diretamente do tópico do Gazebo
        # As coordenadas já estão na orientação correta
        
        # Converte metros para graus usando constantes do ArduPilot
        # Latitude: variação Norte-Sul (Y direto)
        lat = y / (radius_of_earth * math.pi / 180.0) + self.lat_ref
        # Longitude: variação Leste-Oeste (X direto)
        lon = x / (radius_of_earth * math.cos(math.radians(self.lat_ref)) * math.pi / 180.0) + self.lon_ref
        
        return lat, lon
    
    def get_object_dimensions(self, object_name: str) -> Tuple[float, float]:
        """Determina as dimensões do objeto baseado no nome"""
        name_lower = object_name.lower()
        
        # Procura por padrões no nome do objeto
        for key, dimensions in self.DIMENSIONS.items():
            if key in name_lower:
                return dimensions
        
        # Se não encontrou, usa dimensão padrão
        return (1.0, 1.0)
    
    def generate_vertex_coordinates(self, x: float, y: float, width: float, height: float) -> Tuple[List, List]:
        """Gera coordenadas dos vértices do objeto"""
        # Calcula os vértices do retângulo
        half_width = width / 2
        half_height = height / 2
        
        # Vértices em coordenadas cartesianas
        vertices_x = [
            x - half_width,  # Vértice inferior esquerdo
            x + half_width   # Vértice inferior direito
        ]
        
        vertices_y = [
            y - half_height,  # Vértice inferior esquerdo
            y + half_height   # Vértice superior esquerdo
        ]
        
        # Converte para GPS
        vx_gps = []
        vy_gps = []
        
        for vx, vy in zip(vertices_x, vertices_y):
            lat, lon = self.gazebo_to_gps_coords(vx, vy)
            vx_gps.append((lat, lon))
        
        for vx, vy in zip([x - half_width, x + half_width], [y - half_height, y + half_height]):
            lat, lon = self.gazebo_to_gps_coords(vx, vy)
            vy_gps.append((lat, lon))
        
        return vx_gps, vy_gps
    
    def model_states_callback(self, msg: ModelStates):
        """Callback para receber estados dos modelos"""
        for i, model_name in enumerate(msg.name):
            if model_name in self.objects_to_monitor:
                pose = msg.pose[i]
                
                # Extrai posição x, y, z
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Converte para GPS
                latitude, longitude = self.gazebo_to_gps_coords(x, y)
                
                # Obtém dimensões do objeto
                width, height = self.get_object_dimensions(model_name)
                
                # Gera coordenadas dos vértices
                vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                
                # Gera ID único
                object_id = f"ef_{model_name.lower().replace(' ', '_')}"
                
                # Armazena os dados no formato do obstaculos_processado6.xlsx
                self.objects_data[model_name] = {
                    'Model Name': model_name,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Altitude': int(z + 77),  # z + 77 como solicitado
                    'Vx': vx_coords,
                    'Vy': vy_coords,
                    'Py': y,  # Coordenada Y cartesiana
                    'Px': x,  # Coordenada X cartesiana
                    'Vx_largura': width,
                    'Vy_altura': height,
                    'ID': object_id
                }
    
    def get_objects_from_tf(self) -> Dict[str, Dict]:
        """Obtém posições dos objetos via TF frames"""
        objects_tf = {}
        
        try:
            # Obtém lista de frames disponíveis
            frames = self.tf_buffer.all_frames_as_string()
            
            # Lista de frames comuns de objetos no Gazebo
            common_object_frames = [
                'ground_plane', 'sun', 'box', 'sphere', 'cylinder',
                'cone', 'mesh', 'building', 'tree', 'car', 'person'
            ]
            
            for frame_id in self.tf_buffer.all_frames_as_string().split('\n'):
                if frame_id.strip():
                    try:
                        # Tenta obter transformação para o frame
                        transform = self.tf_buffer.lookup_transform(
                            'world', frame_id, rospy.Time(0), timeout=rospy.Duration(1.0)
                        )
                        
                        # Extrai posição
                        x = transform.transform.translation.x
                        y = transform.transform.translation.y
                        z = transform.transform.translation.z
                        
                        # Converte para GPS
                        latitude, longitude = self.gazebo_to_gps_coords(x, y)
                        
                        # Obtém dimensões do objeto
                        width, height = self.get_object_dimensions(frame_id)
                        
                        # Gera coordenadas dos vértices
                        vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                        
                        # Gera ID único
                        object_id = f"ef_{frame_id.lower().replace(' ', '_')}"
                        
                        objects_tf[frame_id] = {
                            'Model Name': frame_id,
                            'Latitude': latitude,
                            'Longitude': longitude,
                            'Altitude': int(z + 77),
                            'Vx': vx_coords,
                            'Vy': vy_coords,
                            'Py': y,
                            'Px': x,
                            'Vx_largura': width,
                            'Vy_altura': height,
                            'ID': object_id
                        }
                        
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                            tf2_ros.ExtrapolationException):
                        continue
                        
        except Exception as e:
            print(f"⚠️ Erro ao obter objetos via TF: {e}")
        
        return objects_tf
    
    def extract_objects_from_world_file(self, world_file_path: str = None) -> pd.DataFrame:
        """Extrai TODOS os objetos e subobjetos do arquivo do mundo Gazebo, mantendo filtros de exclusão"""
        import xml.etree.ElementTree as ET
        
        if world_file_path is None:
            # Define caminhos baseado no tipo de mundo selecionado
            if self.world_type == "charlie_delta":
                possible_paths = [
                    "/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/models/CHARLIE_DELTA/model.sdf",
                    "/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/worlds/parnaibaiii_charlie_delta.world",
                    "./parnaibaiii_charlie_delta.world"
                ]
            else:  # mundo completo (padrão)
                possible_paths = [
                    "/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/models/ARGO_PARNAIBAIII_V3/model.sdf",
                    "/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/worlds/parnaibaiii_simple_v3.world",
                    "/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/worlds/parnaibaiii_simple_v3_com_bolas_azuis.world",
                    "./parnaibaiii_simple_v3_com_bolas_azuis.world"
                ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    world_file_path = path
                    break
        
        if not world_file_path or not os.path.exists(world_file_path):
            print("⚠️ Arquivo do mundo não encontrado para extração estática")
            return pd.DataFrame()
        
        print(f"📖 Extraindo objetos do arquivo: {world_file_path}")
        
        try:
            tree = ET.parse(world_file_path)
            root = tree.getroot()
            
            objects_data = []
            object_id = 1
            
            # Verifica se é um arquivo de modelo (.sdf) ou mundo (.world)
            if world_file_path.endswith('.sdf'):
                # É um arquivo de modelo - procura por links
                print("🔍 Extraindo links do arquivo de modelo...")
                
                # Procura pelo modelo baseado no tipo de mundo
                if self.world_type == "charlie_delta":
                    argo_model = root.find('.//model[@name="CHARLIE_DELTA"]')
                    model_name = "CHARLIE_DELTA"
                else:
                    argo_model = root.find('.//model[@name="ARGO_PARNAIBAIII_V3"]')
                    model_name = "ARGO_PARNAIBAIII_V3"
                    
                if argo_model is not None:
                    print(f"🎯 Encontrado modelo {model_name}")
                    
                    # Lista de objetos a serem excluídos baseado no tipo de mundo
                    if self.world_type == "charlie_delta":
                        excluded_objects = [
                            'PAREDEREATOR', 'fence', 'plane_simple', 'DIVERSOS', 
                            'casinha_vermelha', 'ground', 'ESTRUTURA', 'CABEAMENTO'
                        ]
                    else:
                        excluded_objects = [
                            'PAREDEREATOR', 'fence', 'plane_simple', 'DIVERSOS', 
                            'casinha_vermelha', 'ground', 'ESTRUTURA', 'CABEAMENTO'
                        ]
                    
                    # Procura por TODOS os links dentro do modelo
                    for link in argo_model.findall('.//link'):
                        link_name = link.get('name')
                        if link_name:  # Aceita TODOS os links, incluindo base_link
                            # Verifica se o nome do objeto deve ser excluído (mantém filtros)
                            should_exclude = False
                            for excluded in excluded_objects:
                                if excluded in link_name:
                                    should_exclude = True
                                    break
                            
                            if should_exclude:
                                print(f"  ❌ Excluído: {link_name}")
                                continue
                            # Extrai posição do link
                            pose_elem = link.find('pose')
                            if pose_elem is not None:
                                pose_text = pose_elem.text.strip()
                                pose_values = [float(x) for x in pose_text.split()]
                                if len(pose_values) >= 2:
                                    x, y = pose_values[0], pose_values[1]
                                    z = pose_values[2] if len(pose_values) > 2 else 0.0
                                    
                                    # Converte para GPS
                                    latitude, longitude = self.gazebo_to_gps_coords(x, y)
                                    
                                    # Obtém dimensões do objeto
                                    width, height = self.get_object_dimensions(link_name)
                                    
                                    # Gera coordenadas dos vértices
                                    vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                                    
                                    # Gera ID único
                                    object_id_str = f"ef_{link_name.lower().replace(' ', '_')}"
                                    
                                    objects_data.append({
                                        'Model Name': link_name,
                                        'Latitude': latitude,
                                        'Longitude': longitude,
                                        'Altitude': int(z + 77),
                                        'Vx': vx_coords,
                                        'Vy': vy_coords,
                                        'Py': y,
                                        'Px': x,
                                        'Vx_largura': width,
                                        'Vy_altura': height,
                                        'ID': object_id_str
                                    })
                                    object_id += 1
                                    print(f"  ✅ {link_name}: ({x:.2f}, {y:.2f})")
                else:
                    print(f"⚠️ Modelo {model_name if 'model_name' in locals() else 'ARGO_PARNAIBAIII_V3'} não encontrado no arquivo SDF")
            else:
                # É um arquivo de mundo - procura por modelos
                print("🔍 Extraindo modelos do arquivo de mundo...")
                
                # Procura especificamente pelo modelo ARGO_PARNAIBAIII_V3
                argo_model = None
                for model in root.findall('.//model'):
                    name = model.get('name')
                    if name and 'ARGO_PARNAIBAIII_V3' in name:
                        argo_model = model
                        print(f"🎯 Encontrado modelo principal: {name}")
                        break
                
                if argo_model is not None:
                    # Procura por modelos aninhados dentro do ARGO_PARNAIBAIII_V3
                    print("🔍 Procurando modelos secundários dentro do ARGO_PARNAIBAIII_V3...")
                    
                    # Procura por includes (modelos referenciados)
                    for include in argo_model.findall('.//include'):
                        name_elem = include.find('name')
                        if name_elem is not None:
                            model_name = name_elem.text.strip()
                            if model_name:  # Aceita TODOS os modelos incluídos
                                # Extrai posição do include
                                pose_elem = include.find('pose')
                                if pose_elem is not None:
                                    pose_text = pose_elem.text.strip()
                                    pose_values = [float(x) for x in pose_text.split()]
                                    if len(pose_values) >= 2:
                                        x, y = pose_values[0], pose_values[1]
                                        z = pose_values[2] if len(pose_values) > 2 else 0.0
                                        
                                        # Converte para GPS
                                        latitude, longitude = self.gazebo_to_gps_coords(x, y)
                                        
                                        # Obtém dimensões do objeto
                                        width, height = self.get_object_dimensions(model_name)
                                        
                                        # Gera coordenadas dos vértices
                                        vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                                        
                                        # Gera ID único
                                        object_id_str = f"ef_{model_name.lower().replace(' ', '_')}"
                                        
                                        objects_data.append({
                                            'Model Name': model_name,
                                            'Latitude': latitude,
                                            'Longitude': longitude,
                                            'Altitude': int(z + 77),
                                            'Vx': vx_coords,
                                            'Vy': vy_coords,
                                            'Py': y,
                                            'Px': x,
                                            'Vx_largura': width,
                                            'Vy_altura': height,
                                            'ID': object_id_str
                                        })
                                        object_id += 1
                                        print(f"  ✅ {model_name}: ({x:.2f}, {y:.2f})")
                    
                    # Também procura por TODOS os modelos diretos dentro do modelo principal
                    for model in argo_model.findall('.//model'):
                        name = model.get('name')
                        if name and name not in [model_name]:  # Exclui apenas o modelo pai
                            # Extrai posição
                            pose_elem = model.find('pose')
                            if pose_elem is not None:
                                pose_text = pose_elem.text.strip()
                                pose_values = [float(x) for x in pose_text.split()]
                                if len(pose_values) >= 2:
                                    x, y = pose_values[0], pose_values[1]
                                    z = pose_values[2] if len(pose_values) > 2 else 0.0
                                    
                                    # Converte para GPS
                                    latitude, longitude = self.gazebo_to_gps_coords(x, y)
                                    
                                    # Obtém dimensões do objeto
                                    width, height = self.get_object_dimensions(name)
                                    
                                    # Gera coordenadas dos vértices
                                    vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                                    
                                    # Gera ID único
                                    object_id_str = f"ef_{name.lower().replace(' ', '_')}"
                                    
                                    objects_data.append({
                                        'Model Name': name,
                                        'Latitude': latitude,
                                        'Longitude': longitude,
                                        'Altitude': int(z + 77),
                                        'Vx': vx_coords,
                                        'Vy': vy_coords,
                                        'Py': y,
                                        'Px': x,
                                        'Vx_largura': width,
                                        'Vy_altura': height,
                                        'ID': object_id_str
                                    })
                                    object_id += 1
                                    print(f"  ✅ {name}: ({x:.2f}, {y:.2f})")
                else:
                    print("⚠️ Modelo ARGO_PARNAIBAIII_V3 não encontrado no arquivo")
                    # Fallback: procura por TODOS os modelos
                    for model in root.findall('.//model'):
                        name = model.get('name')
                        if name:  # Aceita TODOS os modelos
                            # Extrai posição
                            pose_elem = model.find('pose')
                            if pose_elem is not None:
                                pose_text = pose_elem.text.strip()
                                pose_values = [float(x) for x in pose_text.split()]
                                if len(pose_values) >= 2:
                                    x, y = pose_values[0], pose_values[1]
                                    z = pose_values[2] if len(pose_values) > 2 else 0.0
                                    
                                    # Converte para GPS
                                    latitude, longitude = self.gazebo_to_gps_coords(x, y)
                                    
                                    # Obtém dimensões do objeto
                                    width, height = self.get_object_dimensions(name)
                                    
                                    # Gera coordenadas dos vértices
                                    vx_coords, vy_coords = self.generate_vertex_coordinates(x, y, width, height)
                                    
                                    # Gera ID único
                                    object_id_str = f"ef_{name.lower().replace(' ', '_')}"
                                    
                                    objects_data.append({
                                        'Model Name': name,
                                        'Latitude': latitude,
                                        'Longitude': longitude,
                                        'Altitude': int(z + 77),
                                        'Vx': vx_coords,
                                        'Vy': vy_coords,
                                        'Py': y,
                                        'Px': x,
                                        'Vx_largura': width,
                                        'Vy_altura': height,
                                        'ID': object_id_str
                                    })
                                    object_id += 1
            
            if objects_data:
                df = pd.DataFrame(objects_data)
                print(f"✅ Extraídos {len(df)} objetos do arquivo do mundo!")
                return df
            else:
                print("⚠️ Nenhum objeto encontrado no arquivo do mundo")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Erro ao processar arquivo do mundo: {e}")
            return pd.DataFrame()

    def extract_all_objects(self, timeout: float = 30.0) -> pd.DataFrame:
        """Extrai posições de todos os objetos do mundo Gazebo aberto via ROS"""
        print("🔍 Extraindo objetos do mundo Gazebo aberto via ROS...")
        
        # Primeiro, aguarda as coordenadas geográficas
        self.wait_for_geo_coordinates(timeout=10.0)
        
        # ESTRATÉGIA PRINCIPAL: Extrair TODOS os links via LinkStates
        print("🎯 Modo: TODOS OS OBJETOS E SUBOBJETOS via LinkStates")
        
        # Extrai TODOS os links do Gazebo
        links_data = self.extract_links_from_gazebo()
        
        if links_data:
            print(f"✅ Extraídos {len(links_data)} links via LinkStates!")
            all_objects = links_data
        else:
            print("⚠️ Falha na extração via LinkStates, tentando método alternativo...")
            
            # Fallback: método original com modelos
            model_names = self.get_model_list_from_states()
            if not model_names:
                print("❌ Nenhum modelo encontrado no Gazebo!")
                return pd.DataFrame()
            
            print(f"📋 Encontrados {len(model_names)} modelos no mundo aberto:")
            for name in model_names:
                print(f"  - {name}")
            
            # Lista de objetos do sistema a serem sempre excluídos
            system_excluded = ['ground_plane', 'sun', 'default', 'world']
            
            # Monitora TODOS os modelos encontrados, excluindo apenas rovers e objetos do sistema
            self.objects_to_monitor = [name for name in model_names 
                                     if not any(excluded in name.lower() for excluded in system_excluded)
                                     and 'rover' not in name.lower()]
            
            print(f"🎯 Monitorando {len(self.objects_to_monitor)} objetos relevantes")
            
            # Subscreve ao tópico de estados dos modelos
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
            
            # Aguarda coleta de dados
            print(f"⏳ Coletando dados por {timeout} segundos...")
            start_time = time.time()
            
            while time.time() - start_time < timeout and not rospy.is_shutdown():
                rospy.sleep(0.1)
                
                # Verifica se já coletou dados suficientes
                if len(self.objects_data) >= len(self.objects_to_monitor):
                    print(f"✅ Coletados dados de todos os {len(self.objects_data)} objetos monitorados")
                    break
            
            # Tenta obter objetos via TF também
            print("🔍 Procurando objetos via TF frames...")
            objects_tf = self.get_objects_from_tf()
            
            # Combina dados de modelos e TF
            all_objects = {}
            all_objects.update(self.objects_data)
            
            # Adiciona objetos TF que não foram encontrados via model_states
            for name, data in objects_tf.items():
                if name not in all_objects:
                    all_objects[name] = data
        
        if not all_objects:
            print("❌ Nenhum objeto encontrado!")
            # Fallback: tenta extrair do arquivo se não conseguiu via ROS
            print("🔄 Tentando fallback para extração de arquivo...")
            return self.extract_objects_from_world_file()
        
        # Converte para DataFrame (já está no formato correto)
        df_data = list(all_objects.values())
        
        df = pd.DataFrame(df_data)
        print(f"✅ Extraídos {len(df)} objetos com sucesso do mundo aberto!")
        
        return df
    
    def save_to_files(self, df: pd.DataFrame, base_name: str = "todos_pontos_gps"):
        """Salva os dados em arquivos CSV e Excel dentro da pasta Gazebo/"""
        if df.empty:
            print("❌ Nenhum dado para salvar!")
            return
        
        # Define o diretório da pasta Gazebo
        gazebo_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Mantém o nome base sem sufixos adicionais
        # base_name permanece como "todos_pontos_gps"
        
        # Salva CSV
        csv_file = os.path.join(gazebo_dir, f"{base_name}.csv")
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV salvo: {csv_file}")
        
        # Salva Excel
        excel_file = os.path.join(gazebo_dir, f"{base_name}.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"💾 Excel salvo: {excel_file}")
        
        # Mostra estatísticas
        print(f"\n📊 Estatísticas:")
        print(f"  - Total de objetos: {len(df)}")
        print(f"  - Coordenadas Px: {df['Px'].min():.2f} a {df['Px'].max():.2f}")
        print(f"  - Coordenadas Py: {df['Py'].min():.2f} a {df['Py'].max():.2f}")
        print(f"  - Latitude: {df['Latitude'].min():.6f} a {df['Latitude'].max():.6f}")
        print(f"  - Longitude: {df['Longitude'].min():.6f} a {df['Longitude'].max():.6f}")
        print(f"  - Altitude: {df['Altitude'].min()} a {df['Altitude'].max()}")
        print(f"  - Vx_largura: {df['Vx_largura'].min():.3f} a {df['Vx_largura'].max():.3f}")
        print(f"  - Vy_altura: {df['Vy_altura'].min():.3f} a {df['Vy_altura'].max():.3f}")
        
        # Estatísticas por tipo de objeto
        reatores = df[df['Model Name'].str.contains('REATOR', case=False, na=False)]
        outros = df[~df['Model Name'].str.contains('REATOR', case=False, na=False)]
        print(f"  - Reatores: {len(reatores)} objetos")
        print(f"  - Outros objetos: {len(outros)} objetos")

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Extrai posições de objetos do mundo Gazebo e converte para coordenadas GPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python3 Gazebo2CSV.py                    # Usa mundo completo (padrão)
  python3 Gazebo2CSV.py --mundo completo   # Usa mundo completo
  python3 Gazebo2CSV.py --mundo charlie_delta  # Usa mundo charlie_delta
  python3 Gazebo2CSV.py -m charlie_delta   # Forma abreviada
        """
    )
    
    parser.add_argument(
        '--mundo', '-m',
        choices=['completo', 'charlie_delta'],
        default='completo',
        help='Tipo de mundo para extrair objetos (padrão: completo)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=30.0,
        help='Tempo limite para coleta de dados em segundos (padrão: 30.0)'
    )
    
    return parser.parse_args()

def main():
    """Função principal"""
    # Processa argumentos da linha de comando
    args = parse_arguments()
    
    print("🚀 Iniciando extração de objetos do Gazebo via ROS...")
    print(f"🌍 Mundo selecionado: {args.mundo}")
    print(f"⏱️ Timeout: {args.timeout} segundos")
    
    try:
        # Cria o extrator com o tipo de mundo especificado
        extractor = GazeboObjectExtractor(world_type=args.mundo)
        
        # Aguarda um pouco para o ROS inicializar
        rospy.sleep(2.0)
        
        print("🔗 Conectando ao Gazebo e aguardando dados...")
        
        # Extrai todos os objetos do mundo aberto
        df = extractor.extract_all_objects(timeout=args.timeout)
        
        if not df.empty:
            # Salva os arquivos
            extractor.save_to_files(df, "todos_pontos_gps")
            print("✅ Extração concluída com sucesso!")
        else:
            print("❌ Nenhum objeto foi extraído!")
            
    except rospy.ROSInterruptException:
        print("⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
