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
from gazebo_msgs.msg import ModelStates
import pandas as pd
import math
import time
import os
import sys
from typing import Dict, List, Tuple, Optional

class GazeboObjectExtractor:
    def __init__(self):
        """Inicializa o extrator de objetos do Gazebo"""
        rospy.init_node('gazebo_object_extractor', anonymous=True)
        
        # Coordenadas de referência do mundo Gazebo (mesmas usadas nos outros scripts)
        self.lat_ref = -3.123199
        self.lon_ref = -41.764537
        
        # Buffer para transformações TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Dicionário para armazenar posições dos objetos
        self.objects_data = {}
        
        # Lista de objetos para monitorar
        self.objects_to_monitor = []
        
        print("🚀 Inicializando extrator de objetos do Gazebo...")
        print(f"🌍 Coordenadas de referência: lat={self.lat_ref}, lon={self.lon_ref}")
    
    def get_model_list_from_states(self) -> List[str]:
        """Obtém lista de modelos a partir do tópico ModelStates"""
        try:
            # Aguarda uma mensagem do tópico model_states
            msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=10.0)
            return list(msg.name)
        except rospy.ROSException as e:
            print(f"❌ Erro ao obter lista de modelos: {e}")
            return []
    
    def gazebo_to_gps_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Converte coordenadas Gazebo para GPS"""
        lat = y / 111132.0 + self.lat_ref
        lon = x / (111320.0 * math.cos(math.radians(self.lat_ref))) + self.lon_ref
        return lat, lon
    
    def model_states_callback(self, msg: ModelStates):
        """Callback para receber estados dos modelos"""
        for i, model_name in enumerate(msg.name):
            if model_name in self.objects_to_monitor:
                pose = msg.pose[i]
                
                # Extrai posição x, y
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Converte para GPS
                latitude, longitude = self.gazebo_to_gps_coords(x, y)
                
                # Define tamanhos Px e Py baseado no tipo de objeto
                if 'REATOR' in model_name:
                    px, py = 3.0, 3.0  # Reatores: tamanho 3
                else:
                    px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                
                # Armazena os dados
                self.objects_data[model_name] = {
                    'id': len(self.objects_data) + 1,
                    'label': model_name,
                    'tipo': 'objeto',
                    'x_cartesiano': x,
                    'y_cartesiano': y,
                    'z_cartesiano': z,
                    'latitude': latitude,
                    'longitude': longitude,
                    'Px': px,
                    'Py': py
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
                        
                        # Define tamanhos Px e Py baseado no tipo de objeto
                        if 'REATOR' in frame_id:
                            px, py = 3.0, 3.0  # Reatores: tamanho 3
                        else:
                            px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                        
                        objects_tf[frame_id] = {
                            'id': len(objects_tf) + 1,
                            'label': frame_id,
                            'tipo': 'objeto_tf',
                            'x_cartesiano': x,
                            'y_cartesiano': y,
                            'z_cartesiano': z,
                            'latitude': latitude,
                            'longitude': longitude,
                            'Px': px,
                            'Py': py
                        }
                        
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                            tf2_ros.ExtrapolationException):
                        continue
                        
        except Exception as e:
            print(f"⚠️ Erro ao obter objetos via TF: {e}")
        
        return objects_tf
    
    def extract_objects_from_world_file(self, world_file_path: str = None) -> pd.DataFrame:
        """Extrai objetos do arquivo do mundo Gazebo, focando nos modelos secundários do ARGO_PARNAIBAIII_V3"""
        import xml.etree.ElementTree as ET
        
        if world_file_path is None:
            # Tenta encontrar o arquivo do modelo ARGO_PARNAIBAIII_V3
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
                
                # Procura pelo modelo ARGO_PARNAIBAIII_V3
                argo_model = root.find('.//model[@name="ARGO_PARNAIBAIII_V3"]')
                if argo_model is not None:
                    print("🎯 Encontrado modelo ARGO_PARNAIBAIII_V3")
                    
                    # Lista de objetos a serem excluídos
                    excluded_objects = [
                        'PAREDEREATOR', 'fence', 'plane_simple', 'DIVERSOS', 
                        'casinha_vermelha', 'ground', 'ESTRUTURA', 'CABEAMENTO'
                    ]
                    
                    # Procura por todos os links dentro do modelo
                    for link in argo_model.findall('.//link'):
                        link_name = link.get('name')
                        if link_name and link_name not in ['base_link', 'world']:
                            # Verifica se o nome do objeto deve ser excluído
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
                                    
                                    # Define tamanhos Px e Py baseado no tipo de objeto
                                    if 'REATOR' in link_name:
                                        px, py = 3.0, 3.0  # Reatores: tamanho 3
                                    else:
                                        px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                                    
                                    objects_data.append({
                                        'id': object_id,
                                        'label': link_name,
                                        'tipo': 'equipamento',
                                        'x_cartesiano': x,
                                        'y_cartesiano': y,
                                        'z_cartesiano': z,
                                        'latitude': latitude,
                                        'longitude': longitude,
                                        'Px': px,
                                        'Py': py
                                    })
                                    object_id += 1
                                    print(f"  ✅ {link_name}: ({x:.2f}, {y:.2f})")
                else:
                    print("⚠️ Modelo ARGO_PARNAIBAIII_V3 não encontrado no arquivo SDF")
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
                            if model_name and 'rover' not in model_name.lower():
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
                                        
                                        # Define tamanhos Px e Py baseado no tipo de objeto
                                        if 'REATOR' in model_name:
                                            px, py = 3.0, 3.0  # Reatores: tamanho 3
                                        else:
                                            px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                                        
                                        objects_data.append({
                                            'id': object_id,
                                            'label': model_name,
                                            'tipo': 'equipamento',
                                            'x_cartesiano': x,
                                            'y_cartesiano': y,
                                            'z_cartesiano': z,
                                            'latitude': latitude,
                                            'longitude': longitude,
                                            'Px': px,
                                            'Py': py
                                        })
                                        object_id += 1
                                        print(f"  ✅ {model_name}: ({x:.2f}, {y:.2f})")
                    
                    # Também procura por modelos diretos dentro do ARGO_PARNAIBAIII_V3
                    for model in argo_model.findall('.//model'):
                        name = model.get('name')
                        if name and 'rover' not in name.lower() and name != 'ARGO_PARNAIBAIII_V3':
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
                                    
                                    # Define tamanhos Px e Py baseado no tipo de objeto
                                    if 'REATOR' in name:
                                        px, py = 3.0, 3.0  # Reatores: tamanho 3
                                    else:
                                        px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                                    
                                    objects_data.append({
                                        'id': object_id,
                                        'label': name,
                                        'tipo': 'equipamento',
                                        'x_cartesiano': x,
                                        'y_cartesiano': y,
                                        'z_cartesiano': z,
                                        'latitude': latitude,
                                        'longitude': longitude,
                                        'Px': px,
                                        'Py': py
                                    })
                                    object_id += 1
                                    print(f"  ✅ {name}: ({x:.2f}, {y:.2f})")
                else:
                    print("⚠️ Modelo ARGO_PARNAIBAIII_V3 não encontrado no arquivo")
                    # Fallback: procura por todos os modelos que não são rovers
                    for model in root.findall('.//model'):
                        name = model.get('name')
                        if name and 'rover' not in name.lower():
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
                                    
                                    # Define tamanhos Px e Py baseado no tipo de objeto
                                    if 'REATOR' in name:
                                        px, py = 3.0, 3.0  # Reatores: tamanho 3
                                    else:
                                        px, py = 1.0, 1.0  # Outros objetos: tamanho 1
                                    
                                    objects_data.append({
                                        'id': object_id,
                                        'label': name,
                                        'tipo': 'objeto_mundo',
                                        'x_cartesiano': x,
                                        'y_cartesiano': y,
                                        'z_cartesiano': z,
                                        'latitude': latitude,
                                        'longitude': longitude,
                                        'Px': px,
                                        'Py': py
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
        """Extrai posições de todos os objetos do mundo Gazebo, focando nos modelos secundários do ARGO_PARNAIBAIII_V3"""
        print("🔍 Procurando modelos secundários do ARGO_PARNAIBAIII_V3...")
        
        # Força a extração do arquivo do mundo para obter os modelos secundários
        print("📖 Extraindo modelos secundários do arquivo do mundo...")
        df = self.extract_objects_from_world_file()
        
        if not df.empty:
            return df
        
        # Se não conseguiu extrair do arquivo, tenta via ROS como fallback
        print("⚠️ Não conseguiu extrair do arquivo, tentando via ROS...")
        
        # Obtém lista de modelos
        model_names = self.get_model_list_from_states()
        if not model_names:
            print("❌ Nenhum modelo encontrado no Gazebo!")
            return pd.DataFrame()
        
        print(f"📋 Encontrados {len(model_names)} modelos:")
        for name in model_names:
            print(f"  - {name}")
        
        # Filtra objetos relevantes (exclui rovers e objetos do sistema)
        excluded_objects = ['ground_plane', 'sun', 'default', 'world', 'rover_argo']
        self.objects_to_monitor = [name for name in model_names 
                                 if not any(excluded in name.lower() for excluded in excluded_objects)]
        
        # Se não encontrou objetos específicos, usa todos exceto rovers
        if not self.objects_to_monitor:
            self.objects_to_monitor = [name for name in model_names 
                                     if 'rover' not in name.lower()]
        
        print(f"🎯 Monitorando {len(self.objects_to_monitor)} objetos relevantes")
        
        # Subscreve ao tópico de estados dos modelos
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        
        # Aguarda coleta de dados
        print(f"⏳ Coletando dados por {timeout} segundos...")
        start_time = time.time()
        
        while time.time() - start_time < timeout and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
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
            return pd.DataFrame()
        
        # Converte para DataFrame
        df_data = []
        for i, (name, data) in enumerate(all_objects.items(), 1):
            df_data.append({
                'id': i,
                'label': data['label'],
                'tipo': data['tipo'],
                'x_cartesiano': data['x_cartesiano'],
                'y_cartesiano': data['y_cartesiano'],
                'z_cartesiano': data.get('z_cartesiano', 0.0),
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'Px': data.get('Px', 1.0),
                'Py': data.get('Py', 1.0)
            })
        
        df = pd.DataFrame(df_data)
        print(f"✅ Extraídos {len(df)} objetos com sucesso!")
        
        return df
    
    def save_to_files(self, df: pd.DataFrame, base_name: str = "todos_pontos_gps"):
        """Salva os dados em arquivos CSV e Excel"""
        if df.empty:
            print("❌ Nenhum dado para salvar!")
            return
        
        # Salva CSV
        csv_file = f"{base_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 CSV salvo: {csv_file}")
        
        # Salva Excel
        excel_file = f"{base_name}.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"💾 Excel salvo: {excel_file}")
        
        # Mostra estatísticas
        print(f"\n📊 Estatísticas:")
        print(f"  - Total de objetos: {len(df)}")
        print(f"  - Coordenadas X: {df['x_cartesiano'].min():.2f} a {df['x_cartesiano'].max():.2f}")
        print(f"  - Coordenadas Y: {df['y_cartesiano'].min():.2f} a {df['y_cartesiano'].max():.2f}")
        print(f"  - Latitude: {df['latitude'].min():.6f} a {df['latitude'].max():.6f}")
        print(f"  - Longitude: {df['longitude'].min():.6f} a {df['longitude'].max():.6f}")
        
        # Estatísticas dos tamanhos
        reatores = df[df['label'].str.contains('REATOR', case=False, na=False)]
        outros = df[~df['label'].str.contains('REATOR', case=False, na=False)]
        print(f"  - Reatores: {len(reatores)} objetos (Px=Py=3.0)")
        print(f"  - Outros objetos: {len(outros)} objetos (Px=Py=1.0)")

def main():
    """Função principal"""
    print("🚀 Iniciando extração de objetos do Gazebo via ROS...")
    
    try:
        # Cria o extrator
        extractor = GazeboObjectExtractor()
        
        # Aguarda um pouco para o ROS inicializar
        rospy.sleep(2.0)
        
        # Extrai todos os objetos
        df = extractor.extract_all_objects(timeout=30.0)
        
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
