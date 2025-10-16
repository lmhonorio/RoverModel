#!/usr/bin/env python3
"""
Script para criar um mundo Gazebo modificado com bolas baseado na saída do Gazebo2CSV.py
Combina funcionalidades dos scripts adicionar_bolas_gazebo.py e adicionar_bolas_azuis_gazebo.py
"""

import pandas as pd
import xml.etree.ElementTree as ET
import json
import math
import os
import sys

def gps_to_gazebo_coords(lat, lon, lat_ref, lon_ref):
    """Converte coordenadas GPS para coordenadas locais do Gazebo"""
    x = (lon - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
    y = (lat - lat_ref) * 111132.0
    return x, y

def carregar_pontos_json(json_file):
    """Carrega pontos do arquivo JSON graph_equipment.json"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        pontos = []
        if 'nodes' in data:
            for node_id, node_data in data['nodes'].items():
                if 'pos' in node_data and len(node_data['pos']) >= 2:
                    x, y = node_data['pos'][0], node_data['pos'][1]
                    pontos.append({
                        'id': node_id,
                        'label': node_data.get('label', node_id),
                        'x': x,
                        'y': y,
                        'z': 1.0  # Altura padrão
                    })
        
        print(f"✅ Carregados {len(pontos)} pontos do arquivo JSON")
        return pontos
        
    except FileNotFoundError:
        print(f"❌ Arquivo JSON não encontrado: {json_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar JSON: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar JSON: {e}")
        return []

def criar_bola_verde(node_id, x, y, z=1.0):
    """Cria o XML para uma bola verde no Gazebo"""
    return f'''
    <model name='bola_verde_{node_id.replace(".", "_").replace("::", "_")}'>
      <static>1</static>
      <pose frame='world'>{x:.3f} {y:.3f} {z:.1f} 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <emissive>0 0.5 0 1</emissive>
          </material>
        </visual>
        <collision name='collision'>
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
        </collision>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.1</ixx>
            <iyy>0.1</iyy>
            <izz>0.1</izz>
          </inertia>
        </inertial>
      </link>
    </model>'''

def criar_bola_azul(node_id, x, y, z=1.0):
    """Cria o XML para uma bola azul no Gazebo"""
    return f'''
    <model name='bola_azul_{node_id.replace(".", "_").replace("::", "_")}'>
      <static>1</static>
      <pose frame='world'>{x:.3f} {y:.3f} {z:.1f} 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            <sphere>
              <radius>0.8</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <emissive>0 0 0.5 1</emissive>
          </material>
        </visual>
        <collision name='collision'>
          <geometry>
            <sphere>
              <radius>0.8</radius>
            </sphere>
          </geometry>
        </collision>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.1</ixx>
            <iyy>0.1</iyy>
            <izz>0.1</izz>
          </inertia>
        </inertial>
      </link>
    </model>'''

def perguntar_tipo_bolas():
    """Pergunta ao usuário que tipo de bolas adicionar"""
    print("\n🎨 Escolha o tipo de bolas para adicionar ao mundo:")
    print("1. 🟢 Bolas Verdes")
    print("2. 🔵 Bolas Azuis") 
    print("3. 🟢🔵 Ambas (Verdes e Azuis)")
    print("4. ❌ Cancelar")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha (1-4): ").strip()
            
            if escolha == '1':
                return 'verdes'
            elif escolha == '2':
                return 'azuis'
            elif escolha == '3':
                return 'ambas'
            elif escolha == '4':
                print("❌ Operação cancelada pelo usuário.")
                return None
            else:
                print("❌ Opção inválida! Digite 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada pelo usuário.")
            return None
        except Exception as e:
            print(f"❌ Erro na entrada: {e}")

def criar_mundo_modificado():
    """Cria um mundo Gazebo modificado com bolas baseado no CSV do Gazebo2CSV.py e JSON graph_equipment.json"""
    
    print("🚀 Iniciando criação de mundo Gazebo modificado...")
    
    # Obtém o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Sobe um nível para o diretório do projeto
    
    # Arquivos com caminhos absolutos
    csv_file = os.path.join(script_dir, "todos_pontos_gps.csv")
    json_file = os.path.join(project_root, "jsons", "graph_equipment.json")
    world_file_original = os.path.join(script_dir, "parnaibaiii_simple_v3.world")
    
    # Verifica se os arquivos existem
    if not os.path.exists(csv_file):
        print(f"❌ Erro: Arquivo {csv_file} não encontrado!")
        print("💡 Execute primeiro o Gazebo2CSV.py para gerar o arquivo CSV.")
        return None
    
    if not os.path.exists(json_file):
        print(f"❌ Erro: Arquivo {json_file} não encontrado!")
        return None
    
    if not os.path.exists(world_file_original):
        print(f"❌ Erro: Arquivo {world_file_original} não encontrado!")
        return None
    
    # Pergunta que tipo de bolas adicionar
    tipo_bolas = perguntar_tipo_bolas()
    if tipo_bolas is None:
        return None
    
    # Define o nome do arquivo de saída
    output_file = os.path.join(script_dir, "parnaibaiii_simple_v3_modificado.world")
    
    # Carrega os dados
    pontos_csv = []
    pontos_json = []
    
    # Carrega dados do CSV (para pontos azuis)
    if tipo_bolas in ['azuis', 'ambas']:
        print("📊 Carregando dados do CSV...")
        df = pd.read_csv(csv_file)
        print(f"✅ {len(df)} objetos carregados do CSV")
        
        # Coordenadas de referência do mundo Gazebo
        lat_ref = -3.123199
        lon_ref = -41.764537
        
        # Aplica rotação do modelo ARGO_PARNAIBAIII_V3 (-90 graus) às coordenadas do CSV
        print("🌍 Aplicando rotação do modelo às coordenadas do CSV...")
        angle = -1.570796  # -90 graus em radianos
        
        for _, row in df.iterrows():
            # Coordenadas locais do modelo
            x_local = row['Px']  # Usa Px e Py do novo formato
            y_local = row['Py']
            
            # Aplica rotação para obter coordenadas no mundo
            x_world = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_world = x_local * math.sin(angle) + y_local * math.cos(angle)
            
            # Ajusta altura Z baseado no tipo de objeto
            z_base = 1.0  # Altura padrão
            if 'REATOR' in row['Model Name']:
                z_base += 2.0  # +2m adicional para reatores
            
            pontos_csv.append({
                'id': row['ID'],
                'label': row['Model Name'],
                'x': x_world,  # Coordenada X após rotação
                'y': y_world,  # Coordenada Y após rotação
                'z': z_base,  # Altura ajustada
                'tipo': 'azul'
            })
        
        print(f"✅ {len(pontos_csv)} pontos do CSV convertidos")
    
    # Carrega dados do JSON (para pontos verdes)
    if tipo_bolas in ['verdes', 'ambas']:
        print("📊 Carregando dados do JSON...")
        pontos_json_raw = carregar_pontos_json(json_file)
        
        if pontos_json_raw:
            # Aplica a mesma transformação de coordenadas para pontos verdes
            print("🌍 Aplicando transformação de coordenadas aos pontos do JSON...")
            for ponto in pontos_json_raw:
                # Aplica a mesma transformação: X_verde = yJSON, Y_verde = -xJSON
                x_verde = ponto['y']
                y_verde = -ponto['x']
                
                pontos_json.append({
                    'id': ponto['id'],
                    'label': ponto['label'],
                    'x': x_verde,
                    'y': y_verde,
                    'z': ponto['z'],
                    'tipo': 'verde'
                })
            
            print(f"✅ {len(pontos_json)} pontos do JSON convertidos")
    
    # Carrega o arquivo do mundo original (cria uma cópia)
    print("📖 Carregando arquivo do mundo original...")
    tree = ET.parse(world_file_original)
    root = tree.getroot()
    
    print(f"✅ Mundo original carregado: {world_file_original}")
    print(f"📋 Criando cópia modificada: {output_file}")
    
    # Encontra o elemento <world>
    world = root.find('world')
    if world is None:
        print("❌ Erro: Elemento <world> não encontrado!")
        return None
    
    # Adiciona as bolas conforme escolha do usuário
    bolas_adicionadas = 0
    
    if tipo_bolas in ['verdes', 'ambas']:
        print("🟢 Adicionando bolas verdes...")
        for i, ponto in enumerate(pontos_json):
            # Cria o XML da bola verde
            bola_xml = criar_bola_verde(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            
            # Adiciona ao mundo
            bola_element = ET.fromstring(bola_xml)
            world.append(bola_element)
            bolas_adicionadas += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processados: {i + 1}/{len(pontos_json)} pontos verdes")
    
    if tipo_bolas in ['azuis', 'ambas']:
        print("🔵 Adicionando bolas azuis...")
        for i, ponto in enumerate(pontos_csv):
            # Cria o XML da bola azul
            bola_xml = criar_bola_azul(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            
            # Adiciona ao mundo
            bola_element = ET.fromstring(bola_xml)
            world.append(bola_element)
            bolas_adicionadas += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processados: {i + 1}/{len(pontos_csv)} pontos azuis")
    
    # Salva a cópia modificada do mundo (sem alterar o original)
    print(f"💾 Salvando cópia modificada: {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS:")
    print(f"Total de bolas adicionadas: {bolas_adicionadas}")
    
    # Combina todos os pontos para estatísticas
    todos_pontos = pontos_csv + pontos_json
    
    if todos_pontos:
        x_coords = [p['x'] for p in todos_pontos]
        y_coords = [p['y'] for p in todos_pontos]
        z_coords = [p['z'] for p in todos_pontos]
        print(f"Coordenadas Gazebo:")
        print(f"  X: {min(x_coords):.2f} a {max(x_coords):.2f}m")
        print(f"  Y: {min(y_coords):.2f} a {max(y_coords):.2f}m")
        print(f"  Z: {min(z_coords):.2f} a {max(z_coords):.2f}m")
        
        # Estatísticas por tipo de bola
        verdes_count = len(pontos_json)
        azuis_count = len(pontos_csv)
        print(f"\n🎨 Tipos de bolas:")
        if verdes_count > 0:
            print(f"  🟢 Verdes: {verdes_count} bolas (do JSON)")
        if azuis_count > 0:
            print(f"  🔵 Azuis: {azuis_count} bolas (do CSV)")
    
    # Análise por tipo de equipamento (apenas para pontos azuis do CSV)
    if pontos_csv:
        tipos_equipamentos = {}
        for ponto in pontos_csv:
            nome = ponto['label']
            # Extrai o tipo do nome (ex: TPC9 -> TPC, REATOR1 -> REATOR)
            tipo = ''.join([c for c in nome if not c.isdigit() and c not in ['.', '_', '-']])
            if tipo and len(tipo) > 1:
                if tipo not in tipos_equipamentos:
                    tipos_equipamentos[tipo] = 0
                tipos_equipamentos[tipo] += 1
        
        print(f"\n🏷️ Tipos de equipamentos (bolas azuis):")
        for tipo, count in sorted(tipos_equipamentos.items()):
            print(f"  {tipo}: {count} equipamentos")
    
    print(f"\n✅ Cópia modificada salva em: {output_file}")
    print(f"📁 Tamanho do arquivo: {os.path.getsize(output_file):,} bytes")
    print(f"🔒 Arquivo original preservado: {world_file_original}")
    
    return output_file

def main():
    """Função principal"""
    print("🎯 CSV2World - Criador de Mundo Gazebo Modificado")
    print("=" * 50)
    
    arquivo_gerado = criar_mundo_modificado()
    
    if arquivo_gerado:
        print("\n🎉 Processo concluído com sucesso!")
        print("🚀 Para usar a cópia modificada:")
        print(f"   gazebo {arquivo_gerado}")
        print("🔒 O arquivo original permanece inalterado!")
        print("\n💡 Dicas:")
        print("  - 🟢 Bolas verdes: pontos do arquivo JSON graph_equipment.json")
        print("  - 🔵 Bolas azuis: equipamentos do arquivo CSV todos_pontos_gps.csv")
        print("  - Use o mundo modificado para visualização e planejamento")
        print("  - O arquivo original parnaibaiii_simple_v3.world não foi alterado")
    else:
        print("\n❌ Processo falhou ou foi cancelado!")

if __name__ == "__main__":
    main()
