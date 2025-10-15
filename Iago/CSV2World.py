#!/usr/bin/env python3
"""
Script para criar um mundo Gazebo modificado com bolas baseado na saída do Gazebo2CSV.py
Combina funcionalidades dos scripts adicionar_bolas_gazebo.py e adicionar_bolas_azuis_gazebo.py
"""

import pandas as pd
import xml.etree.ElementTree as ET
import math
import os
import sys

def gps_to_gazebo_coords(lat, lon, lat_ref, lon_ref):
    """Converte coordenadas GPS para coordenadas locais do Gazebo"""
    x = (lon - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
    y = (lat - lat_ref) * 111132.0
    return x, y

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
    """Cria um mundo Gazebo modificado com bolas basseriuneado no CSV do Gazebo2CSV.py"""
    
    print("🚀 Iniciando criação de mundo Gazebo modificado...")
    
    # Arquivos
    csv_file = "todos_pontos_gps.csv"
    world_file_original = "parnaibaiii_simple_v3.world"
    
    # Verifica se os arquivos existem
    if not os.path.exists(csv_file):
        print(f"❌ Erro: Arquivo {csv_file} não encontrado!")
        print("💡 Execute primeiro o Gazebo2CSV.py para gerar o arquivo CSV.")
        return None
    
    if not os.path.exists(world_file_original):
        print(f"❌ Erro: Arquivo {world_file_original} não encontrado!")
        return None
    
    # Pergunta que tipo de bolas adicionar
    tipo_bolas = perguntar_tipo_bolas()
    if tipo_bolas is None:
        return None
    
    # Define o nome do arquivo de saída
    output_file = "parnaibaiii_simple_v3_modificado.world"
    
    # Carrega o CSV
    print("📊 Carregando dados do CSV...")
    df = pd.read_csv(csv_file)
    print(f"✅ {len(df)} objetos carregados do CSV")
    
    # Coordenadas de referência do mundo Gazebo
    lat_ref = -3.123199
    lon_ref = -41.764537
    
    # Aplica rotação do modelo ARGO_PARNAIBAIII_V3 (-90 graus) às coordenadas do CSV
    print("🌍 Aplicando rotação do modelo às coordenadas do CSV...")
    angle = -1.570796  # -90 graus em radianos
    pontos_gazebo = []
    for _, row in df.iterrows():
        # Coordenadas locais do modelo
        x_local = row['x_cartesiano']
        y_local = row['y_cartesiano']
        
        # Aplica rotação para obter coordenadas no mundo
        x_world = x_local * math.cos(angle) - y_local * math.sin(angle)
        y_world = x_local * math.sin(angle) + y_local * math.cos(angle)
        
        # Ajusta altura Z baseado no tipo de objeto
        z_base = row['z_cartesiano'] + 1.0  # Altura padrão + 1m
        if 'REATOR' in row['label']:
            z_base += 2.0  # +2m adicional para reatores
        
        pontos_gazebo.append({
            'id': row['id'],
            'label': row['label'],
            'tipo': row['tipo'],
            'x': x_world,  # Coordenada X após rotação
            'y': y_world,  # Coordenada Y após rotação
            'z': z_base,  # Altura ajustada
            'latitude': row['latitude'],
            'longitude': row['longitude']
        })
    
    print(f"✅ {len(pontos_gazebo)} pontos convertidos")
    
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
        for i, ponto in enumerate(pontos_gazebo):
            # Cria o XML da bola verde
            bola_xml = criar_bola_verde(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            
            # Adiciona ao mundo
            bola_element = ET.fromstring(bola_xml)
            world.append(bola_element)
            bolas_adicionadas += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processados: {i + 1}/{len(pontos_gazebo)} pontos verdes")
    
    if tipo_bolas in ['azuis', 'ambas']:
        print("🔵 Adicionando bolas azuis...")
        for i, ponto in enumerate(pontos_gazebo):
            # Cria o XML da bola azul
            bola_xml = criar_bola_azul(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            
            # Adiciona ao mundo
            bola_element = ET.fromstring(bola_xml)
            world.append(bola_element)
            bolas_adicionadas += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processados: {i + 1}/{len(pontos_gazebo)} pontos azuis")
    
    # Salva a cópia modificada do mundo (sem alterar o original)
    print(f"💾 Salvando cópia modificada: {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS:")
    print(f"Total de bolas adicionadas: {bolas_adicionadas}")
    x_coords = [p['x'] for p in pontos_gazebo]
    y_coords = [p['y'] for p in pontos_gazebo]
    z_coords = [p['z'] for p in pontos_gazebo]
    print(f"Coordenadas Gazebo:")
    print(f"  X: {min(x_coords):.2f} a {max(x_coords):.2f}m")
    print(f"  Y: {min(y_coords):.2f} a {max(y_coords):.2f}m")
    print(f"  Z: {min(z_coords):.2f} a {max(z_coords):.2f}m")
    
    # Análise por tipo de equipamento
    tipos_equipamentos = {}
    for ponto in pontos_gazebo:
        nome = ponto['label']
        # Extrai o tipo do nome (ex: TPC9 -> TPC, REATOR1 -> REATOR)
        tipo = ''.join([c for c in nome if not c.isdigit() and c not in ['.', '_', '-']])
        if tipo and len(tipo) > 1:
            if tipo not in tipos_equipamentos:
                tipos_equipamentos[tipo] = 0
            tipos_equipamentos[tipo] += 1
    
    print(f"\n🏷️ Tipos de equipamentos:")
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
        print("  - As bolas verdes/azuis representam os equipamentos industriais")
        print("  - Use o mundo modificado para visualização e planejamento")
        print("  - O arquivo original parnaibaiii_simple_v3.world não foi alterado")
    else:
        print("\n❌ Processo falhou ou foi cancelado!")

if __name__ == "__main__":
    main()
