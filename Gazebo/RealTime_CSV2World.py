#!/usr/bin/env python3
"""
RealTime_CSV2World.py - Versão otimizada e leve para adicionar bolas em tempo real ao Gazebo
Funcionalidades:
- Adiciona bolas ao mundo Gazebo aberto via SDF
- Modelos leves sem colisão para melhor performance
- Interface simples para escolher tipo de bolas
- Carregamento otimizado de dados
"""

import pandas as pd
import json
import math
import os
import sys
import subprocess
import time
from pathlib import Path

def gps_to_gazebo_coords(lat, lon, lat_ref, lon_ref):
    """Converte coordenadas GPS para coordenadas locais do Gazebo"""
    x = (lon - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
    y = (lat - lat_ref) * 111132.0
    return x, y

def carregar_pontos_json(json_file):
    """Carrega pontos do arquivo JSON graph_equipment.json de forma otimizada"""
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
                        'z': 0.5  # Altura reduzida para melhor visualização
                    })
        
        print(f"✅ Carregados {len(pontos)} pontos do JSON")
        return pontos
        
    except Exception as e:
        print(f"❌ Erro ao carregar JSON: {e}")
        return []

def carregar_pontos_csv(csv_file):
    """Carrega pontos do CSV de forma otimizada (coordenadas já corrigidas)"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ {len(df)} objetos carregados do CSV")
        
        pontos = []
        
        for _, row in df.iterrows():
            # Coordenadas já estão corrigidas no CSV (não precisa aplicar rotação)
            x_world = row['Px']
            y_world = row['Py']
            
            # Altura baseada na altitude do CSV (altitude - 77)
            z_base = row['Altitude'] - 75
            
            pontos.append({
                'id': row['ID'],
                'label': row['Model Name'],
                'x': x_world,  # Coordenada já corrigida
                'y': y_world,  # Coordenada já corrigida
                'z': z_base,   # Altura = altitude - 77
                'tipo': 'azul'
            })
        
        print(f"✅ {len(pontos)} pontos do CSV carregados (coordenadas já corrigidas)")
        return pontos
        
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return []

def criar_bola_leve_verde(node_id, x, y, z=0.5):
    """Cria modelo SDF leve para bola verde (sem colisão)"""
    # Sanitiza o nome do modelo
    safe_name = node_id.replace(".", "_").replace("::", "_").replace("-", "_")
    
    return f'''<?xml version="1.0"?>
<sdf version="1.4">
  <model name="bola_verde_{safe_name}">
    <static>1</static>
    <pose frame="world">{x:.3f} {y:.3f} {z:.1f} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.3</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0 0.8 0 1</ambient>
          <diffuse>0 0.8 0 1</diffuse>
          <emissive>0 0.3 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''

def criar_bola_leve_azul(node_id, x, y, z=0.5):
    """Cria modelo SDF leve para bola azul (sem colisão)"""
    # Sanitiza o nome do modelo
    safe_name = node_id.replace(".", "_").replace("::", "_").replace("-", "_")
    
    return f'''<?xml version="1.0"?>
<sdf version="1.4">
  <model name="bola_azul_{safe_name}">
    <static>1</static>
    <pose frame="world">{x:.3f} {y:.3f} {z:.1f} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.4</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0 0 0.8 1</ambient>
          <diffuse>0 0 0.8 1</diffuse>
          <emissive>0 0 0.3 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''

def perguntar_acao():
    """Interface para escolher ação (adicionar ou deletar bolas)"""
    print("\n🎯 Escolha uma ação:")
    print("1. ➕ Adicionar Bolas")
    print("2. 🗑️  Deletar Todas as Bolas")
    print("3. ❌ Cancelar")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha (1-3): ").strip()
            
            if escolha == '1':
                return 'adicionar'
            elif escolha == '2':
                return 'deletar'
            elif escolha == '3':
                print("❌ Operação cancelada.")
                return None
            else:
                print("❌ Opção inválida! Digite 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada.")
            return None

def perguntar_tipo_bolas():
    """Interface simples para escolher tipo de bolas"""
    print("\n🎨 Escolha o tipo de bolas para adicionar:")
    print("1. 🟢 Bolas Verdes (pontos do JSON)")
    print("2. 🔵 Bolas Azuis (equipamentos do CSV)")
    print("3. 🟢🔵 Ambas")
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
                print("❌ Operação cancelada.")
                return None
            else:
                print("❌ Opção inválida! Digite 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada.")
            return None

def get_equipment_groups():
    """Define os grupos de equipamentos disponíveis"""
    return {
        'REATOR': 'Reatores',
        'SVC': 'SVC (Seccionadores)',
        'TPC': 'TPC (Transformadores de Potência)',
        'TC': 'TC (Transformadores de Corrente)',
        'SECV': 'SECV (Seccionadores Verticais)',
        'SECH': 'SECH (Seccionadores Horizontais)',
        'IP': 'IP (Interruptores de Potência)',
        'DISJUNTOR': 'Disjuntores',
        'BUSIP': 'BUSIP (Barramentos de Potência)',
        'BUSCSB': 'BUSCSB (Barramentos de Controle)',
        'PR': 'PR (Protetores)'
    }

def perguntar_grupos_equipamentos():
    """Pergunta se quer todos os equipamentos ou por grupos específicos"""
    print("\n🔧 Escolha como adicionar bolas azuis:")
    print("1. 📋 Todos os equipamentos")
    print("2. 🎯 Por grupos específicos")
    print("3. ❌ Cancelar")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha (1-3): ").strip()
            
            if escolha == '1':
                return 'todos'
            elif escolha == '2':
                return 'grupos'
            elif escolha == '3':
                print("❌ Operação cancelada.")
                return None
            else:
                print("❌ Opção inválida! Digite 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada.")
            return None

def perguntar_grupos_especificos():
    """Interface para escolher grupos específicos de equipamentos"""
    grupos = get_equipment_groups()
    
    print("\n🎯 Escolha os grupos de equipamentos:")
    print("Digite os números separados por vírgula (ex: 1,2,3,4)")
    print("0 = Todos os grupos")
    print()
    
    # Lista os grupos disponíveis
    for i, (codigo, nome) in enumerate(grupos.items(), 1):
        print(f"{i:2d}. {nome} ({codigo})")
    
    print(" 0. Todos os grupos")
    print("99. ❌ Cancelar")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha: ").strip()
            
            if escolha == '99':
                print("❌ Operação cancelada.")
                return None
            
            if escolha == '0':
                return list(grupos.keys())
            
            # Processa a entrada (ex: "1,2,3,4")
            try:
                numeros = [int(x.strip()) for x in escolha.split(',')]
                grupos_escolhidos = []
                
                for num in numeros:
                    if 1 <= num <= len(grupos):
                        codigo_grupo = list(grupos.keys())[num - 1]
                        grupos_escolhidos.append(codigo_grupo)
                    else:
                        print(f"❌ Número inválido: {num}")
                        break
                else:
                    if grupos_escolhidos:
                        print(f"✅ Grupos selecionados: {', '.join([grupos[cod] for cod in grupos_escolhidos])}")
                        return grupos_escolhidos
                    else:
                        print("❌ Nenhum grupo válido selecionado!")
                        
            except ValueError:
                print("❌ Formato inválido! Use números separados por vírgula (ex: 1,2,3,4)")
                
        except KeyboardInterrupt:
            print("\n❌ Operação cancelada.")
            return None

def filtrar_equipamentos_por_grupo(pontos_csv, grupos_escolhidos):
    """Filtra equipamentos baseado nos grupos selecionados"""
    if not grupos_escolhidos:
        return pontos_csv
    
    pontos_filtrados = []
    for ponto in pontos_csv:
        nome_equipamento = ponto['label']
        
        # Verifica se o equipamento pertence a algum dos grupos selecionados
        for grupo in grupos_escolhidos:
            if grupo in nome_equipamento:
                pontos_filtrados.append(ponto)
                break
    
    return pontos_filtrados

def verificar_gazebo_rodando():
    """Verifica se o Gazebo está rodando (ROS ou standalone)"""
    try:
        # Verifica processos do Gazebo
        result = subprocess.run(['pgrep', '-f', 'gzserver|gzclient|gazebo'], 
                              capture_output=True, text=True)
        gazebo_running = len(result.stdout.strip()) > 0
        
        # Verifica se ROS está rodando
        result_ros = subprocess.run(['pgrep', '-f', 'roslaunch'], 
                                  capture_output=True, text=True)
        ros_running = len(result_ros.stdout.strip()) > 0
        
        return gazebo_running, ros_running
        
    except:
        return False, False

def adicionar_modelo_ros_gazebo(sdf_content, model_name):
    """Adiciona um modelo ao ROS Gazebo usando rosrun gazebo_ros spawn_model"""
    try:
        # Cria arquivo temporário
        temp_file = f"/tmp/temp_model_{model_name}.sdf"
        with open(temp_file, 'w') as f:
            f.write(sdf_content)
        
        # Adiciona ao Gazebo via rosrun
        cmd = ['rosrun', 'gazebo_ros', 'spawn_model', 
               '-file', temp_file, '-model', model_name, '-sdf']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Remove arquivo temporário
        os.remove(temp_file)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro ao adicionar modelo via ROS: {e}")
        return False

def adicionar_modelo_gazebo_standalone(sdf_content):
    """Adiciona um modelo ao Gazebo standalone usando gz model"""
    try:
        # Cria arquivo temporário
        temp_file = "/tmp/temp_model.sdf"
        with open(temp_file, 'w') as f:
            f.write(sdf_content)
        
        # Adiciona ao Gazebo
        result = subprocess.run(['gz', 'model', '-f', temp_file], 
                              capture_output=True, text=True, timeout=5)
        
        # Remove arquivo temporário
        os.remove(temp_file)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro ao adicionar modelo: {e}")
        return False

def listar_modelos_bolas():
    """Lista todos os modelos de bolas no mundo Gazebo usando rostopic"""
    try:
        # Usa rostopic para obter lista de modelos
        result = subprocess.run(['rostopic', 'echo', '/gazebo/model_states', '-n', '1'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            bolas = []
            
            # Procura por nomes de modelos que contêm 'bola_'
            import re
            # Extrai os nomes dos modelos da saída do rostopic
            # Procura por linhas que começam com "  - " após "name:"
            lines = output.split('\n')
            in_name_section = False
            for line in lines:
                if line.strip() == 'name:':
                    in_name_section = True
                    continue
                elif in_name_section and line.strip().startswith('- '):
                    name = line.strip()[2:].strip()  # Remove "- " do início
                    if 'bola_' in name:
                        bolas.append(name)
                elif in_name_section and not line.strip().startswith('- '):
                    break  # Sai da seção de nomes
            
            return bolas
        else:
            return []
            
    except Exception as e:
        print(f"❌ Erro ao listar modelos: {e}")
        return []

def deletar_modelo_ros_gazebo(model_name):
    """Deleta um modelo do ROS Gazebo usando rosservice"""
    try:
        cmd = ['rosservice', 'call', '/gazebo/delete_model', 
               f'{{model_name: "{model_name}"}}']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro ao deletar modelo via ROS: {e}")
        return False

def deletar_modelo_gazebo_standalone(model_name):
    """Deleta um modelo do Gazebo standalone usando gz model"""
    try:
        result = subprocess.run(['gz', 'model', '-d', model_name], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Erro ao deletar modelo: {e}")
        return False

def deletar_todas_bolas(usar_ros=True):
    """Deleta todas as bolas do mundo Gazebo"""
    print("🗑️  Procurando bolas no mundo Gazebo...")
    
    # Lista modelos de bolas usando gz model -l (funciona para ambos)
    bolas = listar_modelos_bolas()
    
    if not bolas:
        print("✅ Nenhuma bola encontrada no mundo.")
        return True
    
    print(f"🔍 Encontradas {len(bolas)} bolas para deletar:")
    for bola in bolas[:10]:  # Mostra apenas as primeiras 10
        print(f"  - {bola}")
    if len(bolas) > 10:
        print(f"  ... e mais {len(bolas) - 10} bolas")
    
    # Prossegue automaticamente sem confirmação
    print(f"🗑️ Deletando {len(bolas)} bolas automaticamente...")
    deletadas = 0
    erros = 0
    
    for i, bola in enumerate(bolas):
        if usar_ros:
            sucesso = deletar_modelo_ros_gazebo(bola)
        else:
            sucesso = deletar_modelo_gazebo_standalone(bola)
        
        if sucesso:
            deletadas += 1
        else:
            erros += 1
        
        # Progresso a cada 25 bolas
        if (i + 1) % 25 == 0:
            print(f"  Progresso: {i + 1}/{len(bolas)} bolas deletadas")
        
        # Pequena pausa
        time.sleep(0.01)
    
    # Resultados
    print(f"\n📊 RESULTADOS:")
    print(f"✅ Bolas deletadas com sucesso: {deletadas}")
    if erros > 0:
        print(f"❌ Erros: {erros}")
    
    return deletadas > 0

def gerenciar_bolas_tempo_real():
    """Gerencia bolas no mundo Gazebo em tempo real (adicionar/deletar)"""
    
    print("🚀 RealTime_CSV2World - Gerenciador de Bolas em Tempo Real")
    print("=" * 60)
    
    # Verifica se o Gazebo está rodando
    gazebo_running, ros_running = verificar_gazebo_rodando()
    
    if not gazebo_running:
        print("❌ Gazebo não está rodando!")
        print("💡 Inicie o Gazebo primeiro:")
        print("   ./world.sh 1    # Para ROS Gazebo")
        print("   ou")
        print("   gazebo parnaibaiii_simple_v3.world    # Para Gazebo standalone")
        return False
    
    if ros_running:
        print("✅ ROS Gazebo detectado rodando!")
        usar_ros = True
    else:
        print("✅ Gazebo standalone detectado rodando!")
        usar_ros = False
    
    # Pergunta ação
    acao = perguntar_acao()
    if acao is None:
        return False
    
    # Se for deletar, executa e termina
    if acao == 'deletar':
        return deletar_todas_bolas(usar_ros)
    
    # Se for adicionar, continua com o processo normal
    # Obtém caminhos dos arquivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    csv_file = os.path.join(script_dir, "todos_pontos_gps.csv")
    json_file = os.path.join(project_root, "jsons", "graph_equipment.json")
    
    # Verifica arquivos
    if not os.path.exists(csv_file):
        print(f"❌ Arquivo CSV não encontrado: {csv_file}")
        return False
    
    if not os.path.exists(json_file):
        print(f"❌ Arquivo JSON não encontrado: {json_file}")
        return False
    
    # Pergunta tipo de bolas
    tipo_bolas = perguntar_tipo_bolas()
    if tipo_bolas is None:
        return False
    
    # Carrega dados
    pontos_csv = []
    pontos_json = []
    
    if tipo_bolas in ['azuis', 'ambas']:
        pontos_csv_raw = carregar_pontos_csv(csv_file)
        
        # Pergunta se quer todos os equipamentos ou por grupos
        if pontos_csv_raw:
            modo_grupos = perguntar_grupos_equipamentos()
            if modo_grupos is None:
                return False
            
            if modo_grupos == 'todos':
                pontos_csv = pontos_csv_raw
                print("✅ Todos os equipamentos selecionados")
            elif modo_grupos == 'grupos':
                grupos_escolhidos = perguntar_grupos_especificos()
                if grupos_escolhidos is None:
                    return False
                
                pontos_csv = filtrar_equipamentos_por_grupo(pontos_csv_raw, grupos_escolhidos)
                print(f"✅ Filtrados {len(pontos_csv)} equipamentos dos grupos selecionados")
    
    if tipo_bolas in ['verdes', 'ambas']:
        pontos_json_raw = carregar_pontos_json(json_file)
        # Aplica transformação para pontos verdes
        for ponto in pontos_json_raw:
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
    
    # Confirmação
    total_bolas = len(pontos_csv) + len(pontos_json)
    print(f"\n📊 Total de bolas a adicionar: {total_bolas}")
    
    if total_bolas == 0:
        print("❌ Nenhuma bola para adicionar!")
        return False
    
    # Mostra estatísticas por tipo
    if pontos_csv:
        print(f"🔵 Bolas azuis: {len(pontos_csv)} equipamentos")
    if pontos_json:
        print(f"🟢 Bolas verdes: {len(pontos_json)} pontos")
    
    # Prossegue automaticamente sem confirmação
    print(f"🚀 Adicionando {total_bolas} bolas ao mundo automaticamente...")
    
    # Adiciona bolas
    print(f"\n🎯 Adicionando bolas ao mundo Gazebo...")
    
    bolas_adicionadas = 0
    erros = 0
    
    # Adiciona bolas verdes
    if tipo_bolas in ['verdes', 'ambas'] and pontos_json:
        print("🟢 Adicionando bolas verdes...")
        for i, ponto in enumerate(pontos_json):
            sdf_content = criar_bola_leve_verde(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            model_name = f"bola_verde_{ponto['id'].replace('.', '_').replace('::', '_').replace('-', '_')}"
            
            if usar_ros:
                sucesso = adicionar_modelo_ros_gazebo(sdf_content, model_name)
            else:
                sucesso = adicionar_modelo_gazebo_standalone(sdf_content)
            
            if sucesso:
                bolas_adicionadas += 1
            else:
                erros += 1
            
            # Progresso a cada 25 bolas (mais frequente)
            if (i + 1) % 25 == 0:
                print(f"  Progresso: {i + 1}/{len(pontos_json)} bolas verdes")
            
            # Pausa menor para acelerar
            time.sleep(0.005)
    
    # Adiciona bolas azuis
    if tipo_bolas in ['azuis', 'ambas'] and pontos_csv:
        print("🔵 Adicionando bolas azuis...")
        for i, ponto in enumerate(pontos_csv):
            sdf_content = criar_bola_leve_azul(ponto['label'], ponto['x'], ponto['y'], ponto['z'])
            model_name = f"bola_azul_{ponto['id'].replace('.', '_').replace('::', '_').replace('-', '_')}"
            
            if usar_ros:
                sucesso = adicionar_modelo_ros_gazebo(sdf_content, model_name)
            else:
                sucesso = adicionar_modelo_gazebo_standalone(sdf_content)
            
            if sucesso:
                bolas_adicionadas += 1
            else:
                erros += 1
            
            # Progresso a cada 25 bolas (mais frequente)
            if (i + 1) % 25 == 0:
                print(f"  Progresso: {i + 1}/{len(pontos_csv)} bolas azuis")
            
            # Pausa menor para acelerar
            time.sleep(0.005)
    
    # Resultados
    print(f"\n📊 RESULTADOS:")
    print(f"✅ Bolas adicionadas com sucesso: {bolas_adicionadas}")
    if erros > 0:
        print(f"❌ Erros: {erros}")
    
    # Estatísticas
    if pontos_json:
        print(f"🟢 Bolas verdes: {len(pontos_json)}")
    if pontos_csv:
        print(f"🔵 Bolas azuis: {len(pontos_csv)}")
    
    print(f"\n🎉 Processo concluído!")
    print("💡 As bolas foram adicionadas ao mundo Gazebo em tempo real.")
    print("🔧 Para remover bolas, use: gz model -d <nome_do_modelo>")
    
    return True

def main():
    """Função principal"""
    try:
        sucesso = gerenciar_bolas_tempo_real()
        if not sucesso:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Operação interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

