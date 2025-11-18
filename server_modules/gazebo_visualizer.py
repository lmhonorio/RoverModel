"""
Módulo para visualizar waypoints de missões no Gazebo em tempo real
Cores: Verde (rover1), Azul (rover2), Vermelho (rover3), Amarelo (rover4), etc.
"""

import subprocess
import os
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cores para diferentes robôs (RGB normalized 0-1)
# Suporta nomes: R1, R2, R3, etc. e rover1, rover2, rover3, etc.
ROBOT_COLORS = {
    'R1': {'name': 'verde', 'ambient': '0 0.8 0', 'diffuse': '0 0.8 0', 'emissive': '0 0.3 0'},
    'R2': {'name': 'azul', 'ambient': '0 0 0.8', 'diffuse': '0 0 0.8', 'emissive': '0 0 0.3'},
    'R3': {'name': 'vermelho', 'ambient': '0.8 0 0', 'diffuse': '0.8 0 0', 'emissive': '0.3 0 0'},
    'R4': {'name': 'amarelo', 'ambient': '0.8 0.8 0', 'diffuse': '0.8 0.8 0', 'emissive': '0.3 0.3 0'},
    'R5': {'name': 'magenta', 'ambient': '0.8 0 0.8', 'diffuse': '0.8 0 0.8', 'emissive': '0.3 0 0.3'},
    'R6': {'name': 'ciano', 'ambient': '0 0.8 0.8', 'diffuse': '0 0.8 0.8', 'emissive': '0 0.3 0.3'},
    # Aliases
    'rover1': {'name': 'verde', 'ambient': '0 0.8 0', 'diffuse': '0 0.8 0', 'emissive': '0 0.3 0'},
    'rover2': {'name': 'azul', 'ambient': '0 0 0.8', 'diffuse': '0 0 0.8', 'emissive': '0 0 0.3'},
    'rover3': {'name': 'vermelho', 'ambient': '0.8 0 0', 'diffuse': '0.8 0 0', 'emissive': '0.3 0 0'},
    'rover4': {'name': 'amarelo', 'ambient': '0.8 0.8 0', 'diffuse': '0.8 0.8 0', 'emissive': '0.3 0.3 0'},
    'rover5': {'name': 'magenta', 'ambient': '0.8 0 0.8', 'diffuse': '0.8 0 0.8', 'emissive': '0.3 0 0.3'},
    'rover6': {'name': 'ciano', 'ambient': '0 0.8 0.8', 'diffuse': '0 0.8 0.8', 'emissive': '0 0.3 0.3'},
}

# Referência GPS para conversão
# IMPORTANTE: Deve estar próximo das coordenadas dos waypoints!
# Waypoints típicos: lat=-3.123, lon=-41.764
LAT_REF = -3.123  # Latitude de referência (centro aproximado da área)
LON_REF = -41.764  # Longitude de referência (centro aproximado da área)

def gps_to_gazebo_coords(lat, lon):
    """Converte coordenadas GPS para coordenadas locais do Gazebo"""
    x = (lon - LON_REF) * 111320.0 * math.cos(math.radians(LAT_REF))
    y = (lat - LAT_REF) * 111132.0
    return x, y

def criar_bola_waypoint(waypoint_id, x, y, z, color_config, radius=0.35):
    """
    Cria modelo SDF para bola de waypoint com cor específica
    
    Args:
        waypoint_id: ID único do waypoint
        x, y, z: Coordenadas no Gazebo
        color_config: Dict com 'ambient', 'diffuse', 'emissive'
        radius: Raio da bola (padrão 0.35m)
    """
    # Sanitiza o nome do modelo
    safe_name = str(waypoint_id).replace(".", "_").replace("::", "_").replace("-", "_").replace(" ", "_")
    model_name = f"waypoint_{safe_name}"
    
    return model_name, f'''<?xml version="1.0"?>
<sdf version="1.4">
  <model name="{model_name}">
    <static>1</static>
    <pose frame="world">{x:.3f} {y:.3f} {z:.1f} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color_config['ambient']} 1</ambient>
          <diffuse>{color_config['diffuse']} 1</diffuse>
          <emissive>{color_config['emissive']} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''

def criar_linha_waypoints(wp1_id, wp2_id, x1, y1, x2, y2, z, color_config, radius=0.08):
    """
    Cria modelo SDF para linha conectando dois waypoints consecutivos
    
    Args:
        wp1_id, wp2_id: IDs dos waypoints
        x1, y1, x2, y2: Coordenadas dos waypoints
        z: Altura
        color_config: Dict com cores
        radius: Raio do cilindro (padrão 0.08m)
    """
    safe_name1 = str(wp1_id).replace(".", "_").replace("::", "_").replace("-", "_").replace(" ", "_")
    safe_name2 = str(wp2_id).replace(".", "_").replace("::", "_").replace("-", "_").replace(" ", "_")
    model_name = f"linha_{safe_name1}_{safe_name2}"
    
    # Calcula ponto médio e distância
    x_medio = (x1 + x2) / 2
    y_medio = (y1 + y2) / 2
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calcula ângulo de rotação no plano XY (yaw)
    angulo_yaw = math.atan2(y2 - y1, x2 - x1)
    
    # Rotação: pitch = 90 graus para deitar o cilindro
    pitch_rad = math.pi / 2
    
    return model_name, f'''<?xml version="1.0"?>
<sdf version="1.4">
  <model name="{model_name}">
    <static>1</static>
    <pose frame="world">{x_medio:.3f} {y_medio:.3f} {z:.1f} 0 {pitch_rad:.3f} {angulo_yaw:.3f}</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{distancia:.3f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color_config['ambient']} 0.6</ambient>
          <diffuse>{color_config['diffuse']} 0.6</diffuse>
          <emissive>{color_config['emissive']} 0.6</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''

def verificar_gazebo_rodando():
    """Verifica se o Gazebo está rodando (ROS ou standalone)"""
    try:
        result = subprocess.run(['pgrep', '-f', 'gzserver|gzclient|gazebo'], 
                              capture_output=True, text=True)
        gazebo_running = len(result.stdout.strip()) > 0
        
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
        
        # Preparar ambiente ROS para subprocess
        # CRÍTICO: subprocess.run não herda ambiente ROS automaticamente!
        env = os.environ.copy()
        
        # Se ROS não estiver no ambiente, tentar configurar
        if 'ROS_DISTRO' not in env:
            # Adicionar caminhos ROS Noetic ao ambiente
            env['ROS_DISTRO'] = 'noetic'
            env['ROS_VERSION'] = '1'
            env['ROS_PYTHON_VERSION'] = '3'
            
            # Adicionar PYTHONPATH do ROS
            ros_python_path = '/opt/ros/noetic/lib/python3/dist-packages'
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{ros_python_path}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = ros_python_path
            
            # Adicionar PATH do ROS
            ros_bin_path = '/opt/ros/noetic/bin'
            if 'PATH' in env:
                env['PATH'] = f"{ros_bin_path}:{env['PATH']}"
            else:
                env['PATH'] = ros_bin_path
            
            # Adicionar LD_LIBRARY_PATH do ROS
            ros_lib_path = '/opt/ros/noetic/lib'
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{ros_lib_path}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = ros_lib_path
            
            # Adicionar CMAKE_PREFIX_PATH
            env['CMAKE_PREFIX_PATH'] = '/opt/ros/noetic'
            env['ROS_PACKAGE_PATH'] = '/opt/ros/noetic/share'
        
        # Adiciona ao Gazebo via rosrun COM AMBIENTE ROS
        cmd = ['rosrun', 'gazebo_ros', 'spawn_model', 
               '-file', temp_file, '-model', model_name, '-sdf']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
        
        # Remove arquivo temporário
        try:
            os.remove(temp_file)
        except:
            pass
        
        # DEBUG: mostrar primeiro erro
        if result.returncode != 0:
            if not hasattr(adicionar_modelo_ros_gazebo, '_primeiro_erro'):
                adicionar_modelo_ros_gazebo._primeiro_erro = True
                print(f"\n⚠️ DEBUG: Primeiro erro ao adicionar modelo:")
                print(f"   Modelo: {model_name}")
                print(f"   Comando: {' '.join(cmd)}")
                print(f"   Return code: {result.returncode}")
                print(f"   ROS_DISTRO no env: {env.get('ROS_DISTRO', 'NÃO DEFINIDO')}")
                print(f"   PYTHONPATH: {env.get('PYTHONPATH', 'NÃO DEFINIDO')[:100]}...")
                print(f"   STDOUT: {result.stdout[:300]}")
                print(f"   STDERR: {result.stderr[:300]}\n")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"⚠️ Exceção ao adicionar {model_name}: {e}")
        return False

def adicionar_modelo_gazebo_standalone(sdf_content):
    """Adiciona um modelo ao Gazebo standalone usando gz model"""
    try:
        temp_file = "/tmp/temp_waypoint.sdf"
        with open(temp_file, 'w') as f:
            f.write(sdf_content)
        
        result = subprocess.run(['gz', 'model', '-f', temp_file], 
                              capture_output=True, text=True, timeout=5)
        
        os.remove(temp_file)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Erro ao adicionar modelo: {e}")
        return False

def adicionar_waypoint_worker(args):
    """Worker para adicionar um waypoint em paralelo"""
    sdf_content, model_name, usar_ros = args
    
    if usar_ros:
        return adicionar_modelo_ros_gazebo(sdf_content, model_name)
    else:
        return adicionar_modelo_gazebo_standalone(sdf_content)

def listar_modelos_waypoints():
    """Lista todos os modelos de waypoints no mundo Gazebo"""
    try:
        # Preparar ambiente ROS para subprocess
        env = os.environ.copy()
        if 'ROS_DISTRO' not in env:
            env['ROS_DISTRO'] = 'noetic'
            env['PYTHONPATH'] = '/opt/ros/noetic/lib/python3/dist-packages'
            env['PATH'] = f"/opt/ros/noetic/bin:{env.get('PATH', '')}"
        
        result = subprocess.run(['rostopic', 'echo', '/gazebo/model_states', '-n', '1'], 
                              capture_output=True, text=True, timeout=10, env=env)
        
        if result.returncode == 0:
            output = result.stdout
            waypoints = []
            
            lines = output.split('\n')
            in_name_section = False
            for line in lines:
                if line.strip() == 'name:':
                    in_name_section = True
                    continue
                elif in_name_section and line.strip().startswith('- '):
                    name = line.strip()[2:].strip()
                    if 'waypoint_' in name or 'linha_' in name:
                        waypoints.append(name)
                elif in_name_section and not line.strip().startswith('- '):
                    break
            
            return waypoints
        else:
            return []
    except Exception as e:
        print(f"❌ Erro ao listar waypoints: {e}")
        return []

def deletar_modelo_ros_gazebo(model_name):
    """Deleta um modelo do ROS Gazebo usando rosservice"""
    try:
        # Preparar ambiente ROS para subprocess
        env = os.environ.copy()
        if 'ROS_DISTRO' not in env:
            env['ROS_DISTRO'] = 'noetic'
            env['PYTHONPATH'] = '/opt/ros/noetic/lib/python3/dist-packages'
            env['PATH'] = f"/opt/ros/noetic/bin:{env.get('PATH', '')}"
        
        cmd = ['rosservice', 'call', '/gazebo/delete_model', 
               f'{{model_name: "{model_name}"}}']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, env=env)
        return result.returncode == 0
        
    except Exception as e:
        return False

def deletar_modelo_gazebo_standalone(model_name):
    """Deleta um modelo do Gazebo standalone usando gz model"""
    try:
        result = subprocess.run(['gz', 'model', '-d', model_name], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception as e:
        return False

def deletar_waypoint_worker(args):
    """Worker para deletar um waypoint em paralelo"""
    model_name, usar_ros = args
    
    if usar_ros:
        return deletar_modelo_ros_gazebo(model_name)
    else:
        return deletar_modelo_gazebo_standalone(model_name)

def limpar_waypoints_antigos(usar_ros=True, max_workers=8):
    """
    Remove todos os waypoints antigos do Gazebo antes de adicionar novos
    
    Args:
        usar_ros: Se True, usa comandos ROS; senão, usa comandos standalone
        max_workers: Número de threads paralelas
        
    Returns:
        int: Número de waypoints deletados
    """
    print("🧹 Limpando waypoints antigos do Gazebo...")
    
    waypoints = listar_modelos_waypoints()
    
    if not waypoints:
        print("✅ Nenhum waypoint antigo encontrado")
        return 0
    
    print(f"🔍 Encontrados {len(waypoints)} waypoints antigos")
    
    deletados = 0
    erros = 0
    
    # Deletar em paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tarefas = [(wp, usar_ros) for wp in waypoints]
        futures = [executor.submit(deletar_waypoint_worker, tarefa) for tarefa in tarefas]
        
        for future in as_completed(futures):
            try:
                sucesso = future.result()
                if sucesso:
                    deletados += 1
                else:
                    erros += 1
            except Exception:
                erros += 1
    
    print(f"✅ {deletados} waypoints deletados")
    if erros > 0:
        print(f"⚠️ {erros} erros ao deletar")
    
    return deletados

def adicionar_waypoints_missao(waypoints_by_robot, z_altura=3.0, adicionar_linhas=True, max_workers=8):
    """
    Adiciona waypoints de uma missão no Gazebo com cores diferentes por robô
    
    Args:
        waypoints_by_robot: Dict {robot_name: [waypoints]}
        z_altura: Altura das bolas (padrão 3.0m)
        adicionar_linhas: Se True, adiciona linhas conectando waypoints consecutivos
        max_workers: Número de threads paralelas
        
    Returns:
        dict: Estatísticas da operação
    """
    print("\n🎯 VISUALIZADOR DE WAYPOINTS NO GAZEBO")
    print("=" * 60)
    
    # Verificar se Gazebo está rodando
    gazebo_running, ros_running = verificar_gazebo_rodando()
    
    if not gazebo_running:
        print("❌ Gazebo não está rodando!")
        return {
            'success': False,
            'message': 'Gazebo não está rodando',
            'waypoints_added': 0,
            'lines_added': 0
        }
    
    usar_ros = ros_running
    print(f"✅ {'ROS Gazebo' if usar_ros else 'Gazebo standalone'} detectado")
    
    # Limpar waypoints antigos
    limpar_waypoints_antigos(usar_ros, max_workers)
    
    # Contar total de waypoints
    total_waypoints = sum(len(waypoints) for waypoints in waypoints_by_robot.values())
    print(f"\n📊 Total de waypoints a adicionar: {total_waypoints}")
    
    waypoints_adicionados = 0
    linhas_adicionadas = 0
    erros = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for robot_name, waypoints in waypoints_by_robot.items():
            if not waypoints:
                continue
            
            # Obter configuração de cor para este robô
            color_config = ROBOT_COLORS.get(robot_name, ROBOT_COLORS['rover1'])
            cor_nome = color_config['name']
            
            print(f"\n🤖 Processando {robot_name}: {len(waypoints)} waypoints (cor: {cor_nome})")
            
            # Preparar tarefas de bolas
            tarefas_bolas = []
            coordenadas_waypoints = []  # Para criar linhas depois
            
            for wp in waypoints:
                # Converter GPS para Gazebo
                x, y = gps_to_gazebo_coords(wp['lat'], wp['lon'])
                coordenadas_waypoints.append((wp['id'], x, y))
                
                # Criar bola
                model_name, sdf_content = criar_bola_waypoint(
                    f"{robot_name}_{wp['id']}", 
                    x, y, z_altura, 
                    color_config
                )
                tarefas_bolas.append((sdf_content, model_name, usar_ros))
            
            # Adicionar bolas em paralelo
            futures = [executor.submit(adicionar_waypoint_worker, tarefa) for tarefa in tarefas_bolas]
            
            sucessos_robo = 0
            erros_robo = 0
            for future in as_completed(futures):
                try:
                    sucesso = future.result()
                    if sucesso:
                        waypoints_adicionados += 1
                        sucessos_robo += 1
                    else:
                        erros += 1
                        erros_robo += 1
                except Exception:
                    erros += 1
                    erros_robo += 1
            
            print(f"  ✅ {sucessos_robo} bolas adicionadas para {robot_name}")
            if erros_robo > 0:
                print(f"  ❌ {erros_robo} erros para {robot_name}")
            
            # Adicionar linhas conectando waypoints consecutivos
            if adicionar_linhas and len(coordenadas_waypoints) > 1:
                tarefas_linhas = []
                
                for i in range(len(coordenadas_waypoints) - 1):
                    wp1_id, x1, y1 = coordenadas_waypoints[i]
                    wp2_id, x2, y2 = coordenadas_waypoints[i + 1]
                    
                    model_name, sdf_content = criar_linha_waypoints(
                        f"{robot_name}_{wp1_id}",
                        f"{robot_name}_{wp2_id}",
                        x1, y1, x2, y2,
                        z_altura,
                        color_config
                    )
                    tarefas_linhas.append((sdf_content, model_name, usar_ros))
                
                # Adicionar linhas em paralelo
                futures = [executor.submit(adicionar_waypoint_worker, tarefa) for tarefa in tarefas_linhas]
                
                sucessos_linhas = 0
                erros_linhas = 0
                for future in as_completed(futures):
                    try:
                        sucesso = future.result()
                        if sucesso:
                            linhas_adicionadas += 1
                            sucessos_linhas += 1
                        else:
                            erros += 1
                            erros_linhas += 1
                    except Exception:
                        erros += 1
                        erros_linhas += 1
                
                print(f"  ✅ {sucessos_linhas} linhas adicionadas para {robot_name}")
                if erros_linhas > 0:
                    print(f"  ❌ {erros_linhas} erros de linhas para {robot_name}")
    
    # Resultados
    print(f"\n📊 RESULTADOS:")
    print(f"✅ Waypoints adicionados: {waypoints_adicionados}")
    if adicionar_linhas:
        print(f"✅ Linhas adicionadas: {linhas_adicionadas}")
    if erros > 0:
        print(f"❌ Erros: {erros}")
    
    print(f"\n🎉 Waypoints visualizados no Gazebo!")
    print("=" * 60)
    
    return {
        'success': True,
        'waypoints_added': waypoints_adicionados,
        'lines_added': linhas_adicionadas,
        'errors': erros,
        'robots': list(waypoints_by_robot.keys())
    }

def remover_waypoints_missao(max_workers=8):
    """
    Remove todos os waypoints de missão do Gazebo
    
    Args:
        max_workers: Número de threads paralelas
        
    Returns:
        dict: Estatísticas da operação
    """
    gazebo_running, ros_running = verificar_gazebo_rodando()
    
    if not gazebo_running:
        return {
            'success': False,
            'message': 'Gazebo não está rodando',
            'waypoints_removed': 0
        }
    
    usar_ros = ros_running
    deletados = limpar_waypoints_antigos(usar_ros, max_workers)
    
    return {
        'success': True,
        'waypoints_removed': deletados
    }


