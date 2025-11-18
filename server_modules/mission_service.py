"""
Módulo para execução e gerenciamento de missões
"""

import os
import time
from datetime import datetime
from mission_bridgetoap import preparar_e_enviar_missoes
from .rover_manager import create_rover_config_from_frontend_data, REVERSE_ROVER_MAPPING
from .gazebo_visualizer import adicionar_waypoints_missao, remover_waypoints_missao

class MissionService:
    def __init__(self, socketio, monitoring_service):
        self.socketio = socketio
        self.monitoring_service = monitoring_service
        self.mission_manager = None
        self.mission_active = False
        
        # Configurações com caminhos relativos ao diretório RoverModel
        # (assumindo que o servidor executa de /home/viki/OLHE-5G-Dashboard/RoverModel/)
        self.graph_path = "./jsons/graph_equipment.json"
        self.observation_points_json_path = "./jsons/obs_equipment.json"
        self.parameters_file_path = "./planilhas/equipment_processado.xlsx"
        
        # Validar se arquivos existem ao inicializar
        for path_name, path_value in [
            ("graph_path", self.graph_path),
            ("observation_points_json_path", self.observation_points_json_path),
            ("parameters_file_path", self.parameters_file_path)
        ]:
            if not os.path.exists(path_value):
                abs_path = os.path.abspath(path_value)
                print(f"⚠️  AVISO: Arquivo não encontrado: {path_name} = {path_value} (absoluto: {abs_path})")
            else:
                abs_path = os.path.abspath(path_value)
                print(f"✅ Arquivo encontrado: {path_name} = {abs_path}")
        
        self.deltax_m = -2
        self.deltay_m = -12.0
    
    def extract_mission_id_from_equipment(self, equipment_data):
        """
        Extrai o ID da missão diretamente do campo equipmentId do equipamento
        
        Args:
            equipment_data: Dados do equipamento do frontend
            
        Returns:
            str: Identificador da missão (ex: 'ef_reator1', 'b_busip4', etc.)
        """
        # O equipmentId já contém o identificador correto da missão (última coluna do CSV)
        mission_id = equipment_data.get('equipmentId', '')
        
        if not mission_id:
            # Fallback: tentar extrair do ID principal se equipmentId não estiver disponível
            equipment_id = equipment_data.get('id', '')
            equipment_name = equipment_data.get('name', '')
            print(f"⚠️ equipmentId não encontrado para {equipment_name}, usando fallback")
            return None
        
        print(f"✅ Equipamento {equipment_data.get('name', 'N/A')} -> missão: {mission_id}")
        return mission_id
    
    def execute_mission(self, data):
        """
        Executa uma missão com base nos dados recebidos do frontend
        
        Args:
            data: Dados da missão do frontend
            
        Returns:
            dict: Resultado da execução da missão
        """
        try:
            print(f"\n" + "="*80)
            print(f"🎯 NOVA MISSÃO RECEBIDA")
            print(f"="*80)
            print(f"📋 Dados recebidos: {data}")
            
            # Processar dados do frontend (formato do MissionPlanningView.js)
            if 'rovers' in data and isinstance(data['rovers'], list):
                mission_name = data.get('name', f'Missão {datetime.now().strftime("%Y-%m-%d %H:%M")}')
                rovers_list = data.get('rovers', [])
                equipments_list = data.get('equipments', [])
                
                print(f"📝 Nome: {mission_name}")
                print(f"🤖 Rovers: {[r.get('identifier', r.get('name', 'N/A')) for r in rovers_list]}")
                print(f"⚡ Equipamentos: {[e.get('name', e.get('id', 'N/A')) for e in equipments_list]}")
                
                # Converter rovers usando mapeamento
                robots = []
                for i, rover_data in enumerate(rovers_list):
                    rover_config = create_rover_config_from_frontend_data(rover_data, i)
                    robots.append(rover_config)
                
                # Converter equipamentos para lista de missões
                missions = []
                print(f"\n📋 PROCESSANDO EQUIPAMENTOS SELECIONADOS:")
                for equipment in equipments_list:
                    mission_id = self.extract_mission_id_from_equipment(equipment)
                    if mission_id:
                        missions.append(mission_id)
                        print(f"   ✅ {equipment.get('name', 'N/A')} -> {mission_id}")
                    else:
                        print(f"   ❌ Falha ao extrair mission_id para {equipment.get('name', 'N/A')}")
                
                # Remover duplicatas mantendo ordem
                missions = list(dict.fromkeys(missions))
                
            else:
                return {
                    "success": False,
                    "message": "Formato de dados inválido. Esperado: {rovers: [...], equipments: [...]}",
                    "status_code": 400
                }
            
            # Validações
            if not robots:
                return {
                    "success": False,
                    "message": "Lista de robôs não pode estar vazia",
                    "status_code": 400
                }
                
            if not missions:
                return {
                    "success": False,
                    "message": "Lista de equipamentos não pode estar vazia",
                    "status_code": 400
                }
            
            print(f"\n📋 CONFIGURAÇÃO PROCESSADA:")
            print(f"   • Missões: {', '.join(missions)}")
            print(f"   • Robôs: {', '.join([r['name'] for r in robots])}")
            
            # Executar a função principal do mission_bridgetoap.py
            print(f"\n🚀 EXECUTANDO preparar_e_enviar_missoes...")
            
            resultado = preparar_e_enviar_missoes(
                robots=robots,
                missions=missions,
                graph_path=self.graph_path,
                observation_points_json_path=self.observation_points_json_path,
                file_path_parametros=self.parameters_file_path,
                deltax_m=self.deltax_m,
                deltay_m=self.deltay_m,
                gps_rate_hz=5.0,
                wait_timeout_s=25.0,
                wait_each_timeout_s=5.0,
                duplicate_first=True,
                hold_vistoria_s=2.0,
                hold_passagem_s=0.0,
                tol_ct_m=0.10,
                preserve_loop_closure=True,
                renumber_ids=True,
                do_plots=False
            )
            
            print("✅ Missão executada com sucesso!")
            
            # Debug: verificar se a função retornou os dados esperados
            print(f"\n🔍 DEBUG: Verificando retorno da função preparar_e_enviar_missoes...")
            print(f"   • Tipo do resultado: {type(resultado)}")
            print(f"   • Chaves do resultado: {list(resultado.keys())}")
            
            # Armazenar mission_manager para monitoramento
            self.mission_manager = resultado["mission_manager"]
            self.mission_active = True
            
            # Configurar monitoramento para usar o mission_manager
            self.monitoring_service.set_mission_manager(self.mission_manager, self.mission_active)
            
            # Extrair waypoints das missões otimizadas para o frontend
            waypoints_for_frontend = []
            total_waypoints = 0
            
            print(f"\n🔍 DEBUG: Verificando estrutura dos waypoints...")
            print(f"   • Chaves disponíveis no resultado: {list(resultado.keys())}")
            
            if "missoes_otimizadas" in resultado:
                print(f"   • Robôs com missões otimizadas: {list(resultado['missoes_otimizadas'].keys())}")
                
                for robot_name, mission_points in resultado["missoes_otimizadas"].items():
                    print(f"   • {robot_name}: {len(mission_points)} pontos de missão")
                    
                    # Debug: mostrar estrutura do primeiro ponto
                    if mission_points:
                        first_point = mission_points[0]
                        print(f"     - Estrutura do primeiro ponto: {list(first_point.keys())}")
                        print(f"     - Exemplo: {first_point}")
                    
                    for point in mission_points:
                        waypoint = {
                            'id': point['id'],
                            'lat': point['lat'],
                            'lon': point['lon'],
                            'hold': point.get('hold', 0.0),
                            'robot': robot_name,
                            'original_identifier': REVERSE_ROVER_MAPPING.get(robot_name, robot_name)
                        }
                        waypoints_for_frontend.append(waypoint)
                        total_waypoints += 1
            else:
                print(f"   ❌ Chave 'missoes_otimizadas' não encontrada no resultado!")
                print(f"   • Chaves disponíveis: {list(resultado.keys())}")
            
            # Organizar waypoints por robô para o frontend (formato esperado pelo useWebSocket.js)
            waypoints_by_robot = {}
            for robot_name, mission_points in resultado["missoes_otimizadas"].items():
                waypoints_by_robot[robot_name] = []
                for point in mission_points:
                    waypoint = {
                        'id': point['id'],
                        'lat': point['lat'],
                        'lon': point['lon'],
                        'latitude': point['lat'],  # Duplicar para compatibilidade
                        'longitude': point['lon'],  # Duplicar para compatibilidade
                        'hold': point.get('hold', 0.0),
                        'accept_radius': point.get('accept_radius', 0.0),
                        'pass_radius': point.get('pass_radius', 0.0),
                        'yaw_deg': point.get('yaw_deg', 0.0),
                        'robot': robot_name,
                        'original_identifier': REVERSE_ROVER_MAPPING.get(robot_name, robot_name)
                    }
                    waypoints_by_robot[robot_name].append(waypoint)
            
            # Adicionar waypoints no Gazebo como bolas coloridas
            try:
                print(f"\n🎨 Adicionando waypoints no Gazebo...")
                visualizacao_resultado = adicionar_waypoints_missao(
                    waypoints_by_robot,
                    z_altura=3.0,  # Altura das bolas (3 metros)
                    adicionar_linhas=True,  # Adicionar linhas conectando waypoints
                    max_workers=8  # Threads paralelas (mesmo do RealTime_CSV2World.py)
                )
                if visualizacao_resultado['success']:
                    print(f"✅ {visualizacao_resultado['waypoints_added']} waypoints adicionados no Gazebo")
                    print(f"✅ {visualizacao_resultado['lines_added']} linhas adicionadas no Gazebo")
                else:
                    print(f"⚠️ Falha ao adicionar waypoints no Gazebo: {visualizacao_resultado.get('message', 'Erro desconhecido')}")
            except Exception as e:
                print(f"⚠️ Erro ao visualizar waypoints no Gazebo: {e}")
                import traceback
                traceback.print_exc()
            
            # Emitir via WebSocket
            try:
                print(f"\n📡 [WEBSOCKET] Enviando waypoints da missão:")
                print(f"   🎯 Total de waypoints: {total_waypoints}")
                print(f"   🤖 Robôs: {list(resultado['missoes_otimizadas'].keys())}")
                print(f"   📋 Waypoints por robô:")
                for robot_name, robot_waypoints in waypoints_by_robot.items():
                    print(f"     • {robot_name}: {len(robot_waypoints)} waypoints")
                    if robot_waypoints:
                        first_wp = robot_waypoints[0]
                        last_wp = robot_waypoints[-1]
                        print(f"       - Primeiro: lat={first_wp['lat']:.6f}, lon={first_wp['lon']:.6f}")
                        print(f"       - Último: lat={last_wp['lat']:.6f}, lon={last_wp['lon']:.6f}")
                        print(f"       - Estrutura completa do primeiro waypoint: {first_wp}")
                
                websocket_data = {
                    "waypoints": waypoints_for_frontend,  # Array de todos os waypoints (formato antigo)
                    "waypoints_by_robot": waypoints_by_robot,  # Formato esperado pelo frontend
                    "mission_active": True,
                    "robots": [r['name'] for r in robots],
                    "timestamp": time.time()
                }
                
                print(f"   📤 Dados do WebSocket: {len(websocket_data['waypoints'])} waypoints totais")
                print(f"   📤 Formato waypoints_by_robot: {len(waypoints_by_robot)} robôs")
                print(f"   📤 Evento: mission_waypoints_update")
                print(f"   📤 SocketIO disponível: {self.socketio is not None}")
                
                # Debug: verificar se há clientes conectados
                try:
                    connected_clients = len(self.socketio.server.manager.rooms.get('/', {}).get('', set()))
                    print(f"   📤 Clientes conectados: {connected_clients}")
                except:
                    print(f"   📤 Clientes conectados: Não foi possível determinar")
                
                self.socketio.emit('mission_waypoints_update', websocket_data)
                print(f"   ✅ WebSocket emitido com sucesso!")
                print(f"   📊 Dados enviados: {len(websocket_data['waypoints'])} waypoints para {len(websocket_data['robots'])} robôs")
                print(f"   📊 Formato waypoints_by_robot enviado com {len(waypoints_by_robot)} robôs")
                
            except Exception as e:
                print(f"⚠️ Erro ao emitir waypoints via WebSocket: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n✅ MISSÃO CONFIGURADA COM SUCESSO!")
            print(f"   • {len(robots)} robôs configurados")
            print(f"   • {len(missions)} missões planejadas")
            print(f"   • {total_waypoints} waypoints totais gerados")
            
            return {
                "success": True,
                "message": "Missão executada e enviada com sucesso",
                "data": {
                    "robots_used": len(robots),
                    "robots": [r['name'] for r in robots],
                    "total_missions": len(missions),
                    "missions": missions,
                    "waypoints": waypoints_for_frontend,
                    "total_waypoints": total_waypoints,
                    "mission_active": True
                },
                "status_code": 200
            }
            
        except ValueError as e:
            # Erros de validação (arrays vazios, missões inválidas, etc.)
            print(f"\n❌ Erro de validação: {str(e)}")
            return {
                "success": False,
                "message": f"Validação falhou: {str(e)}",
                "status_code": 400
            }
        except Exception as e:
            # Outros erros
            print(f"\n❌ Erro durante execução: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "message": f"Erro durante planejamento: {str(e)}",
                "status_code": 500
            }
    
    def stop_mission(self):
        """Para a missão atual"""
        self.mission_active = False
        
        # Fechar conexões do mission_manager se existir
        if self.mission_manager:
            print(f"🧹 Fechando conexões do canal de missão...")
            try:
                self.mission_manager.close()
                print("✅ Conexões do canal de missão fechadas com sucesso")
            except Exception as e:
                print(f"⚠️ Erro ao fechar conexões do canal de missão: {e}")
            self.mission_manager = None
        
        # Remover waypoints do Gazebo
        try:
            print(f"🧹 Removendo waypoints do Gazebo...")
            remocao_resultado = remover_waypoints_missao(max_workers=8)
            if remocao_resultado['success']:
                print(f"✅ {remocao_resultado['waypoints_removed']} waypoints removidos do Gazebo")
            else:
                print(f"⚠️ Falha ao remover waypoints do Gazebo: {remocao_resultado.get('message', 'Erro desconhecido')}")
        except Exception as e:
            print(f"⚠️ Erro ao remover waypoints do Gazebo: {e}")
        
        # Configurar monitoramento para voltar ao monitoramento contínuo
        self.monitoring_service.set_mission_manager(None, False)
        
        # Emitir via WebSocket
        try:
            print(f"📡 [WEBSOCKET] Enviando parada de missão")
            self.socketio.emit('mission_stopped', {
                "message": "Missão interrompida",
                "mission_active": False,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"⚠️ Erro ao emitir parada de missão via WebSocket: {e}")
        
        return {
            "success": True,
            "message": "Missão interrompida com sucesso"
        }
    
    def get_mission_status(self):
        """Obtém o status atual da missão"""
        return {
            "success": True,
            "data": {
                "mission_active": self.mission_active,
                "connected_robots": len(self.mission_manager.connected) if self.mission_manager and self.mission_manager.is_connected else 0,
                "robots": list(self.mission_manager.connected) if self.mission_manager and self.mission_manager.is_connected else [],
                "timestamp": time.time()
            }
        }
