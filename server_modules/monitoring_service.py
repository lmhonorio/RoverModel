"""
Módulo para monitoramento de posições dos robôs
"""

import time
import threading
from mission_bridgetoap import get_latlon
from .rover_manager import REVERSE_ROVER_MAPPING

class MonitoringService:
    def __init__(self, socketio):
        self.socketio = socketio
        self.stop_monitoring = False
        self.monitoring_thread = None
        self.mission_manager = None
        self.mission_active = False
        self.position_monitoring_manager = None
    
    def start_monitoring(self, position_monitoring_manager):
        """Inicia o monitoramento de posições"""
        self.position_monitoring_manager = position_monitoring_manager
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_robot_positions)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print("✅ Thread de monitoramento iniciada")
    
    def stop_monitoring_service(self):
        """Para o monitoramento de posições"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("🔄 Monitoramento de posições finalizado")
    
    def set_mission_manager(self, mission_manager, mission_active):
        """Define o mission_manager para monitoramento durante missões"""
        self.mission_manager = mission_manager
        self.mission_active = mission_active
    
    def _monitor_robot_positions(self):
        """
        Thread para monitorar posições dos robôs de 2 em 2 segundos
        Usa mission_manager durante missões ou position_monitoring_manager para monitoramento contínuo
        """
        print("🔄 Iniciando monitoramento de posições dos robôs...")
        
        cycle_count = 0
        while not self.stop_monitoring:
            try:
                # Escolher qual manager usar: mission_manager (durante missão) ou position_monitoring_manager (monitoramento contínuo)
                current_manager = self.mission_manager if self.mission_active and self.mission_manager else self.position_monitoring_manager
                
                if current_manager and current_manager.is_connected:
                    current_time = time.time()
                    positions_data = []
                    
                    # Log de debug a cada 10 ciclos
                    if cycle_count % 10 == 0:
                        print(f"🔍 [MONITOR] Ciclo {cycle_count}: {len(current_manager.connected)} robôs conectados: {list(current_manager.connected)}")
                    
                    # Fazer polling de telemetria para coletar dados atualizados
                    try:
                        current_manager.poll_once(per_robot_reads=10)
                    except Exception as e:
                        print(f"⚠️ Erro no polling de telemetria: {e}")
                    
                    # Obter posições de cada robô individualmente para garantir que todos sejam monitorados
                    for robot_name in current_manager.connected:
                        try:
                            # Obter posição específica de cada robô
                            robot_state = current_manager.new_wait_for_position(robot=robot_name, timeout=1.0, require_all=False)
                            
                            if robot_state:
                                # O método new_wait_for_position retorna o estado diretamente para um robô específico
                                lat, lon = get_latlon({robot_name: robot_state}, robot_name)
                                
                                if lat is not None and lon is not None:
                                    # Obter identifier original do mapeamento reverso
                                    original_identifier = REVERSE_ROVER_MAPPING.get(robot_name, robot_name)
                                    
                                    # Obter status do robô
                                    robot_status = robot_state.get("status", "unknown") if isinstance(robot_state, dict) else "unknown"
                                    
                                    position_data = {
                                        "robot_id": robot_name,
                                        "original_identifier": original_identifier,
                                        "latitude": round(float(lat), 8),
                                        "longitude": round(float(lon), 8),
                                        "timestamp": current_time,
                                        "status": robot_status,
                                        "mission_active": self.mission_active
                                    }
                                    
                                    positions_data.append(position_data)
                                    print(f"📍 {robot_name} ({original_identifier}): lat={lat:.6f}, lon={lon:.6f} {'[MISSÃO]' if self.mission_active else '[MONITOR]'}")
                                else:
                                    print(f"⚠️ {robot_name}: Posição inválida (lat={lat}, lon={lon})")
                            else:
                                print(f"⚠️ {robot_name}: Sem dados de posição disponíveis")
                                
                        except Exception as e:
                            print(f"⚠️ Erro ao obter posição do {robot_name}: {e}")
                    
                    # Enviar via WebSocket - usar evento que o frontend espera
                    if positions_data:
                        try:
                            # Log de debug a cada 10 ciclos
                            if cycle_count % 10 == 0:
                                print(f"🔍 [MONITOR] Enviando {len(positions_data)} posições via WebSocket")
                            
                            # Enviar cada posição individualmente no formato que o frontend espera
                            for position_data in positions_data:
                                self.socketio.emit('robot_position_continuous', position_data)
                                
                                # Log detalhado a cada 10 ciclos
                                if cycle_count % 10 == 0:
                                    print(f"🔍 [MONITOR] Enviado: {position_data['robot_id']} - lat={position_data['latitude']:.6f}, lon={position_data['longitude']:.6f}")
                        except Exception as e:
                            print(f"⚠️ Erro ao enviar posições via WebSocket: {e}")
                    else:
                        if cycle_count % 10 == 0:
                            print("⚠️ Nenhuma posição válida obtida de nenhum robô")
                        
                        # Se não há posições, tentar reconectar se for o position_monitoring_manager
                        if not self.mission_active and self.position_monitoring_manager:
                            print("🔄 Tentando reconectar robôs para monitoramento...")
                            try:
                                self.position_monitoring_manager.connect_all()
                            except Exception as e:
                                print(f"⚠️ Erro ao reconectar: {e}")
                
                time.sleep(2)  # Aguardar 2 segundos
                cycle_count += 1
                
            except Exception as e:
                print(f"⚠️ Erro no monitoramento de posições: {e}")
                time.sleep(5)
                cycle_count += 1
        
        print("🔄 Monitoramento de posições finalizado")
