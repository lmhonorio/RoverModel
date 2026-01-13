#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Operation_Mavlink.py
====================
Escuta mensagens MAVLink e monitora comandos de missão.
Conecta-se ao veículo para receber cópias de todas as mensagens de missão.

Para usar este script, você precisa adicionar uma saída extra no sim_vehicle.py:
    --out=udp:127.0.0.1:14552

Ou rode este script e ele tentará se conectar à porta 14551 (MAVROS do primeiro rover).

Uso:
    python3 Operation_Mavlink.py [porta]
    
Exemplos:
    python3 Operation_Mavlink.py          # Conecta a 14551 (porta MAVROS rover 0)
    python3 Operation_Mavlink.py 14550    # Conecta a 14550 (porta QGC)
    python3 Operation_Mavlink.py 14552    # Escuta na 14552 (precisa adicionar --out no SITL)
"""

import sys
import time
import json
import threading
import socket
import select
from datetime import datetime
from pymavlink import mavutil
from collections import OrderedDict


class MavlinkMissionListener:
    """
    Classe para escutar e registrar comandos de missão MAVLink.
    """
    
    def __init__(self, port=14551, use_udpin=False):
        """
        Inicializa o listener MAVLink.
        
        Args:
            port: Porta para conectar/escutar (padrão: 14551)
            use_udpin: Se True, escuta conexões de entrada (servidor)
        """
        self.port = port
        self.use_udpin = use_udpin
        self.master = None
        self.mission_commands = OrderedDict()  # Dicionário ordenado para armazenar comandos
        self.command_counter = 0
        self.last_mission_count = 0
        self.mission_items = {}  # Cache temporário de itens de missão por sequence
        self.last_current_item = -1  # Último item MISSION_CURRENT exibido (para evitar spam)
        self.last_reached_item = -1  # Último item MISSION_ITEM_REACHED exibido
        self.last_mode = None
        self.loiter_active = False
        
        # Dicionário de nomes de comandos MAV_CMD (mais completo)
        self.command_names = {
            16: "MAV_CMD_NAV_WAYPOINT",
            17: "MAV_CMD_NAV_LOITER_UNLIM",
            18: "MAV_CMD_NAV_LOITER_TURNS",
            19: "MAV_CMD_NAV_LOITER_TIME",
            20: "MAV_CMD_NAV_RETURN_TO_LAUNCH",
            21: "MAV_CMD_NAV_LAND",
            22: "MAV_CMD_NAV_TAKEOFF",
            23: "MAV_CMD_NAV_LAND_LOCAL",
            24: "MAV_CMD_NAV_TAKEOFF_LOCAL",
            25: "MAV_CMD_NAV_FOLLOW",
            80: "MAV_CMD_NAV_ROI",
            81: "MAV_CMD_NAV_PATHPLANNING",
            82: "MAV_CMD_NAV_SPLINE_WAYPOINT",
            83: "MAV_CMD_NAV_VTOL_TAKEOFF",
            84: "MAV_CMD_NAV_VTOL_LAND",
            85: "MAV_CMD_NAV_GUIDED_ENABLE",
            86: "MAV_CMD_NAV_DELAY",
            89: "MAV_CMD_NAV_PAYLOAD_PLACE",
            92: "MAV_CMD_NAV_LAST",
            93: "MAV_CMD_CONDITION_DELAY",
            94: "MAV_CMD_CONDITION_CHANGE_ALT",
            95: "MAV_CMD_CONDITION_DISTANCE",
            96: "MAV_CMD_CONDITION_YAW",
            112: "MAV_CMD_DO_SET_MODE",
            113: "MAV_CMD_DO_JUMP",
            114: "MAV_CMD_DO_CHANGE_SPEED",
            115: "MAV_CMD_DO_SET_HOME",
            159: "MAV_CMD_DO_SET_ROI_LOCATION",
            176: "MAV_CMD_DO_SET_PARAMETER",
            177: "MAV_CMD_DO_SET_RELAY",
            178: "MAV_CMD_DO_REPEAT_RELAY",
            179: "MAV_CMD_DO_SET_SERVO",
            180: "MAV_CMD_DO_REPEAT_SERVO",
            181: "MAV_CMD_DO_FLIGHTTERMINATION",
            182: "MAV_CMD_DO_CHANGE_ALTITUDE",
            183: "MAV_CMD_DO_LAND_START",
            184: "MAV_CMD_DO_RALLY_LAND",
            189: "MAV_CMD_DO_GO_AROUND",
            190: "MAV_CMD_DO_REPOSITION",
            191: "MAV_CMD_DO_PAUSE_CONTINUE",
            192: "MAV_CMD_DO_SET_REVERSE",
            193: "MAV_CMD_DO_SET_ROI_LOCATION",
            195: "MAV_CMD_DO_SET_ROI_WPNEXT_OFFSET",
            196: "MAV_CMD_DO_SET_ROI_NONE",
            200: "MAV_CMD_DO_CONTROL_VIDEO",
            2000: "MAV_CMD_IMAGE_START_CAPTURE_CUSTOM",
            201: "MAV_CMD_DO_SET_ROI",
            202: "MAV_CMD_DO_DIGICAM_CONFIGURE",
            203: "MAV_CMD_DO_DIGICAM_CONTROL",
            204: "MAV_CMD_DO_MOUNT_CONFIGURE",
            205: "MAV_CMD_DO_MOUNT_CONTROL",
            206: "MAV_CMD_DO_SET_CAM_TRIGG_DIST",
            207: "MAV_CMD_DO_FENCE_ENABLE",
            208: "MAV_CMD_DO_PARACHUTE",
            209: "MAV_CMD_DO_MOTOR_TEST",
            210: "MAV_CMD_DO_INVERTED_FLIGHT",
            211: "MAV_CMD_NAV_SET_YAW_SPEED",
            212: "MAV_CMD_DO_SET_CAM_TRIGG_INTERVAL",
            213: "MAV_CMD_DO_MOUNT_CONTROL_QUAT",
            214: "MAV_CMD_DO_GUIDED_MASTER",
            215: "MAV_CMD_DO_GUIDED_LIMITS",
            216: "MAV_CMD_DO_ENGINE_CONTROL",
            220: "MAV_CMD_DO_LAST",
            221: "MAV_CMD_DO_TRIGGER_CONTROL",
            240: "MAV_CMD_DO_VTOL_TRANSITION",
            241: "MAV_CMD_ARM_AUTHORIZATION_REQUEST",
            252: "MAV_CMD_PREFLIGHT_CALIBRATION",
            253: "MAV_CMD_PREFLIGHT_SET_SENSOR_OFFSETS",
            2500: "MAV_CMD_IMAGE_START_CAPTURE",
            2501: "MAV_CMD_IMAGE_STOP_CAPTURE",
            2502: "MAV_CMD_REQUEST_CAMERA_IMAGE_CAPTURE",
            2503: "MAV_CMD_DO_TRIGGER_CONTROL",
            2504: "MAV_CMD_VIDEO_START_STREAMING",
            2505: "MAV_CMD_VIDEO_STOP_STREAMING",
            2510: "MAV_CMD_LOGGING_START",
            2511: "MAV_CMD_LOGGING_STOP",
            2520: "MAV_CMD_AIRFRAME_CONFIGURATION",
            2600: "MAV_CMD_VIDEO_START_CAPTURE",
            2601: "MAV_CMD_VIDEO_STOP_CAPTURE",
            30001: "MAV_CMD_PAYLOAD_PREPARE_DEPLOY",
            30002: "MAV_CMD_PAYLOAD_CONTROL_DEPLOY",
            42000: "MAV_CMD_WAYPOINT_USER_1",
            42001: "MAV_CMD_WAYPOINT_USER_2",
            42002: "MAV_CMD_WAYPOINT_USER_3",
            42003: "MAV_CMD_WAYPOINT_USER_4",
            42004: "MAV_CMD_WAYPOINT_USER_5",
            42005: "MAV_CMD_SPATIAL_USER_1",
            42006: "MAV_CMD_SPATIAL_USER_2",
            42007: "MAV_CMD_SPATIAL_USER_3",
            42008: "MAV_CMD_SPATIAL_USER_4",
            42009: "MAV_CMD_SPATIAL_USER_5",
            42424: "MAV_CMD_DO_WINCH",
            42600: "MAV_CMD_USER_1",
            42601: "MAV_CMD_USER_2",
            42602: "MAV_CMD_USER_3",
            42603: "MAV_CMD_USER_4",
            42604: "MAV_CMD_USER_5",
        }
        
    def connect(self):
        """
        Estabelece conexão com o veículo MAVLink.
        """
        print("=" * 80)
        print("🔌 MAVLINK MISSION LISTENER")
        print("=" * 80)
        
        if self.use_udpin:
            connection_string = f'udpin:127.0.0.1:{self.port}'
            print(f"📡 Escutando em: {connection_string} (modo servidor)")
            print("   ⚠️  IMPORTANTE: Adicione ao sim_vehicle.py:")
            print(f"      --out=udp:127.0.0.1:{self.port}")
        else:
            connection_string = f'udp:127.0.0.1:{self.port}'
            print(f"📡 Conectando a: {connection_string} (modo cliente)")
        
        print("⏳ Aguardando heartbeat...")
        
        try:
            # Conectar ao MAVLink
            self.master = mavutil.mavlink_connection(
                connection_string,
                input=True,  # Permitir receber mensagens
                dialect="ardupilotmega",
                source_system=254,  # ID do GCS
                source_component=190  # MAV_COMP_ID_PATHPLANNER
            )
            
            # Aguardar primeiro heartbeat (timeout aumentado)
            msg = self.master.wait_heartbeat(timeout=60)
            
            if msg:
                print(f"✅ Conectado ao sistema {self.master.target_system}, componente {self.master.target_component}")
                print(f"   Tipo: {msg.type}, Autopilot: {msg.autopilot}")
                print(f"   Base mode: {msg.base_mode}, Custom mode: {msg.custom_mode}")
            else:
                print("⚠️  Conectado mas sem heartbeat ainda...")
            
            print("=" * 80)
            print("🎧 Escutando comandos de missão...")
            print("   💡 Envie uma missão pelo QGroundControl para vê-la aqui")
            print("   ⏹️  Pressione Ctrl+C para parar")
            print("=" * 80)
            print()
            
            # Solicitar missão atual do veículo (se houver)
            print("📥 Verificando se há missão já carregada no veículo...")
            try:
                self.master.mav.mission_request_list_send(
                    self.master.target_system,
                    self.master.target_component,
                    mission_type=0  # MAV_MISSION_TYPE_MISSION
                )
                print("   ✅ Solicitação enviada")
            except Exception as e:
                print(f"   ⚠️  Erro: {e}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            print()
            print("💡 Dicas de troubleshooting:")
            print("   1. Verifique se o SITL está rodando")
            print("   2. Tente outra porta: python3 Operation_Mavlink.py 14550")
            if self.use_udpin:
                print(f"   3. Adicione ao sim_vehicle.py: --out=udp:127.0.0.1:{self.port}")
            print()
            return False
    
    def request_mission_list(self):
        """
        Solicita a lista completa de missão do veículo.
        """
        if not self.master:
            return
        
        try:
            print(f"📥 Solicitando lista de missão do veículo...")
            
            # Enviar MISSION_REQUEST_LIST
            self.master.mav.mission_request_list_send(
                self.master.target_system,
                self.master.target_component,
                mission_type=0  # MAV_MISSION_TYPE_MISSION
            )
            
            print(f"   ✅ MISSION_REQUEST_LIST enviado")
            
        except Exception as e:
            print(f"   ⚠️  Erro ao solicitar lista: {e}")
    
    def get_command_name(self, command_id):
        """
        Retorna o nome legível de um comando MAV_CMD.
        
        Args:
            command_id: ID numérico do comando
            
        Returns:
            Nome do comando ou "UNKNOWN_COMMAND_<id>"
        """
        return self.command_names.get(command_id, f"UNKNOWN_CMD_{command_id}")
    
    def format_mission_item(self, msg):
        """
        Formata uma mensagem MISSION_ITEM para exibição.
        
        Args:
            msg: Mensagem MAVLink MISSION_ITEM ou MISSION_ITEM_INT
            
        Returns:
            Dicionário com informações formatadas
        """
        command_name = self.get_command_name(msg.command)
        
        mission_data = {
            'seq': msg.seq,
            'command_id': msg.command,
            'command_name': command_name,
            'frame': msg.frame,
            'current': msg.current,
            'autocontinue': msg.autocontinue,
            'params': {
                'param1': msg.param1,
                'param2': msg.param2,
                'param3': msg.param3,
                'param4': msg.param4,
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        # Adicionar coordenadas dependendo do tipo de mensagem
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            msg_type = msg.get_type()
            if msg_type == 'MISSION_ITEM_INT':
                # MISSION_ITEM_INT usa coordenadas em graus * 1e7
                mission_data['latitude'] = msg.x / 1e7 if msg.x != 0 else 0
                mission_data['longitude'] = msg.y / 1e7 if msg.y != 0 else 0
            else:
                # MISSION_ITEM usa coordenadas diretas (float)
                mission_data['latitude'] = msg.x
                mission_data['longitude'] = msg.y
            mission_data['altitude'] = msg.z
        
        # Informação sobre o sistema
        if hasattr(msg, 'target_system') and hasattr(msg, 'target_component'):
            mission_data['target_system'] = msg.target_system
            mission_data['target_component'] = msg.target_component
        
        return mission_data
    
    def print_mission_command(self, mission_data):
        """
        Imprime informações de um comando de missão.
        
        Args:
            mission_data: Dicionário com dados da missão
        """
        # Usar cores ANSI para melhor visualização
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        command_id = mission_data['command_id']
        marker = ""
        # Comandos de foto mais comuns
        if command_id == 2000:
            marker = f" {BOLD}{YELLOW}📸 (FOTO DETECTADA: ID {command_id}){RESET}"
        # Comando de direcionamento/gimbal
        elif command_id == 205:
            marker = f" {BOLD}{CYAN}🧭 (DIRECIONAMENTO: ID {command_id}){RESET}"

        print(f"{BOLD}{BLUE}📍 Comando {mission_data['seq']:02d}{RESET} | {GREEN}{mission_data['command_name']}{RESET} {CYAN}(ID: {command_id}){RESET}{marker}")
        print(f"   ⏰ Timestamp: {mission_data['timestamp']}")
        print(f"   {YELLOW}🎯 Params:{RESET}")
        print(f"      • param1: {mission_data['params']['param1']:.6f}")
        print(f"      • param2: {mission_data['params']['param2']:.6f}")
        print(f"      • param3: {mission_data['params']['param3']:.6f}")
        print(f"      • param4: {mission_data['params']['param4']:.6f}")
        
        if 'latitude' in mission_data and 'longitude' in mission_data:
            lat = mission_data['latitude']
            lon = mission_data['longitude']
            alt = mission_data['altitude']
            
            if lat != 0 or lon != 0:  # Só mostrar se tiver coordenadas válidas
                print(f"   {GREEN}📍 Coordenadas:{RESET}")
                print(f"      • Latitude:  {lat:11.7f}°")
                print(f"      • Longitude: {lon:11.7f}°")
                print(f"      • Altitude:  {alt:8.2f} m")
        
        print(f"   ⚙️  Frame: {mission_data['frame']} | Current: {mission_data['current']} | AutoContinue: {mission_data['autocontinue']}")
        
        if 'target_system' in mission_data:
            print(f"   🎯 Target: System={mission_data['target_system']}, Component={mission_data['target_component']}")
        
        print("-" * 80)
        print()

    def save_mission_to_json(self, filename="mission.json"):
        """Salva a missão atual em um arquivo JSON."""
        try:
            # Converter para lista ordenada por seq
            mission_list = []
            for seq in sorted(self.mission_items.keys()):
                mission_list.append(self.mission_items[seq])
            
            with open(filename, 'w') as f:
                json.dump(mission_list, f, indent=4)
            print(f"   💾 Missão salva/atualizada em '{filename}'")
        except Exception as e:
            print(f"   ❌ Erro ao salvar missão em JSON: {e}")
    
    def listen(self):
        """
        Loop principal para escutar mensagens MAVLink.
        """
        if not self.master:
            print("❌ Não conectado! Execute connect() primeiro.")
            return False
        
        try:
            print("🔄 Loop de escuta iniciado...")
            print()
            
            while True:
                # Receber mensagem MAVLink (timeout de 1 segundo)
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                
                # ===== MENSAGENS DE CONTAGEM DE MISSÃO =====
                if msg_type == 'MISSION_COUNT':
                    self.last_mission_count = msg.count
                    print(f"📊 {BOLD}MISSION_COUNT:{RESET} {msg.count} itens na missão")
                    print(f"   🎯 Target: System={msg.target_system}, Component={msg.target_component}")
                    
                    # Verificar se é resposta para nós (target_system é nosso ID de GCS)
                    is_for_us = (msg.target_system == self.master.source_system or 
                                 msg.target_system == 254 or msg.target_system == 255)
                    
                    if is_for_us and msg.count > 0:
                        print(f"   📥 Resposta à nossa solicitação - baixando {msg.count} itens...")
                    else:
                        print(f"   🔄 Missão sendo carregada (upload do QGC)...")
                    
                    print("-" * 80)
                    print()
                    
                    # Limpar cache de itens anteriores e contadores
                    self.mission_items.clear()
                    self.last_current_item = -1
                    self.last_reached_item = -1
                    
                    # Se for resposta para nós, solicitar cada item individualmente
                    if is_for_us and msg.count > 0:
                        for seq in range(msg.count):
                            try:
                                self.master.mav.mission_request_int_send(
                                    self.master.target_system,
                                    self.master.target_component,
                                    seq,
                                    mission_type=0  # MAV_MISSION_TYPE_MISSION
                                )
                                time.sleep(0.01)  # Pequeno delay para não sobrecarregar
                            except Exception as e:
                                print(f"   ⚠️  Erro ao solicitar item {seq}: {e}")
                        print(f"   ✅ Todas as {msg.count} solicitações enviadas")
                        print()
                
                # ===== REQUISIÇÕES DE ITENS DE MISSÃO =====
                elif msg_type in ['MISSION_REQUEST', 'MISSION_REQUEST_INT']:
                    print(f"📥 {msg_type}: Solicitando item {msg.seq} de {self.last_mission_count}")
                
                # ===== ITENS DE MISSÃO (O QUE QUEREMOS!) =====
                elif msg_type in ['MISSION_ITEM', 'MISSION_ITEM_INT']:
                    # Processar comando de missão
                    mission_data = self.format_mission_item(msg)
                    
                    # Armazenar no cache temporário
                    self.mission_items[msg.seq] = mission_data
                    
                    # Armazenar no dicionário principal
                    key = f"cmd_{mission_data['seq']:03d}_{int(time.time())}"
                    self.mission_commands[key] = mission_data
                    self.command_counter += 1
                    
                    # Imprimir informações
                    self.print_mission_command(mission_data)
                    
                    # Mostrar progresso se estiver baixando múltiplos itens
                    if len(self.mission_items) < self.last_mission_count:
                        print(f"   💾 Armazenado no cache: {len(self.mission_items)}/{self.last_mission_count}")
                        print()
                    elif len(self.mission_items) == self.last_mission_count:
                        print(f"   {GREEN}✅ Cache completo! Todos os {self.last_mission_count} comandos armazenados.{RESET}")
                        self.save_mission_to_json()
                        print()
                
                # ===== ACKNOWLEDGMENT DE MISSÃO =====
                elif msg_type == 'MISSION_ACK':
                    ack_result = {
                        0: "✅ MAV_MISSION_ACCEPTED",
                        1: "❌ MAV_MISSION_ERROR",
                        2: "❌ MAV_MISSION_UNSUPPORTED_FRAME",
                        3: "❌ MAV_MISSION_UNSUPPORTED",
                        4: "❌ MAV_MISSION_NO_SPACE",
                        5: "❌ MAV_MISSION_INVALID",
                        6: "❌ MAV_MISSION_INVALID_PARAM1",
                        7: "❌ MAV_MISSION_INVALID_PARAM2",
                        8: "❌ MAV_MISSION_INVALID_PARAM3",
                        9: "❌ MAV_MISSION_INVALID_PARAM4",
                        10: "❌ MAV_MISSION_INVALID_PARAM5_X",
                        11: "❌ MAV_MISSION_INVALID_PARAM6_Y",
                        12: "❌ MAV_MISSION_INVALID_PARAM7",
                        13: "❌ MAV_MISSION_INVALID_SEQUENCE",
                        14: "❌ MAV_MISSION_DENIED",
                    }.get(msg.type, f"❓ UNKNOWN_{msg.type}")
                    
                    print(f"{BOLD}📬 MISSION_ACK:{RESET} {ack_result}")
                    print(f"   🎯 Target: System={msg.target_system}, Component={msg.target_component}")
                    
                    if msg.type == 0:  # MAV_MISSION_ACCEPTED
                        items_cached = len(self.mission_items)
                        print(f"   {GREEN}✅ Missão aceita com sucesso!{RESET}")
                        
                        # Se não temos todos os itens no cache, solicitar novamente
                        if items_cached < self.last_mission_count:
                            print(f"   ⚠️  Cache incompleto: {items_cached}/{self.last_mission_count} itens")
                            print(f"   📥 Solicitando lista completa do veículo...")
                            time.sleep(0.5)  # Aguardar um pouco antes de solicitar
                            
                            # Solicitar a lista completa
                            try:
                                self.master.mav.mission_request_list_send(
                                    self.master.target_system,
                                    self.master.target_component,
                                    mission_type=0  # MAV_MISSION_TYPE_MISSION
                                )
                            except Exception as e:
                                print(f"   ⚠️  Erro ao solicitar lista: {e}")
                        else:
                            print(f"   💾 Cache completo: {items_cached} comandos armazenados")
                            self.save_mission_to_json()
                    
                    print("=" * 80)
                    print()
                
                # ===== ITEM ATUAL DA MISSÃO =====
                elif msg_type == 'MISSION_CURRENT':
                    # Só mostrar quando o item mudar (evitar spam)
                    if msg.seq != self.last_current_item:
                        
                        # Verificar se pulamos itens (comandos DO executados instantaneamente)
                        if self.last_current_item != -1 and msg.seq > self.last_current_item + 1:
                            for skipped_seq in range(self.last_current_item + 1, msg.seq):
                                if skipped_seq in self.mission_items:
                                    item = self.mission_items[skipped_seq]
                                    cmd_id = item['command_id']
                                    
                                    marker = ""
                                    if cmd_id == 2000:
                                        marker = f" {BOLD}{YELLOW}📸 (FOTO){RESET}"
                                    elif cmd_id == 205:
                                        marker = f" {BOLD}{CYAN}🧭 (GIMBAL){RESET}"
                                    print(f"⚡ {BOLD}Execução Rápida Detectada:{RESET} Item {skipped_seq} {GREEN}{item['command_name']}{RESET} (ID: {cmd_id}){marker}")

                        self.last_current_item = msg.seq
                        
                        print(f"🎯 {BOLD}MISSION_CURRENT:{RESET} Executando item {msg.seq}")
                        if msg.seq in self.mission_items:
                            item = self.mission_items[msg.seq]
                            cmd_id = item['command_id']
                            marker = ""
                            if cmd_id == 2000:
                                marker = f" {BOLD}{YELLOW}📸 (FOTO){RESET}"
                            elif cmd_id == 205:
                                marker = f" {BOLD}{CYAN}🧭 (GIMBAL){RESET}"
                            print(f"   → {GREEN}{item['command_name']}{RESET} (ID: {cmd_id}){marker}")
                            print(f"   {YELLOW}📋 Parâmetros do Item {msg.seq}:{RESET}")
                            print(f"      • param1: {item['params']['param1']:.6f}")
                            print(f"      • param2: {item['params']['param2']:.6f}")
                            print(f"      • param3: {item['params']['param3']:.6f}")
                            print(f"      • param4: {item['params']['param4']:.6f}")
                            
                            if 'latitude' in item and 'longitude' in item:
                                if item['latitude'] != 0 or item['longitude'] != 0:
                                    print(f"   {CYAN}📍 Coordenadas:{RESET}")
                                    print(f"      • Latitude:  {item['latitude']:11.7f}°")
                                    print(f"      • Longitude: {item['longitude']:11.7f}°")
                                    print(f"      • Altitude:  {item['altitude']:8.2f} m")
                            
                            print(f"   ⚙️  Frame: {item['frame']} | AutoContinue: {item['autocontinue']}")
                        else:
                            print(f"   ⚠️  Detalhes do item {msg.seq} não disponíveis no cache")
                        print("-" * 80)
                        print()
                
                # ===== ITEM DE MISSÃO ALCANÇADO =====
                elif msg_type == 'MISSION_ITEM_REACHED':
                    # Só mostrar quando for um item novo alcançado (evitar duplicatas)
                    if msg.seq != self.last_reached_item:
                        self.last_reached_item = msg.seq
                        
                        print(f"✅ {BOLD}{GREEN}MISSION_ITEM_REACHED:{RESET} Item {msg.seq} alcançado!")
                        if msg.seq in self.mission_items:
                            item = self.mission_items[msg.seq]
                            print(f"   → {GREEN}{item['command_name']}{RESET} (ID: {item['command_id']}) {GREEN}✓ Completado{RESET}")
                            print(f"   {YELLOW}📋 Parâmetros do Item {msg.seq}:{RESET}")
                            print(f"      • param1: {item['params']['param1']:.6f}")
                            print(f"      • param2: {item['params']['param2']:.6f}")
                            print(f"      • param3: {item['params']['param3']:.6f}")
                            print(f"      • param4: {item['params']['param4']:.6f}")

                            if 'latitude' in item and 'longitude' in item:
                                if item['latitude'] != 0 or item['longitude'] != 0:
                                    print(f"   {CYAN}📍 Coordenadas alcançadas:{RESET}")
                                    print(f"      • Latitude:  {item['latitude']:11.7f}°")
                                    print(f"      • Longitude: {item['longitude']:11.7f}°")
                                    print(f"      • Altitude:  {item['altitude']:8.2f} m")

                            print(f"   ⚙️  Frame: {item['frame']} | AutoContinue: {item['autocontinue']}")
                        else:
                            print(f"   ⚠️  Detalhes do item {msg.seq} não disponíveis no cache")

                        print("-" * 80)
                        print()
                
                # ===== HEARTBEAT (monitorar mudanças de modo) =====
                elif msg_type == 'HEARTBEAT':
                    try:
                        # Tentar extrair modo legível do heartbeat
                        try:
                            mode_str = mavutil.mode_string_v10(msg)
                        except Exception:
                            mode_str = None

                        if mode_str is None:
                            # fallback: usar base_mode/custom_mode
                            mode_str = f"base:{getattr(msg, 'base_mode', '?')} custom:{getattr(msg, 'custom_mode', '?')}"

                        if mode_str != self.last_mode:
                            print(f"🔔 HEARTBEAT: Modo mudou -> {mode_str}")
                            self.last_mode = mode_str
                    except Exception:
                        pass
                
                # ===== OUTRAS MENSAGENS DE DEPURAÇÃO =====
                # Descomente a linha abaixo para ver TODAS as mensagens
                # else:
                #     print(f"[DEBUG] {msg_type}: {msg}")
                    
        except KeyboardInterrupt:
            print()
            print("=" * 80)
            print("⏹️  Interrompido pelo usuário")
            self.print_summary()
            return True
        except Exception as e:
            print(f"❌ Erro durante escuta: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """
        Imprime um resumo dos comandos capturados.
        """
        print("=" * 80)
        print("📊 RESUMO DOS COMANDOS CAPTURADOS")
        print("=" * 80)
        print(f"Total de comandos recebidos: {self.command_counter}")
        print(f"Comandos únicos no dicionário: {len(self.mission_commands)}")
        print()
        
        if self.mission_commands:
            print("📋 Lista de comandos capturados:")
            print()
            for key, data in self.mission_commands.items():
                coords = ""
                if 'latitude' in data and 'longitude' in data:
                    if data['latitude'] != 0 or data['longitude'] != 0:
                        coords = f" @ ({data['latitude']:.6f}, {data['longitude']:.6f})"
                print(f"  • {key}:")
                print(f"      Seq {data['seq']:02d}: {data['command_name']}{coords}")
            
            print()
            print("💾 Dicionário completo salvo em: listener.mission_commands")
        else:
            print("⚠️  Nenhum comando de missão foi capturado.")
            print()
            print("💡 Possíveis razões:")
            print("   1. Nenhuma missão foi enviada ainda")
            print("   2. Porta incorreta (tente: python3 Operation_Mavlink.py 14550)")
            print("   3. O veículo ainda não está conectado")
        
        print("=" * 80)
    
    def get_mission_commands(self):
        """
        Retorna o dicionário com todos os comandos capturados.
        
        Returns:
            Dicionário com comandos de missão
        """
        return self.mission_commands


def main():
    """
    Função principal.
    """
    # Verificar argumentos de linha de comando
    port = 14551  # Porta padrão (MAVROS do primeiro rover)
    use_udpin = False
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"💡 Usando porta especificada: {port}")
        except ValueError:
            print(f"❌ Porta inválida: {sys.argv[1]}")
            print("Uso: python3 Operation_Mavlink.py [porta]")
            sys.exit(1)
    
    # Se porta for 14552, usar modo udpin (servidor)
    if port == 14552:
        use_udpin = True
    
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 24 + "MAVLINK MISSION LISTENER" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Criar listener
    listener = MavlinkMissionListener(port=port, use_udpin=use_udpin)
    
    # Conectar
    if not listener.connect():
        print("❌ Falha na conexão. Encerrando.")
        sys.exit(1)
    
    # Iniciar escuta
    listener.listen()
    
    # Retornar dicionário de comandos (útil se importado como módulo)
    return listener.get_mission_commands()


# Definições para uso das cores ANSI (globais para uso no módulo)
BOLD = '\033[1m'
RESET = '\033[0m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BLUE = '\033[94m'


if __name__ == "__main__":
    main()
