"""
Módulo para gerenciamento de robôs e mapeamento de identificadores
"""

# Mapeamento de rovers do banco Django para identificadores do mission_server
ROVER_ID_MAPPING = {
    # Mapear identifiers do banco para IDs simples do mission_server
    "Rover_Beta": "R1",
    "Rover_Charlie": "R2", 
    "Rover_Delta": "R3"
}

# Mapeamento reverso para logs e debug
REVERSE_ROVER_MAPPING = {v: k for k, v in ROVER_ID_MAPPING.items()}

def map_rover_identifier_to_mission_id(rover_identifier):
    """
    Mapeia identifier do banco Django para ID do mission_server
    
    Args:
        rover_identifier: Identifier do banco (ex: "Rover_Beta")
    
    Returns:
        str: ID mapeado (ex: "R1") ou identifier original se não encontrado
    """
    mapped_id = ROVER_ID_MAPPING.get(rover_identifier, rover_identifier)
    print(f"🔄 Mapeamento rover: {rover_identifier} -> {mapped_id}")
    return mapped_id

def create_rover_config_from_frontend_data(rover_data, index):
    """
    Cria configuração de rover para o mission_server a partir dos dados do frontend
    
    Args:
        rover_data: Dados do rover do frontend
        index: Índice do rover na lista
    
    Returns:
        dict: Configuração do rover para mission_server
    """
    # Obter identifier original do banco
    original_identifier = rover_data.get('identifier', f'R{index+1}')
    
    # Mapear para ID do mission_server
    mission_id = map_rover_identifier_to_mission_id(original_identifier)
    
    # Configuração para mission_server
    config = {
        'name': mission_id,  # ID mapeado (R1, R2, R3, etc.)
        'channel': f"udp:0.0.0.0:{14551 + (index * 100)}",  # 14551, 14651, 14751, etc.
        'source_system': index + 1,
        # Dados originais para referência
        'original_identifier': original_identifier,
        'db_id': rover_data.get('id'),
        'display_name': rover_data.get('name', mission_id),
        'model': rover_data.get('model', 'Unknown')
    }
    
    print(f"🤖 Rover configurado:")
    print(f"   • DB Identifier: {original_identifier}")
    print(f"   • Mission ID: {mission_id}")
    print(f"   • Display Name: {config['display_name']}")
    print(f"   • Channel: {config['channel']}")
    
    return config

def create_position_monitoring_manager():
    """
    Cria um MissionManager apenas para monitoramento contínuo de posições
    Conecta aos robôs padrão sem executar missões
    """
    from missionmanagerunificado import MissionManager
    
    # Configuração padrão dos robôs para monitoramento
    default_robots = [
        {
            'name': 'R1',
            'channel': 'udp:0.0.0.0:14551',
            'source_system': 1
        },
        {
            'name': 'R2', 
            'channel': 'udp:0.0.0.0:14651',
            'source_system': 2
        },
        {
            'name': 'R3',
            'channel': 'udp:0.0.0.0:14751', 
            'source_system': 3
        }
    ]
    
    try:
        print("🔗 Criando MissionManager para monitoramento de posições...")
        position_monitoring_manager = MissionManager(
            robots=default_robots,
            timeout=2.0,
            max_attempts=3
        )
        
        # Tentar conectar aos robôs disponíveis
        connected = position_monitoring_manager.connect_all()
        if connected:
            print(f"✅ Conectado a {len(position_monitoring_manager.connected)} robô(s) para monitoramento")
            print(f"   Robôs conectados: {list(position_monitoring_manager.connected)}")
        else:
            print("⚠️ Nenhum robô conectado para monitoramento")
            
        return position_monitoring_manager
        
    except Exception as e:
        print(f"❌ Erro ao criar MissionManager para monitoramento: {e}")
        return None
