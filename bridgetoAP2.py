from missionmanager import  MissionManager


# Exemplo de uso
# nao pode ter se conectado no qground por enquanto. o link de comunicacao tem que estar livre. vou tentar fazer outro link out
# o link de comunicacao com o qgc tem que ser o 14550 - ou seja, diferente de todos os outros robos.
if __name__ == "__main__":
    # Coordenadas GPS da missão

    # ATENCAO - por algum motivo que ainda nao descobri, o primeiro ponto tem que ser enviado de forma duplicada.

    mission_points_2 = [
        {"id": 0, "lat": -3.12316325, "lon": -41.76546074},
        {"id": 1, "lat": -3.12316325, "lon": -41.76546074},
        {"id": 2, "lat": -3.12404350, "lon": -41.76555384},
        {"id": 3, "lat": -3.12416325, "lon": -41.76566074},
        {"id": 4, "lat": -3.12404350, "lon": -41.76545384},
        {"id": 5, "lat": -3.12401445, "lon": -41.76353468},
        {"id": 6, "lat": -3.12205639, "lon": -41.76364815},
        {"id": 7, "lat": -3.12204477, "lon": -41.76397692},
        {"id": 8, "lat": -3.12256770, "lon": -41.76403220},
        {"id": 9, "lat": -3.12255317, "lon": -41.76507669},
        {"id": 10, "lat": -3.12257641, "lon": -41.76543747}
    ]

    mission_points_1 = [
        {"id": 0, "lat": -3.12257641, "lon": -41.76543747},
        {"id": 1, "lat": -3.12257641, "lon": -41.76543747},
        {"id": 2, "lat": -3.12255317, "lon": -41.76507669},
        {"id": 3, "lat": -3.12256770, "lon": -41.76403220},
        {"id": 4, "lat": -3.12204477, "lon": -41.76397692},
        {"id": 5, "lat": -3.12205639, "lon": -41.76364815},
        {"id": 6, "lat": -3.12401445, "lon": -41.76353468},
        {"id": 7, "lat": -3.12404350, "lon": -41.76555384},
        {"id": 8, "lat": -3.12316325, "lon": -41.76546074}
    ]

    ALT = 2.0  # Altitude padrão

    # Configuração dos robôs
    # robots = [
    #     {"channel": "udp:0.0.0.0:14551", "mission": mission_points_1, "source_system":201 },
    #     {"channel": "udp:0.0.0.0:14552", "mission": mission_points_2, "source_system":202 }
    # ]

    # 0.0.0.0 habilita para receber de qq IP
    robots = [
        {"channel": "udp:0.0.0.0:14551", "mission": mission_points_1, "source_system":1 },
        {"channel": "udp:0.0.0.0:14552", "mission": mission_points_2, "source_system":2 }
    ]

    managers = []

    # Configuração das missões
    for robot in robots:
        print(f"\n🛠 Configurando robô em {robot['channel']}")
        manager = MissionManager(
            udp_channel=robot["channel"],
            source_system=robot["source_system"],
            timeout=15,
            max_attempts=50
        )

        if manager.connect():
            if manager.upload_mission(robot["mission"]):
                managers.append(manager)
                continue

        print(f"❌ Falha na configuração do robô em {robot['channel']}")

    # Iniciar missões
    print("\n🚀 Iniciando missões...")
    # for manager in managers:
    #     if not manager.arm_and_start():
    #         print(f"❌ Falha ao iniciar missão para robô em {manager.udp_channel}")

    print("\n✅ Processo concluído")