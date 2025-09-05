from missionmanagerunificado import MissionManager
from PlanejadorHeterogeneoIntegrado import run_planner, build_mission_points_from_path_gps, extract_path_gps, load_label2gps, extract_path_gps_from_obp
from segmentutils import SegmentUtils  # para carregar o grafo, se precisar


def get_latlon(all_states, robot):
    d = all_states.get(robot)
    if d and "lat" in d and "lon" in d:
        return float(d["lat"]), float(d["lon"])
    return None, None

# Exemplo de uso
# nao pode ter se conectado no qground por enquanto. o link de comunicacao tem que estar livre. vou tentar fazer outro link out
# o link de comunicacao com o qgc tem que ser o 14550 - ou seja, diferente de todos os outros robos.
if __name__ == "__main__":
    # Coordenadas GPS da missão
    file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Ajuste se precisar
    file_path = "./jsons/graph9_new.json"
    observation_points_json_path = "./jsons/obp_6.json"
    missions = ['b_busip4', 'ef_reator1', 'ls_pr4']
    mission_execution_config = {
        "b_busip4": ["R1", "R2"],
        "ef_reator1": ["R1", "R2"],
        "ls_pr4": ["R1", "R2"],
    }


    # 0.0.0.0 habilita para receber de qq IP
    robots = [
        {'name': 'R1', "channel": "udp:0.0.0.0:14551",  "source_system":1 },
        {'name': 'R2', "channel": "udp:0.0.0.0:14561",  "source_system":2 }
    ]

    G_mapa = SegmentUtils.load_graph_json(file_path)

    # pegue a referência de conversão (como já faz no seu código)
    ref = MissionManager.read_parametros_conversao_lat_lon(file_path_parametros)
    lat_ref, lon_ref = ref["lat_ref"], ref["lon_ref"]

    mm = MissionManager(robots=robots)
    mm.connect_all()
    mm.force_gps_stream(rate_hz=5.0)  # todos conectados
    # mm.force_gps_stream(rate_hz=5.0, robot='R1')

    # Ou exigir de todos:
    all_states = mm.wait_for_position(timeout=5.0, require_all=True)

    # Configuração das missões
    for robot in robots:
        print(all_states[robot['name']])

    lat1, lon1 = get_latlon(all_states, "R1")
    lat2, lon2 = get_latlon(all_states, "R2")

    ref_lat_lon = MissionManager.read_parametros_conversao_lat_lon(file_path_parametros)

    # Define posições iniciais reais dos robôs com base em coordenadas (x, y)
    tx1, ty1 = MissionManager.gps_to_xy(lat1, lon1, ref_lat_lon['lat_ref'], ref_lat_lon['lon_ref'])
    tx2, ty2 = MissionManager.gps_to_xy(lat2, lon2, ref_lat_lon['lat_ref'], ref_lat_lon['lon_ref'])

    print((tx1,ty1))
    print((tx2, ty2))


    saida = run_planner(
        file_path,
        observation_points_json_path,
        file_path_parametros,
        missions,
        mission_execution_config,
        robot_positions_xy={"R1":(tx1,ty1), "R2":(tx2,ty2)},   # ou {"R1":(x1,y1), "R2":(x2,y2)}
        do_plots=False
    )

    # exemplo: imprimir rotas label→label
    for r, rota in saida["rotas_otimas_por_robo"].items():
        print(f"{r}: {' -> '.join(rota)}")

    label2gps = load_label2gps(observation_points_json_path)

    missoes_por_robo = {}

    for robo, tarefas in saida["missoes_completas"].items():
        missoes_por_robo[robo] = {}
        for t_idx, tarefa in enumerate(tarefas):
            path_gps = extract_path_gps_from_obp(
                tarefa,
                label2gps,
                rota_labels_fallback=saida.get("rotas_otimas_por_robo", {}).get(robo)
            )
            mission_points = build_mission_points_from_path_gps(
                path_gps,
                holds=0.0,
                default_hold=2.0,
                duplicate_first=True,
                start_id=0
            )
            missoes_por_robo[robo] = mission_points


    mission_1 = missoes_por_robo['R1']
    mission_2 = missoes_por_robo['R2']

    managers = []

    # Configuração das missões
    for robot in robots:
        print(f"\n🛠 enviando missao ao robô em {robot['channel']}")

        if mm.connected:
            mm.upload_mission(missoes_por_robo[robot['name']], robot = robot['name'])


    # # Configuração das missões
    # for robot in robots:
    #     print(f"\n🛠 Configurando robô em {robot['channel']}")
    #     manager = MissionManager(
    #         udp_channel=robot["channel"],
    #         source_system=robot["source_system"],
    #         timeout=15,
    #         max_attempts=50
    #     )
    #     if not manager.connect():
    #         print(f"❌ {robot['name']}: falha ao conectar.")
    #         continue
    #
    #     mission_list = missoes_por_robo.get(robot["name"], [])
    #     if not mission_list:
    #         print(f"ℹ️ {robot['name']}: nenhuma missão atribuída.")
    #         continue
    #
    #     print(f"🚀 Enviando {len(mission_list)} missão(ões) para {robot['name']}")
    #     for idx, mission_points in enumerate(mission_list, start=1):
    #         # sanity-check simples:
    #         if not mission_points or not isinstance(mission_points[0], dict):
    #             print(f"  ⚠️ Missão {idx} de {robot['name']} não está no formato esperado (lista de dicts).")
    #             continue
    #
    #         print(f"  ▶️ Upload missão {idx} ({len(mission_points)} WPs)")
    #         if not manager.upload_mission(mission_points):
    #             print(f"  ❌ Falha no upload da missão {idx} para {robot['name']}")
    #             break
    #
    #         # iniciar + esperar (ajuste para os métodos que sua classe tiver)
    #         if hasattr(manager, "start_mission"):
    #             manager.start_mission()
    #         elif hasattr(manager, "arm_and_start_mission"):
    #             manager.arm_and_start_mission()
    #         elif hasattr(manager, "arm_and_start"):
    #             manager.arm_and_start()
    #         if hasattr(manager, "wait_until_mission_done"): manager.wait_until_mission_done()
    #
    #     managers.append(manager)
    #
    #
    #     print(f"❌ Falha na configuração do robô em {robot['channel']}")

    # Iniciar missões
    print("\n🚀 Iniciando missões...")
    # for manager in managers:
    #     if not manager.arm_and_start():
    #         print(f"❌ Falha ao iniciar missão para robô em {manager.udp_channel}")

    print("\n✅ Processo concluído")