from missionmanagerunificado import MissionManager
from PlanejadorHeterogeneoIntegrado import run_planner, montar_missoes_por_robo, otimizarpontos, retorna_pontos_passagem, ajustar_missoes_deltas, build_mission_points_from_path_gps, extract_path_gps, load_label2gps, extract_path_gps_from_obp
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
    # graph_path = "./jsons/graph9_new.json"
    # observation_points_json_path = "./jsons/obp_6.json"
    deltax_m = -2
    deltay_m = -12.0

    graph_path = "./jsons/graph9d_new.json"
    observation_points_json_path = "./jsons/obpc_7.json"
    # missions = ['b_busip4',  'ls_tpc1']

    missions = ['b_busip20', 'b_busip21', 'b_busip25', 'b_busip33', 'cd_reator3', 'cd_reator4' ]

    # missions = ['b_busip4', 'ef_reator1', 'ls_pr4', 'ef_reator10', 'ef_disjuntor6', 'ls_tpc1']
    robots = [
        {'name': 'R1', "channel": "udp:0.0.0.0:14551",  "source_system":1 },
        {'name': 'R2', "channel": "udp:0.0.0.0:14561",  "source_system":2 }
    ]

    robot_names = [r["name"] for r in robots]
    mission_execution_config = {m: robot_names[:] for m in missions}

    # mission_execution_config = {
    #     "b_busip4": ["R1", "R2"],
    #     "ef_reator1": ["R1", "R2"],
    #     "ls_pr4": ["R1", "R2"],
    # }


    # 0.0.0.0 habilita para receber de qq IP


    G_mapa = SegmentUtils.load_graph_json(graph_path)

    # pegue a referência de conversão (como já faz no seu código)
    ref = MissionManager.read_parametros_conversao_lat_lon(file_path_parametros)
    lat_ref, lon_ref = ref["lat_ref"], ref["lon_ref"]

    mm = MissionManager(robots=robots)
    mm.connect_all()
    mm.force_gps_stream(rate_hz=5.0)  # todos conectados
    # mm.force_gps_stream(rate_hz=5.0, robot='R1')

    # Ou exigir de todos:
    all_states = mm.new_wait_for_position(timeout=25.0, require_all=True)

    # Configuração das missões
    Ipos = {}
    for robot in robots:
        st = mm.new_wait_for_position(robot=robot["name"], timeout=5.0)
        home = mm.set_home_to_current(robot=robot["name"])
        print(all_states[robot['name']])

    lat1, lon1 = get_latlon(all_states, "R1")
    lat2, lon2 = get_latlon(all_states, "R2")


    # Define posições iniciais reais dos robôs com base em coordenadas (x, y)
    tx1, ty1 = MissionManager.gps_to_xy(lat1, lon1, ref["lat_ref"], ref["lon_ref"])
    tx2, ty2 = MissionManager.gps_to_xy(lat2, lon2, ref["lat_ref"], ref["lon_ref"])

    print((tx1,ty1))
    print((tx2, ty2))




    saida = run_planner(
        graph_path,
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

    pontos_vistoria = []
    for robo, missions in saida["missoes_completas"].items():
        for m in missions:
            for t in m["tasks"]:
                pontos_vistoria.append(t["point"])

    # 2) Chama a função corretamente e materializa o gerador
    missoes_completas = list(retorna_pontos_passagem(
        G_mapa,  # grafo completo com 'pos' e 'weight'
        saida["rotas_otimas_por_robo"],  # dict: robo -> [labels na ordem]
        pontos_vistoria  # lista de labels que são marcos de vistoria
    ))

    # (Opcional) Exemplo de uso do retorno
    for item in missoes_completas:  # 1 por robô
        info = item[0]  # a função rende uma lista com um dict dentro
        robo = info["robo"]
        caminho_completo, pts_vistoria, pts_passagem = info["rotas_detalhadas"]
        print(
            f"[{robo}] nós no caminho: {len(caminho_completo)} | vistoria: {len(pts_vistoria)} | passagem: {len(pts_passagem)}")

    missoes_por_robo = montar_missoes_por_robo(
        missoes_completas=missoes_completas,
        G_mapa=G_mapa,
        observation_points_json_path=observation_points_json_path,
        lat_ref=lat_ref, lon_ref=lon_ref,
        duplicate_first=True,  # se quiser repetir o 1º ponto
        hold_vistoria=2.0,
        hold_passagem=0.0,
        MissionManager=MissionManager
    )

    missoes_por_robo_aj = ajustar_missoes_deltas(missoes_por_robo, dx_m=-deltax_m, dy_m=deltay_m)

    missoes_otimizadas = otimizarpontos(missoes_por_robo_aj, tol_ct_m=0.10, preserve_loop_closure=True,
                                        renumber_ids=True)

    # Configuração das missões
    for robot in robots:
        print(f"\n🛠 enviando missao ao robô em {robot['channel']}")

        if mm.connected:
            mm.upload_mission(missoes_otimizadas[robot['name']], robot = robot['name'])




    # Iniciar missões
    print("\n🚀 Iniciando missões...")
    # for manager in managers:
    #     if not manager.arm_and_start():
    #         print(f"❌ Falha ao iniciar missão para robô em {manager.udp_channel}")

    print("\n✅ Processo concluído")