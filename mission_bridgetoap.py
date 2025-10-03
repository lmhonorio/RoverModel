from missionmanagerunificado import MissionManager
from PlanejadorHeterogeneoIntegrado import run_planner, montar_missoes_por_robo, otimizarpontos, retorna_pontos_passagem, ajustar_missoes_deltas, build_mission_points_from_path_gps, extract_path_gps, load_label2gps, extract_path_gps_from_obp
from segmentutils import SegmentUtils  # para carregar o grafo, se precisar
from typing import List, Dict, Tuple, Optional

def get_latlon(all_states, robot):
    d = all_states.get(robot)
    if d and "lat" in d and "lon" in d:
        return float(d["lat"]), float(d["lon"])
    return None, None


def preparar_e_enviar_missoes(
    robots: List[Dict],
    missions: List[str],
    graph_path: str,
    observation_points_json_path: str,
    file_path_parametros: str,
    deltax_m: float,
    deltay_m: float,
    *,
    gps_rate_hz: float = 5.0,
    wait_timeout_s: float = 25.0,
    wait_each_timeout_s: float = 5.0,
    duplicate_first: bool = True,
    hold_vistoria_s: float = 2.0,
    hold_passagem_s: float = 0.0,
    tol_ct_m: float = 0.10,
    preserve_loop_closure: bool = True,
    renumber_ids: bool = True,
    do_plots: bool = False,
) -> Dict:
    """
    Constrói a mission_execution_config, conecta aos robôs, lê posições reais,
    roda o planner, gera missões otimizadas e faz o upload para cada robô.

    Retorna um dicionário com objetos intermediários úteis.
    Requer no escopo/imports: SegmentUtils, MissionManager, run_planner,
    get_latlon, retorna_pontos_passagem, montar_missoes_por_robo,
    ajustar_missoes_deltas, otimizarpontos.
    """

    # 1) mission_execution_config a partir de robots e missions
    robot_names = [r["name"] for r in robots]
    mission_execution_config = {m: robot_names[:] for m in missions}

    # 2) Carrega o grafo e referência de conversão
    G_mapa = SegmentUtils.load_graph_json(graph_path)
    ref = MissionManager.read_parametros_conversao_lat_lon(file_path_parametros)
    lat_ref, lon_ref = ref["lat_ref"], ref["lon_ref"]

    # 3) Conecta e força stream de GPS
    mm = MissionManager(robots=robots)
    mm.connect_all()
    mm.force_gps_stream(rate_hz=gps_rate_hz)

    # Espera todos terem posição
    all_states = mm.new_wait_for_position(timeout=wait_timeout_s, require_all=True)

    # Define home e mostra estados individuais
    for robot in robots:
        _ = mm.new_wait_for_position(robot=robot["name"], timeout=wait_each_timeout_s)
        _ = mm.set_home_to_current(robot=robot["name"])
        print(all_states[robot["name"]])

    # 4) Lê lat/lon atuais dos dois primeiros robôs (R1/R2) e converte para x,y
    #    (ajuste se houver mais robôs)
    lat1, lon1 = get_latlon(all_states, "R1")
    lat2, lon2 = get_latlon(all_states, "R2")

    tx1, ty1 = MissionManager.gps_to_xy(lat1, lon1, lat_ref, lon_ref)
    tx2, ty2 = MissionManager.gps_to_xy(lat2, lon2, lat_ref, lon_ref)

    print((tx1, ty1))
    print((tx2, ty2))

    # 5) Executa o planner com as posições reais (corrigidas pelos deltas)
    saida = run_planner(
        graph_path,
        observation_points_json_path,
        file_path_parametros,
        missions,
        mission_execution_config,
        robot_positions_xy={
            "R1": (tx1 - deltax_m, ty1 - deltay_m),
            "R2": (tx2 - deltax_m, ty2 - deltay_m),
        },
        do_plots=do_plots,
    )

    # 6) Log de rotas ótimas por robô
    for r, rota in saida["rotas_otimas_por_robo"].items():
        print(f"{r}: {' -> '.join(rota)}")

    # 7) Extrai pontos de vistoria
    pontos_vistoria = []
    for robo, mis in saida["missoes_completas"].items():
        for m in mis:
            for t in m["tasks"]:
                pontos_vistoria.append(t["point"])

    # 8) Materializa rotas detalhadas (vistoria/passagem)
    missoes_completas = list(
        retorna_pontos_passagem(
            G_mapa,
            saida["rotas_otimas_por_robo"],
            pontos_vistoria,
        )
    )

    for item in missoes_completas:
        info = item[0]
        robo = info["robo"]
        caminho_completo, pts_vistoria, pts_passagem = info["rotas_detalhadas"]
        print(f"[{robo}] nós: {len(caminho_completo)} | vistoria: {len(pts_vistoria)} | passagem: {len(pts_passagem)}")

    # 9) Constrói missões (lat/lon) por robô
    missoes_por_robo = montar_missoes_por_robo(
        missoes_completas=missoes_completas,
        G_mapa=G_mapa,
        observation_points_json_path=observation_points_json_path,
        lat_ref=lat_ref,
        lon_ref=lon_ref,
        duplicate_first=duplicate_first,
        hold_vistoria=hold_vistoria_s,
        hold_passagem=hold_passagem_s,
        MissionManager=MissionManager,
    )

    # 10) Ajusta deltas e otimiza waypoints/IDs
    missoes_por_robo_aj = ajustar_missoes_deltas(
        missoes_por_robo,
        dx_m=-deltax_m,
        dy_m=deltay_m,
    )

    missoes_otimizadas = otimizarpontos(
        missoes_por_robo_aj,
        tol_ct_m=tol_ct_m,
        preserve_loop_closure=preserve_loop_closure,
        renumber_ids=renumber_ids,
    )

    # 11) Upload das missões para cada robô
    for robot in robots:
        print(f"\n🛠 enviando missão ao robô em {robot['channel']}")
        if mm.connected:
            mm.upload_mission(missoes_otimizadas[robot["name"]], robot=robot["name"])

    # 12) Retorno consolidado
    return {
        "mission_execution_config": mission_execution_config,
        "G_mapa": G_mapa,
        "lat_ref": lat_ref,
        "lon_ref": lon_ref,
        "all_states": all_states,
        "planner_saida": saida,
        "missoes_completas": missoes_completas,
        "missoes_por_robo": missoes_por_robo,
        "missoes_otimizadas": missoes_otimizadas,
        "mission_manager":mm,
    }

# Exemplo de uso
# nao pode ter se conectado no qground por enquanto. o link de comunicacao tem que estar livre. vou tentar fazer outro link out
# o link de comunicacao com o qgc tem que ser o 14550 - ou seja, diferente de todos os outros robos.
if __name__ == "__main__":
    # Coordenadas GPS da missão
    file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Ajuste se precisar
    # graph_path = "./jsons/graph9_new.json"
    # observation_points_json_path = "./jsons/obp_6.json"
    # deltax_m = -2
    # deltay_m = -12.0

    deltax_m = -4
    deltay_m = -4.0

    graph_path = "./jsons/graph9d_new.json"
    observation_points_json_path = "./jsons/obpc_7.json"
    # missions = ['b_busip4',  'ls_tpc1']

    missions = ['b_busip20', 'b_busip21', 'b_busip25', 'b_busip33', 'cd_reator3', 'cd_reator4' ]

    # missions = ['b_busip4', 'ef_reator1', 'ls_pr4', 'ef_reator10', 'ef_disjuntor6', 'ls_tpc1']
    robots = [
        {'name': 'R1', "channel": "udp:0.0.0.0:14551",  "source_system":1 },
        {'name': 'R2', "channel": "udp:0.0.0.0:14561",  "source_system":2 }
    ]

    res = preparar_e_enviar_missoes(
        robots=robots,
        missions=missions,
        graph_path=graph_path,
        observation_points_json_path=observation_points_json_path,
        file_path_parametros=file_path_parametros,
        deltax_m=deltax_m,
        deltay_m=deltay_m,
    )


    mm = res["mission_manager"]


    # 5) Seta modo AUTO, arma e inicia a missão
    if not mm.arm_and_start():
        print("❌ Falha ao armar/iniciar a missão.")

    print("✅ Missão iniciada com sucesso!")








    #
    # robot_names = [r["name"] for r in robots]
    # mission_execution_config = {m: robot_names[:] for m in missions}
    #
    # G_mapa = SegmentUtils.load_graph_json(graph_path)
    #
    # # pegue a referência de conversão (como já faz no seu código)
    # ref = MissionManager.read_parametros_conversao_lat_lon(file_path_parametros)
    # lat_ref, lon_ref = ref["lat_ref"], ref["lon_ref"]
    #
    # mm = MissionManager(robots=robots)
    # mm.connect_all()
    # mm.force_gps_stream(rate_hz=5.0)  # todos conectados
    # # mm.force_gps_stream(rate_hz=5.0, robot='R1')
    #
    # # Ou exigir de todos:
    # all_states = mm.new_wait_for_position(timeout=25.0, require_all=True)
    #
    # # Configuração das missões
    # Ipos = {}
    # for robot in robots:
    #     st = mm.new_wait_for_position(robot=robot["name"], timeout=5.0)
    #     home = mm.set_home_to_current(robot=robot["name"])
    #     print(all_states[robot['name']])
    #
    # lat1, lon1 = get_latlon(all_states, "R1")
    # lat2, lon2 = get_latlon(all_states, "R2")
    #
    #
    # # Define posições iniciais reais dos robôs com base em coordenadas (x, y)
    # tx1, ty1 = MissionManager.gps_to_xy(lat1, lon1, ref["lat_ref"], ref["lon_ref"])
    # tx2, ty2 = MissionManager.gps_to_xy(lat2, lon2, ref["lat_ref"], ref["lon_ref"])
    #
    # print((tx1,ty1))
    # print((tx2, ty2))
    #
    # saida = run_planner(
    #     graph_path,
    #     observation_points_json_path,
    #     file_path_parametros,
    #     missions,
    #     mission_execution_config,
    #     # robot_positions_xy={"R1":(tx1-deltax_m,ty1-deltay_m), "R2":(tx2-deltax_m,ty2-deltay_m)},   # ou {"R1":(x1,y1), "R2":(x2,y2)}
    #     robot_positions_xy={"R1": (tx1 - deltax_m, ty1 - deltay_m), "R2": (tx2 - deltax_m, ty2 - deltay_m)},
    #     do_plots=False
    # )
    #
    # # exemplo: imprimir rotas label→label
    # for r, rota in saida["rotas_otimas_por_robo"].items():
    #     print(f"{r}: {' -> '.join(rota)}")
    #
    # pontos_vistoria = []
    # for robo, missions in saida["missoes_completas"].items():
    #     for m in missions:
    #         for t in m["tasks"]:
    #             pontos_vistoria.append(t["point"])
    #
    # # 2) Chama a função corretamente e materializa o gerador
    # missoes_completas = list(retorna_pontos_passagem(
    #     G_mapa,  # grafo completo com 'pos' e 'weight'
    #     saida["rotas_otimas_por_robo"],  # dict: robo -> [labels na ordem]
    #     pontos_vistoria  # lista de labels que são marcos de vistoria
    # ))
    #
    # # (Opcional) Exemplo de uso do retorno
    # for item in missoes_completas:  # 1 por robô
    #     info = item[0]  # a função rende uma lista com um dict dentro
    #     robo = info["robo"]
    #     caminho_completo, pts_vistoria, pts_passagem = info["rotas_detalhadas"]
    #     print(
    #         f"[{robo}] nós no caminho: {len(caminho_completo)} | vistoria: {len(pts_vistoria)} | passagem: {len(pts_passagem)}")
    #
    # missoes_por_robo = montar_missoes_por_robo(
    #     missoes_completas=missoes_completas,
    #     G_mapa=G_mapa,
    #     observation_points_json_path=observation_points_json_path,
    #     lat_ref=lat_ref, lon_ref=lon_ref,
    #     duplicate_first=True,  # se quiser repetir o 1º ponto
    #     hold_vistoria=2.0,
    #     hold_passagem=0.0,
    #     MissionManager=MissionManager
    # )
    #
    # missoes_por_robo_aj = ajustar_missoes_deltas(missoes_por_robo, dx_m=-deltax_m, dy_m=deltay_m)
    #
    # missoes_otimizadas = otimizarpontos(missoes_por_robo_aj, tol_ct_m=0.10, preserve_loop_closure=True,
    #                                     renumber_ids=True)
    #
    # # Configuração das missões
    # for robot in robots:
    #     print(f"\n🛠 enviando missao ao robô em {robot['channel']}")
    #
    #     if mm.connected:
    #         mm.upload_mission(missoes_otimizadas[robot['name']], robot = robot['name'])
    #
    #
