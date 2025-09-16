from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
from PlanejadorHeterogeneoIntegrado import run_planner, montar_missoes_por_robo, otimizarpontos, retorna_pontos_passagem, ajustar_missoes_deltas, build_mission_points_from_path_gps, extract_path_gps, load_label2gps, extract_path_gps_from_obp
from missionmanagerunificado import MissionManager

from plotutils import PlotUtils
from collections import defaultdict
from ajusteplanilha import AjustePlanilha
from tspOptimization import FixedTaskPlanner
from aabbutils import AABBUtils
from collections import OrderedDict, defaultdict
import math
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import to_agraph
from pygraphviz import AGraph
import networkx as nx
import matplotlib.image as mpimg
import os
import subprocess



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


    tx1, ty1 = 4.088943322324063, -27.844607266401578
    tx2, ty2 = 50.92951784451285, 48.20302033360754

    robot_names = [r["name"] for r in robots]
    mission_execution_config = {m: robot_names[:] for m in missions}


    G_mapa = SegmentUtils.load_graph_json(graph_path)


    saida = run_planner(
        graph_path,
        observation_points_json_path,
        file_path_parametros,
        missions,
        mission_execution_config,
        robot_positions_xy={"R1":(tx1+deltax_m,ty1+deltay_m), "R2":(tx2-deltax_m,ty2-deltay_m)},   # ou {"R1":(x1,y1), "R2":(x2,y2)}
        do_plots=False
    )

    # exemplo: imprimir rotas label→label
    for r, rota in saida["rotas_otimas_por_robo"].items():
        print(f"{r}: {' -> '.join(rota)}")

    # PlotUtils.plot_rotas_grafo(Greduced_map,rotas_otimas_por_robo)
    PlotUtils.plot_rotas_reais(G_mapa,saida["rotas_otimas_por_robo"],saida["point_mission_positions"])

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
