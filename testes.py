from astropy.wcs.docstrings import coord

from roverclass import ObstacleLoader
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from aabbutils import AABBUtils
from networkx.drawing.nx_agraph import to_agraph
import math
import matplotlib.pyplot as plt
import pickle
import json
# from baseclasses import *





def gerar_mission_positions_from_json(observacao_por_obstaculo, obstaculos):
    mission_positions = {}

    for obs in obstaculos:
        pontos = observacao_por_obstaculo.get(obs, [])
        labels = [
            ponto.get("label")
            for ponto in pontos
            if ponto.get("label", "").startswith(obs + ".")
        ]
        mission_positions[obs] = labels

    return mission_positions

file_path = "./jsons/obp_6.json"
observacao_por_obstaculo  = SegmentUtils.load_observation_points_from_json(file_path)

missao = gerar_mission_positions_from_json(observacao_por_obstaculo,['b_busip4','ls_pr4','ef_reator1'])

print(missao)
# for nome, pontos in observacao_por_obstaculo.items():
#     print(f"Obstáculo {nome} com {len(pontos)} pontos de observação.")
#     for ponto in pontos:
#         print(f"  → Label: {ponto['label']}, Coordenada: {ponto['coord']}")
#
#
# todos_label = [label for label,_ in observacao_por_obstaculo.items()]

# print(todos_label)