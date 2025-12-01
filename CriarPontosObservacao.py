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

# Ajustes visuais para os plots: aumenta fontes de títulos, labels, ticks e legendas
plt.rcParams.update({
  'figure.titlesize': 18,
  'axes.titlesize': 18,
  'axes.labelsize': 14,
  'xtick.labelsize': 12,
  'ytick.labelsize': 12,
  'legend.fontsize': 12,
  'font.size': 12,
})

# Config
# file_path = "./planilhas/obstaculos_processado6.xlsx"
file_path = "./planilhas/equipment_processado.xlsx"
sheet_name = "Parnaiba3_Transformado"
# grafo_path = "./jsons/graph9F_new.json"
grafo_path = "./jsons/graph_equipment.json"
grafo_path_geo = "./jsons/graph_equipment_geo.json" # grafo com coordenadas geograficas
# observation_path ="./jsons/obpc_6.json"
observation_path ="./jsons/obs_equipment.json"
# observation_folter = "./pontos_observacao"
observation_folter = "./pontos_observacao2"

###############################################################################
# MAIN
###############################################################################
def build_graph(file_path, sheet_name, margin=3.5, threshold=20, plotting=False):
  """Constrói o grafo final a partir do arquivo de planilha e parâmetros.

  Retorna: (G_corrigido, aabbs, observation_points, obstacles)
  Se plotting=False, evita chamadas de plot para uso em análises automatizadas.
  """
  # Carregar obstáculos
  loader = ObstacleLoader(file_path, sheet_name)
  obstacles = loader.get_obstacles()

  # AABBs
  aabbs = AABBUtils.get_aabbs(obstacles, margin, threshold)

  # Segments entre AABBs
  segments = SegmentUtils.generate_segments_between_aabbs(aabbs, 3.0)

  # Pontos de observação no perímetro
  segments, observation_points = SegmentUtils.generate_perimeter_segments_and_labeled_points(
    segments, aabbs, obstacles, threshold=5.0
  )

  # Salva os arquivos para visualizacao no excel, importacao no planner e visualizacao em gis - aqui encontra os pontos que olham para o objeto da melhor forma (melhor = perto)
  # SegmentUtils.save_observation_points_to_excel(obstacles, observation_points, 6, file_path)
  # SegmentUtils.save_observation_points_to_json(obstacles, observation_points, 6, observation_path, file_path)
  # SegmentUtils.save_observation_points_to_kml(obstacles, observation_points, 6, file_path, observation_folter, offset_lat_meters=0.0, offset_lon_meters=0.0)

  # Resolver interseções e criar grafo
  broken_segments, passage_points = SegmentUtils.resolve_segment_intersections(segments, 2)
  G_corrigido = SegmentUtils.create_graph_with_passage_points_new(
    broken_segments, passage_points, observation_points, obstacles
  )

  # Optionally plot intermediate results
  if plotting:
    PlotUtils.plot_segments_aabbs_vertices(segments, aabbs, 0.5)
    PlotUtils.plot_aabbs_obstacles_points(obstacles,aabbs,observation_points)
    PlotUtils.plot_segments_aabbs_vertices(broken_segments, aabbs, 0.5)

  return G_corrigido, aabbs, observation_points, obstacles

def main():
    padding = 15
    margin = 3.5 #1.5  #colocar esta coluna no xml para definir de forma personalizada a distancia do rover para cada objeto
    # margin = 2.5
    threshold = 20 # 30 # verifica largura e altura do aabb após o merge para permitir sobreposicao de aabbs

    # Carregar obstáculos e construir grafo final
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()

    # Usa a função auxiliar para construir o grafo (retorna também aabbs e pontos)
    G_corrigido, aabbs, observation_points, obstacles = build_graph(file_path, sheet_name, margin, threshold, plotting=True)

    # print(f"verificando ilhas....")
    # hasislands = SegmentUtils.has_islands(G_corrigido)
    # print(f"ilhas: {hasislands}")

    # if hasislands:
    #     print("🔹 Corrigindo conexões faltantes com verificação contra AABBs...")
    #     G_corrigido = SegmentUtils.fix_missing_connections_safe_new(G_corrigido, aabbs)


    print("🔹 plotando grafo...")
    PlotUtils.plot_subgraphs(G_corrigido, scale_x=15.0, scale_y= 10.0)


    print("🔹 SANITY CHECK... voltando para o GRAFO - Extraindo segmentos do grafo final...")
    new_segments = SegmentUtils.graph_to_segments(G_corrigido)

    print("🔹 SANITY CHECK... Plotando novos segmentos finais segmentos com AABBs preenchidas...")
    PlotUtils.plot_segments_aabbs_vertices(new_segments, aabbs, raio=0.5)


    PlotUtils.plot_aabbs_obstacles_points(obstacles,aabbs,observation_points)

    print(f"🔹 Salvando grafo em: {grafo_path} e {grafo_path_geo}")
    SegmentUtils.save_graph_json(G_corrigido, grafo_path, grafo_path_geo, file_path)

    print("✅ Fim do processo ---")

if __name__ == "__main__":
    main()
