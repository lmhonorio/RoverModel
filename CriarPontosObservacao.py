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


###############################################################################
# MAIN
###############################################################################
def main():
    # Config
    # file_path = "./planilhas/obstaculos_processado6.xlsx"
    file_path = "./planilhas/equipment_processado.xlsx"
    sheet_name = "Parnaiba3_Transformado"
    # grafo_path = "./jsons/graph9c_new.json"
    grafo_path = "./jsons/graph_equipment.json"
    grafo_path_geo = "./jsons/graph_equipment_geo.json" # grafo com coordenadas geograficas
    # observation_path ="./jsons/obpc_6.json"
    observation_path ="./jsons/obs_equipment.json"
    # observation_folter = "./pontos_observacao"
    observation_folter = "./pontos_observacao2"
    padding = 15
    margin = 1.5  #colocar esta coluna no xml para definir de forma personalizada a distancia do rover para cada objeto
    # margin = 2.5
    threshold = 30 # verifica largura e altura do aabb após o merge para permitir sobreposicao de aabbs

    # Carregar obstáculos
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()

    # AABBs

    aabbs = AABBUtils.get_aabbs(obstacles, margin, threshold)

    segments = SegmentUtils.generate_segments_between_aabbs(aabbs, 4.0)

    PlotUtils.plot_segments_aabbs_vertices(segments, aabbs, 0.5)

    print("🔹 criando o generate_perimeter_segments_and_labeled_points ...")
    segments, observation_points = SegmentUtils.generate_perimeter_segments_and_labeled_points(segments, aabbs, obstacles, threshold=3.0)


#salva os arquivos para visualizacao no excel, importacao no planner e visualizacao em gis - aqui encontra os pontos que olham para o objeto da melhor forma (melhor = perto)
  #  SegmentUtils.save_observation_points_to_excel(obstacles, observation_points, 6, file_path)
    SegmentUtils.save_observation_points_to_json(obstacles, observation_points, 6, observation_path, file_path)
    SegmentUtils.save_observation_points_to_kml(obstacles, observation_points, 6, file_path, observation_folter, offset_lat_meters=0.0, offset_lon_meters=0.0)

    print("🔹 criando o plot_aabbs_obstacles_points ...")
    PlotUtils.plot_aabbs_obstacles_points(obstacles,aabbs,observation_points)

    #PlotUtils.plot_segments_aabbs_vertices(segments,aabbs,0.5)

    print("🔹 Quebrando segmentos com interseccao...")
    broken_segments, passage_points = SegmentUtils.resolve_segment_intersections(segments, 2)


    PlotUtils.plot_segments_aabbs_vertices(broken_segments, aabbs, 0.5)


    print("🔹 criando o grafo ...")
    G_corrigido = SegmentUtils.create_graph_with_passage_points_new(broken_segments, passage_points, observation_points, obstacles)

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


    # Exemplo de uso:
    # g = SegmentUtils.load_graph_json(file_path)
    # grafo_mapa = AABBUtils.convert_graph_to_dict(g)


    # G1 = SegmentUtils.xml_to_graph(grafo_mapa)
    # agraph1 = to_agraph(G1)
    # agraph1.layout(prog='dot')
    # agraph1.draw('./figuras/graph_with_weights.png')  # Gerar o arquivo de imagem
    #
    # img = plt.imread('./figuras/graph_with_weights.png')
    # plt.imshow(img)
    # plt.axis('off')  # Remover eixos
    # plt.show()

if __name__ == "__main__":
    main()
