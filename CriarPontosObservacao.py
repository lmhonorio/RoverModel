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
    file_path = "./planilhas/obstaculos_processado6.xlsx"
    sheet_name = "Parnaiba3_Transformado"
    padding = 15
    margin = 2.5  #colocar esta coluna no xml para definir de forma personalizada a distancia do rover para cada objeto

    # Carregar obstáculos
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()

    # AABBs
    aabbs = AABBUtils.get_aabbs(obstacles, margin)

    #PlotUtils.plot_obstacles_aabbs(obstacles,aabbs)

    # Determinar extents
    x_min = min(o["pos"][0] for o in obstacles) - padding
    x_max = max(o["pos"][0] for o in obstacles) + padding
    y_min = min(o["pos"][1] for o in obstacles) - padding
    y_max = max(o["pos"][1] for o in obstacles) + padding


    horizontal_paths, vertical_paths = SegmentUtils.get_paths(aabbs, x_min, x_max, y_min, y_max)


    valid_segments = SegmentUtils.split_and_filter_paths(horizontal_paths, vertical_paths, aabbs)



    filtered_segments = SegmentUtils.filter_segments_by_distance(
        valid_segments, aabbs,
        endpoint_threshold=5.0,
        center_threshold=20.0
    )


    prefinal_segments = SegmentUtils.filter_similar_segments(filtered_segments, aabbs,
                                                             parallel_threshold=15.0)


    final_segments = SegmentUtils.add_perimeter_segments(aabbs, prefinal_segments,
                                                         threshold_ponto_por_distancia=4)
    print(f"🔹 Total de segmentos finais (com perímetro): {len(final_segments)}")




    print("🔹 Plotando segmentos e vértices...")
    PlotUtils.plot_segments_with_vertices(final_segments, raio=0.5)




    print("🔹 Criando grafo a partir dos segmentos...")
    G = SegmentUtils.create_graph(final_segments, obstacles)




    print("🔹 Corrigindo conexões faltantes com verificação contra AABBs...")
    G = SegmentUtils.fix_missing_connections_safe(G, aabbs)




    print("🔹 Extraindo segmentos do grafo final...")
    new_segments = SegmentUtils.graph_to_segments(G)



    print("🔹 Quebrando segmentos com interseccao...")
    broken_segments, added_points = SegmentUtils.resolve_segment_intersections(new_segments,2)


    print("🔹 Plotando segmentos com AABBs preenchidas...")
    PlotUtils.plot_segments_aabbs_vertices(broken_segments, aabbs, raio=0.5)

    print("🔹 recriando o grafo ...")
    G_corrigido = SegmentUtils.create_graph_with_passage_points(broken_segments, added_points, obstacles)


    print("🔹 Corrigindo conexões faltantes com verificação contra AABBs...")
    G_corrigido = SegmentUtils.fix_missing_connections_safe(G_corrigido, aabbs)


    print("🔹 Extraindo segmentos do grafo final...")
    new_segments = SegmentUtils.graph_to_segments(G_corrigido)

    print("🔹 Plotando novos segmentos finais segmentos com AABBs preenchidas...")
    PlotUtils.plot_segments_aabbs_vertices(new_segments, aabbs, raio=0.5)



    file_path = "./jsons/graph6.json"
    print(f"🔹 Salvando grafo em: {file_path}")
    SegmentUtils.save_graph_json(G_corrigido, file_path)

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
