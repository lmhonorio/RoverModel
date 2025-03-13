from roverclass import ObstacleLoader
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from aabbutils import AABBUtils
import math
import matplotlib.pyplot as plt
import pickle
import json


###############################################################################
# MAIN
###############################################################################
def main():
    # Config
    file_path = "obstaculos_processado2.xlsx"
    sheet_name = "Parnaiba3_Transformado"
    padding = 15
    margin = 2  #colocar esta coluna no xml para definir de forma personalizada a distancia do rover para cada objeto

    # Carregar obstáculos
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()

    # AABBs
    aabbs = AABBUtils.get_aabbs(obstacles, margin)

    PlotUtils.plot_obstacles_aabbs(obstacles,aabbs)

    # Determinar extents
    x_min = min(o["pos"][0] for o in obstacles) - padding
    x_max = max(o["pos"][0] for o in obstacles) + padding
    y_min = min(o["pos"][1] for o in obstacles) - padding
    y_max = max(o["pos"][1] for o in obstacles) + padding

    # Gera caminhos horizontais e verticais
    horizontal_paths, vertical_paths = SegmentUtils.get_paths(aabbs, x_min, x_max, y_min, y_max)

    # Subdivisão fora de AABB (índice par-ímpar)
    valid_segments = SegmentUtils.split_and_filter_paths(horizontal_paths, vertical_paths, aabbs)

    # Filtrar pelos endpoints e centro
    filtered_segments = SegmentUtils.filter_segments_by_distance(valid_segments, aabbs,
                                                                 endpoint_threshold=3.0,
                                                                 center_threshold=5.0)

    # Filtra paralelos duplicados
    prefinal_segments = SegmentUtils.filter_similar_segments(filtered_segments, aabbs,
                                                             parallel_threshold=15.0)

    #prefinal_segments = []

    # Adiciona subsegmentos do perímetro
    final_segments = SegmentUtils.add_perimeter_segments(aabbs, prefinal_segments, threshold_ponto_por_distancia= 4)


    # Plot final
    #PlotUtils.plot_segments_with_vertices(final_segments, raio=1.0)
    # Criar grafo
    G = SegmentUtils.create_graph(final_segments,obstacles)

    #new_indexed_labels = SegmentUtils.index_graph_labels(G)

    #print(new_indexed_labels)

    #PlotUtils.plot_graph_with_indexed_labels(G, new_indexed_labels)


    # Plot subgrafos conectados
    PlotUtils.plot_subgraphs(G)

    # PlotUtils.plot_grafo(G, "grafo.png", (8000, 8000), "grafo subestacao")

    new_segments = SegmentUtils.graph_to_segments(G)

    # Plot final
    PlotUtils.plot_segments_with_vertices(new_segments, raio=1.0)


    # Salvar grafo e segmentos
    SegmentUtils.save_graph(G, "graph3.pkl")
    SegmentUtils.save_segments(new_segments, "segments2.json")





if __name__ == "__main__":
    main()
