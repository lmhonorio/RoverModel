# Config
from roverclass import ObstacleLoader
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from aabbutils import AABBUtils




file_path = "obstaculos_processado2.xlsx"
sheet_name = "Parnaiba3_Transformado"
padding = 15
margin = 2

# Carregar obstáculos
loader = ObstacleLoader(file_path, sheet_name)
obstacles = loader.get_obstacles()

# AABBs
aabbs = AABBUtils.get_aabbs(obstacles, margin)

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
final_segments = SegmentUtils.filter_similar_segments(filtered_segments, aabbs,
                                                      parallel_threshold=15.0)

# Adiciona subsegmentos do perímetro
SegmentUtils.add_perimeter_segments(aabbs, 4, final_segments)