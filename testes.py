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


file_path = "./jsons/graph6.json"
grafo_mapa = SegmentUtils.load_graph_json(file_path)
agraph1 = to_agraph(grafo_mapa)


PlotUtils.plot_subgraphs(agraph1)
