
import networkx as nx
from segmentutils import SegmentUtils
from aabbutils import AABBUtils

# Caminhos de entrada
file_path = "../jsons/graph9_new.json"
observation_points_json_path = "../jsons/obp_6.json"

# Carrega grafo e pontos
G_nx = SegmentUtils.load_graph_json(file_path)
G_dict = AABBUtils.convert_graph_to_dict(G_nx)
observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(observation_points_json_path)

# Seleciona alguns pontos de missão para testar
test_points = [
    'b_busip4.2493', 'b_busip4.2504', 'b_busip4.2510',
    'ef_reator1.1', 'ef_reator1.23', 'ef_reator1.30',
    'ls_pr4.2569'
]

# Define posição inicial de um robô
# Você pode substituir por qualquer ponto conhecido válido
start = 'b_busip4.2504'

print("🔍 Testando conectividade no grafo NetworkX...")
for target in test_points:
    if target not in G_nx.nodes or start not in G_nx.nodes:
        print(f"❌ Nó ausente no grafo NetworkX: {target}")
        continue
    try:
        path = nx.shortest_path(G_nx, source=start, target=target, weight='weight')
        print(f"✅ Caminho de {start} até {target}: {path}")
    except nx.NetworkXNoPath:
        print(f"❌ Sem caminho no grafo NetworkX de {start} → {target}")

print("\n🔍 Verificando presença e transições no grafo convertido (G_dict)...")
for target in test_points:
    if target not in G_dict['states']:
        print(f"❌ {target} ausente em G_dict['states']")
    else:
        print(f"✅ {target} presente em G_dict['states']")

print("\n🔁 Transições envolvendo start ou target:")
for k, v in G_dict['transitions'].items():
    if start in k or any(tp in k for tp in test_points):
        print(f"  {k} → {v}")
