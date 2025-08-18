import json
from old.multigraphplanner import MultiGraphPlanner
from persistent_astar_cache import PersistentAStarCache
from segmentutils import SegmentUtils
import networkx as nx
from aabbutils import AABBUtils
from plotutils import PlotUtils
from collections import defaultdict
from movns_ains_argo import task_priority_argo
from movns_ains_argo.robot import Robot
from movns_ains_argo import solution_priority_argo
from movns_ains_argo import movns_ains_argo
from tspOptimization import FixedTaskPlanner




####################################### INSTANCIAR ROBÔS DA MISSÃO ########################################

def get_robots(G_mapa, selected_ids=None):
    """
    Retorna robôs instanciados, suas posições e tempos de bateria.

    Parâmetros:
        - G_mapa: grafo com os nós do mapa
        - selected_ids: lista com IDs dos robôs desejados (ex: ["R1", "R3"])
                        se None, retorna todos os robôs disponíveis

    Retorna:
        - robots: lista de objetos Robot
        - robots_positions: dict {robot_id: start_node_label}
        - battery_times: lista de tempos de bateria
    """

    all_robot_coords = [
        (-165.9766, -77.6645),  # R1
        (87.9766, 30.6645),     # R2
        (87.9766, 30.6645),     # R3
        (-165.9766, -77.6645),  # R4
        (-165.9766, -77.6645)   # R5
    ]

    all_robots_data = {}

    for i, (tx, ty) in enumerate(all_robot_coords, 1):
        robot_id = f"R{i}"
        label_pos, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
        all_robots_data[robot_id] = {
            "position": label_pos,
            "coords": (tx, ty),
            "battery_time": 1000000  # padrão, pode customizar
        }

    if selected_ids is None:
        selected_ids = list(all_robots_data.keys())

    robots = []
    robots_positions = {}
    battery_times = []

    for robot_id in selected_ids:
        data = all_robots_data[robot_id]
        robots_positions[robot_id] = data["position"]
        battery_times.append(data["battery_time"])

        robot = Robot(
            id=robot_id,
            battery_time=data["battery_time"],
            start_node_label=data["position"],
            graph=G_mapa
        )
        robots.append(robot)

    return robots

##############################################################################################################################################


# G_p = nx.Graph()
G_m = nx.DiGraph()
G_r = {"R1": nx.Graph(), "R2": nx.Graph()}

#leitura do grafo
file_path = "./jsons/graph8_new_funcionando.json"
# file_path = "./jsons/graph9_new.json"

#pontos de observacao em relacao a distancia dos objetos
observation_points_json_path = "./jsons/obp_6_funcionando.json"
# observation_points_json_path = "./jsons/obp_6.json"

json_mission = "./jsons/missao_6.json"


observacao_por_obstaculo  = SegmentUtils.load_observation_points_from_json(observation_points_json_path)

G_mapa = SegmentUtils.load_graph_json(file_path)

G_p = AABBUtils.convert_graph_to_dict(G_mapa)

planner_grafo = MultiGraphPlanner(G_p, mission_graph=None,
                            robots_graphs=None, mission_positions=None,
                            mission_times=None, mission_execution=None)

cache_astar = PersistentAStarCache(planner_grafo, "./cache_astar.json")

#Equipamentos por AABB
with open("./jsons/aabb_por_equipamento.json", "r") as f:
    equipamento_aabb = json.load(f)

#Informação dos Equipamentos
with open("./jsons/equipamentos_filtrados.json", "r") as f:
    equipamento_data = json.load(f)

with open("./jsons/aabb_info.json", "r") as f:
    aabb_info_data = json.load(f)


################ 22 TASKS ###############################
""" missions = [
    'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
    'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
    'ls_tpc1', 'ls_tpc2', 'ls_tpc3', 'ls_tpc4', 'ls_tpc5', 'ls_tpc6',
    'r_pr1', 'r_reator1', 'r_reator2', 'r_pr2'] """


################ 52 TASKS ###############################
""" missions = [
    'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
    'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
    
    'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',

    'ef_buscsb21', 'ef_buscsb22',
    'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip6', 'ef_busip7', 'ef_busip8', 'ef_busip9', 'ef_busip10',

    'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
    'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb6', 'cd_buscsb5', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
  
    'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
    'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4'
] """

#################################### ALL TASKS ###################################
""" missions = [
    'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
    'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
    'ef_tpc1', 'ef_tpc2', 'ef_tpc3',
    'ef_ip1', 'ef_ip2', 'ef_ip3', 'ef_ip4', 'ef_ip5', 'ef_ip6', 'ef_ip7', 'ef_ip8', 'ef_ip9', 'ef_ip10', 'ef_ip11', 'ef_ip12', 'ef_ip13', 'ef_ip14', 'ef_ip15', 'ef_ip16', 'ef_ip17', 'ef_ip18',
    'ef_sech1', 'ef_sech2', 'ef_sech3', 'ef_sech4', 'ef_sech5', 'ef_sech6', 'ef_sech7', 'ef_sech8', 'ef_sech9', 'ef_sech10',
    'ef_sech11', 'ef_sech12', 'ef_sech13', 'ef_sech14', 'ef_sech15', 'ef_sech16', 'ef_sech17', 'ef_sech18', 'ef_sech19', 'ef_sech20', 'ef_sech21', 'ef_sech22', 'ef_sech23', 'ef_sech24',
    'ef_tc1', 'ef_tc2', 'ef_tc3', 'ef_tc4', 'ef_tc5', 'ef_tc6', 'ef_tc7', 'ef_tc8', 'ef_tc9', 'ef_tc10', 'ef_tc11', 'ef_tc12', 'ef_tc13', 'ef_tc14', 'ef_tc15',
    'ef_secv1', 'ef_secv2', 'ef_secv3', 'ef_secv4', 'ef_secv5', 'ef_secv6', 'ef_secv7', 'ef_secv8', 'ef_secv9', 'ef_secv10', 'ef_secv11', 'ef_secv12', 'ef_secv13', 'ef_secv14', 'ef_secv15',
    'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',
    'ef_buscsb1', 'ef_buscsb2', 'ef_buscsb3', 'ef_buscsb4', 'ef_buscsb5', 'ef_buscsb6', 'ef_buscsb7', 'ef_buscsb8', 'ef_buscsb9', 'ef_buscsb10',
    'ef_buscsb11', 'ef_buscsb12', 'ef_buscsb13', 'ef_buscsb14', 'ef_buscsb15', 'ef_buscsb16', 'ef_buscsb17', 'ef_buscsb18', 'ef_buscsb19', 'ef_buscsb20',
    'ef_buscsb21', 'ef_buscsb22',
    'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip6', 'ef_busip7', 'ef_busip8', 'ef_busip9', 'ef_busip10',
    'ef_busip11', 'ef_busip12', 'ef_busip13', 'ef_busip14', 'ef_busip15', 'ef_busip16', 'ef_busip17', 'ef_busip18', 'ef_busip19', 'ef_busip20',
    'ef_busip21', 'ef_busip22', 'ef_busip23', 'ef_busip24', 'ef_busip25', 'ef_busip26', 'ef_busip27', 'ef_busip28', 'ef_busip29', 'ef_busip30',
    'ef_busip31', 'ef_busip32', 'ef_busip33', 'ef_busip34', 'ef_busip35', 'ef_busip36', 'ef_busip37', 'ef_busip38', 'ef_busip39', 'ef_busip40',
    'ef_busip41', 'ef_busip42', 'ef_busip43', 'ef_busip44', 'ef_busip45', 'ef_busip46', 'ef_busip47',
    'ef_pr1', 'cd_tc9', 'cd_tc8', 'cd_tc7', 'cd_tc6', 'cd_tc5', 'cd_tc4', 'cd_tc3', 'cd_tc2', 'cd_tc12', 'cd_tc11', 'cd_tc10', 'cd_tc1',
    'cd_secv9', 'cd_secv8', 'cd_secv7', 'cd_secv6', 'cd_secv5', 'cd_secv4', 'cd_secv3', 'cd_secv2', 'cd_secv12', 'cd_secv11', 'cd_secv10', 'cd_secv1',
    'cd_sech3', 'cd_sech2', 'cd_sech1', 'cd_sech12', 'cd_sech11', 'cd_sech10', 'cd_sech9', 'cd_sech8', 'cd_sech7', 'cd_sech6', 'cd_sech5', 'cd_sech4',
    'cd_ip9', 'cd_ip8', 'cd_ip7', 'cd_ip12', 'cd_ip11', 'cd_ip10',
    'cd_disjuntor9', 'cd_disjuntor8', 'cd_disjuntor7', 'cd_disjuntor6', 'cd_disjuntor5', 'cd_disjuntor4', 'cd_disjuntor3', 'cd_disjuntor2', 'cd_disjuntor12', 'cd_disjuntor11', 'cd_disjuntor10', 'cd_disjuntor1',
    'cd_busip12', 'cd_busip11', 'cd_busip10', 'cd_busip9', 'cd_busip8', 'cd_busip59', 'cd_busip58', 'cd_busip57', 'cd_busip56', 'cd_busip55',
    'cd_busip54', 'cd_busip53', 'cd_busip52', 'cd_busip7', 'cd_busip51', 'cd_busip50', 'cd_busip49', 'cd_busip46', 'cd_busip45', 'cd_busip44',
    'cd_busip43', 'cd_busip42', 'cd_busip41', 'cd_busip6', 'cd_busip40', 'cd_busip39', 'cd_busip38', 'cd_busip37', 'cd_busip36', 'cd_busip35',
    'cd_busip30', 'cd_busip29', 'cd_busip28', 'cd_busip27', 'cd_busip5', 'cd_busip26', 'cd_busip25', 'cd_busip24', 'cd_busip23', 'cd_busip22',
    'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
    'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb6', 'cd_buscsb5', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
    'cd_buscsb3', 'cd_buscsb22', 'cd_buscsb21', 'cd_buscsb20', 'cd_buscsb19', 'cd_buscsb18', 'cd_buscsb17', 'cd_buscsb16', 'cd_buscsb15', 'cd_buscsb14',
    'cd_buscsb11', 'cd_buscsb2',
    'cd_reator1', 'cd_reator2', 'cd_reator3', 'cd_reator4', 'cd_reator5', 'cd_reator6',
    'cd_busip3', 'cd_busip1', 'cd_busip2', 'cd_busip16', 'cd_busip17', 'cd_busip19', 'cd_busip31', 'cd_busip15', 'cd_buscsb12', 'cd_busip48', 'cd_busip34', 'cd_busip33', 'cd_busip32', 'cd_busip47', 'cd_buscsb13', 'cd_buscsb1',
    'cd_pr1', 'cd_pr2', 'cd_pr3', 'cd_pr4', 'cd_pr5', 'cd_pr6',
    'cd_ip1', 'cd_ip2', 'cd_ip3', 'cd_ip4', 'cd_ip5', 'cd_ip6',
    'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
    'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4',
    'b_sech6', 'b_sech5', 'b_sech4', 'b_sech3', 'b_sech2', 'b_sech1',
    'b_ip3', 'b_ip2', 'b_ip1',
    'b_disjuntor3', 'b_disjuntor2', 'b_disjuntor1', 'b_disjuntor6', 'b_disjuntor5', 'b_disjuntor4',
    'b_busip23', 'b_busip22', 'b_busip21', 'b_busip20', 'b_busip19', 'b_busip31', 'b_busip30', 'b_busip29', 'b_busip28', 'b_busip27',
    'b_busip25', 'b_busip24',
    'b_buscsb5', 'b_buscsb4', 'b_buscsb3', 'b_buscsb2', 'b_buscsb1',
    'b_busip36', 'b_busip32', 'b_buscsb6', 'b_busip37', 'b_busip33', 'b_buscsb7', 'b_busip38', 'b_busip34', 'b_buscsb8', 'b_busip39',
    'b_busip35', 'b_busip40',
    'b_busip16', 'b_busip17', 'b_busip18', 'b_busip15', 'b_busip14', 'b_busip13', 'b_busip9', 'b_busip5', 'b_busip1', 'b_busip2',
    'b_busip3', 'b_busip4', 'b_busip8', 'b_busip7', 'b_busip6', 'b_busip10', 'b_busip11', 'b_busip12', 'b_busip26',
    'ls_pr1', 'ls_pr2', 'ls_pr3', 'ls_pr4', 'ls_pr5', 'ls_pr6',
    'ls_tpc1', 'ls_tpc2', 'ls_tpc3', 'ls_tpc4', 'ls_tpc5', 'ls_tpc6',
    'r_pr1', 'r_reator1', 'r_reator2', 'r_pr2'
] """


################ 45 TASKS ###############################
""" missions = [
    'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
    'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7','ef_pr11', 'ef_pr12', 'ef_pr13',
    
    'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',

    'ef_buscsb21', 'ef_buscsb22',
    'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip10',

    'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
    'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
  
    'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
    'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4',
   
] """

################ 42 TASKS ###############################
""" missions = [
    'ef_reator1','ef_pr2','ef_tpc1','ef_ip1','ef_sech1','ef_sech11','ef_tc1','ef_secv1','ef_disjuntor1','ef_buscsb1','ef_buscsb11','ef_buscsb21','ef_busip1','ef_busip11',
    'ef_busip21','ef_busip31','ef_busip41','ef_pr1','cd_secv9','cd_sech3','cd_ip9','cd_disjuntor9','cd_busip12','cd_busip54','cd_busip43','cd_busip30','cd_busip21','cd_buscsb10',
    'cd_buscsb3','cd_buscsb11','cd_reator1','cd_busip3','cd_pr1','cd_ip1','b_tc3','b_secv3','b_sech6','b_ip3','b_disjuntor3','b_busip23','b_busip25','b_buscsb5',
    'b_busip36','b_busip35','b_busip16','b_busip3','ls_pr1','ls_tpc1','r_pr1',
] """



"""missions = [
    'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
    'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
    'ef_tpc1', 'ef_tpc2', 'ef_tpc3',
    'ef_ip1', 'ef_ip2', 'ef_ip3', 'ef_ip4', 'ef_ip5', 'ef_ip6', 'ef_ip7', 'ef_ip8', 'ef_ip9', 'ef_ip10', 'ef_ip11', 'ef_ip12', 'ef_ip13', 'ef_ip14', 'ef_ip15', 'ef_ip16', 'ef_ip17', 'ef_ip18',
    'ef_sech1', 'ef_sech2', 'ef_sech3', 'ef_sech4', 'ef_sech5', 'ef_sech6', 'ef_sech7', 'ef_sech8', 'ef_sech9', 'ef_sech10',
    'ef_sech11', 'ef_sech12', 'ef_sech13', 'ef_sech14', 'ef_sech15', 'ef_sech16', 'ef_sech17', 'ef_sech18', 'ef_sech19', 'ef_sech20', 'ef_sech21', 'ef_sech22', 'ef_sech23', 'ef_sech24',
    'ef_tc1', 'ef_tc2', 'ef_tc3', 'ef_tc4', 'ef_tc5', 'ef_tc6', 'ef_tc7', 'ef_tc8', 'ef_tc9', 'ef_tc10', 'ef_tc11', 'ef_tc12', 'ef_tc13', 'ef_tc14', 'ef_tc15',
    'ef_secv1', 'ef_secv2', 'ef_secv3', 'ef_secv4', 'ef_secv5', 'ef_secv6', 'ef_secv7', 'ef_secv8', 'ef_secv9', 'ef_secv10', 'ef_secv11', 'ef_secv12', 'ef_secv13', 'ef_secv14', 'ef_secv15',
    'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10']"""


################ 6 TASKS ###############################
missions = ['b_busip4','ef_reator1','ls_pr4', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_disjuntor1', 'ef_sech1', 'b_buscsb5']



tasks = task_priority_argo.gerar_tasks(missions, observacao_por_obstaculo, equipamento_aabb, equipamento_data, aabb_info_data)

robots = get_robots(G_mapa, selected_ids=["R1", "R2", "R3"])


################## SOLUÇÃO GULOSA ############################################

""" solution = solution_priority_argo.Solution(G_p, robots, tasks, cache_astar=cache_astar)
allocations = solution.greedy_allocate_tasks_graph_based(robots, tasks)

solution.calculate_metrics()
solution.print_solution_metrics() """

# PlotUtils.plotar_rotas_de_tasks(tasks)

# PlotUtils.plot_robot_routes_from_solution(solution, G_mapa)

# planner_grafo.execute_plan_from_solution(solution)


############################### SOLUÇÃO COM MOVNS ###################################

population = movns_ains_argo.run_movns(robots, tasks, G_p, cache_astar, time_limit=10)

# UTILIZANDO A SOLUÇÃO DE MENOR TEMPO SOMENTE PARA TESTE !!!
best_by_time = min(population, key=lambda s: s.time)
best_by_time.print_solution_metrics()

rotas_por_robo_movns = solution_priority_argo.calcula_metricas(best_by_time)

planner_grafo.execute_plan_from_solution(best_by_time)

# PlotUtils.plot_robot_routes_from_solution(best_by_time, G_mapa)

############################ ABORDAGEM DINAMICA A SER IMPLEMENTADA ##################################################

""" greedy_solution_failure, all_remaining_tasks = movns_ains_argo.greedy_reallocate_failed_robot_with_graph(robots, best_by_time, 'R1', cache_astar)

final_pop = [greedy_solution_failure]

greedy_solution_failure.print_solution_metrics()

PlotUtils.plot_robot_routes_from_solution(greedy_solution_failure, G_mapa)

final_pop = [greedy_solution_failure]
movns_ains_argo.dynamic_movns(final_pop, greedy_solution_failure.robots, greedy_solution_failure.tasks) """

##################################### PRIMEIRA ABORDAGEM PROFESSOR LEONARDO ##################################################

""" mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)

# PlotUtils.plotar_rotas_dos_robos(G_mapa, robots)


mission_execution = {
    'ef_reator1': ['R1', 'R2'],
    'ef_reator2': ['R1', 'R2'],
    'ef_reator3': ['R1', 'R2'],
    'ef_reator4': ['R1', 'R2'],
    
    'ls_pr4': ['R1', 'R2'],

    'b_busip4': ['R1', 'R2'],
    'ef_disjuntor1': ['R1', 'R2'],
    'ef_sech1': ['R1', 'R2'],
    'b_buscsb5': ['R1', 'R2']
}
mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)

#transforma as missoes em pontos de observacao individuais

# Use:
point_mission_positions = {}
for mission, points in mission_positions.items():
    for point in points:
        point_mission_positions[point] = [point]  # Cada ponto é sua própria posição alvo

G_m.add_nodes_from([ponto for pontos in mission_positions.values() for ponto in pontos])

point_mission_times = {ponto: 15 for pontos in mission_positions.values() for ponto in pontos}

possible_robots_per_execution_point = {}
for mission, robots in mission_execution.items():
    for point in mission_positions[mission]:
        possible_robots_per_execution_point[point] = robots """

tx, ty = -165.9766, -77.6645
tx2, ty2 = 87.9766, 30.6645
tx3, ty3 = 87.9766, 30.6645
tx4, ty4 = -165.9766, -77.6645
tx5, ty5 = -165.9766, -77.6645

label_posr1, nearest, dist = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
label_posr2, nearest2, dist2 = MultiGraphPlanner.find_nearest_node(G_mapa, tx2, ty2)
label_posr3, nearest3, dist3 = MultiGraphPlanner.find_nearest_node(G_mapa, tx3, ty3)
label_posr4, nearest4, dist4 = MultiGraphPlanner.find_nearest_node(G_mapa, tx4, ty4)
label_posr5, nearest5, dist5 = MultiGraphPlanner.find_nearest_node(G_mapa, tx5, ty5)


robots_positions = {"R1": label_posr1, "R2": label_posr2, "R3": label_posr3}


""" start = time.time()
planner = MultiGraphPlanner(G_p, G_m, G_r, point_mission_positions, point_mission_times, possible_robots_per_execution_point)
optimal_plan, min_time, schedule = planner.find_minimum_mission_time_plan(robots_positions, cache_astar)
end = time.time()
print(f"⏱ Tempo de execução: {end - start:.4f} segundos") """



# planner.save_optimal_plan_to_json(optimal_plan,json_mission)

# planner.execute_plan(optimal_plan, min_time, schedule)


########################################### NOVA ABORDAGEM COM CLUSTER ###########################

def gerar_mission_execution(missions, num_robots):
    """
    Gera um dicionário onde cada missão é atribuída a todos os robôs disponíveis.

    Args:
        missions (list): Lista de nomes de missões.
        num_robots (int): Número de robôs disponíveis (ex: 2, 3, 5, etc).

    Returns:
        dict: Dicionário no formato {'missao1': ['R1', 'R2', ...], ...}
    """
    mission_execution = {}
    robot_ids = [f"R{i+1}" for i in range(num_robots)]

    for mission in missions:
        mission_execution[mission] = robot_ids.copy()  # Copiar para evitar efeitos colaterais

    return mission_execution

mission_execution = gerar_mission_execution(missions, num_robots=3)

mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)

#transforma as missoes em pontos de observacao individuais

# Use:
point_mission_positions = {}
for mission, points in mission_positions.items():
    for point in points:
        point_mission_positions[point] = [point]  # Cada ponto é sua própria posição alvo

print("\n🔍 Missões individuais:")
for k, v in point_mission_positions.items():
    print(f"{k} -> {v}")

G_m.add_nodes_from([ponto for pontos in mission_positions.values() for ponto in pontos])

point_mission_times = {ponto: 15 for pontos in mission_positions.values() for ponto in pontos}

possible_robots_per_execution_point = {}
for mission, robots in mission_execution.items():
    for point in mission_positions[mission]:
        possible_robots_per_execution_point[point] = robots

print("Gerando Greduced_map")
Greduced_map = MultiGraphPlanner.build_inspection_graph([label_posr1,label_posr2,label_posr3], point_mission_positions, G_mapa)


# PlotUtils.plot_subgraphs(Greduced_map, scale_x=15.0, scale_y= 10.0)
# PlotUtils.plot_grafo_distance(Greduced_map)


# 🔄 Definição de execução permitida por missão principal
mission_execution_config = mission_execution

# Inicializar
mission_execution = {}
fixed_tasks_per_robot = defaultdict(list)
distributed_tasks = []

# Processar missões e pontos
for mission, points in mission_positions.items():
    robots = mission_execution_config.get(mission, [])
    for point in points:
        mission_execution[point] = robots
        if len(robots) == 1:
            # Tarefa fixa
            fixed_tasks_per_robot[robots[0]].append(point)
        else:
            # Tarefa distribuível
            distributed_tasks.append(point)

# Tempo de execução fixo por ponto
mission_times = {point: 15 for point in point_mission_positions}

grafo_mapa_dict = AABBUtils.convert_graph_to_dict(Greduced_map)

# Executar clusterização
pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(Greduced_map, point_mission_positions, robots_positions, mission_execution)

""" print("\n📌 Clusterização balanceada dos pontos por robô:")
for robo, pontos in pontos_por_robo.items():
    print(f"  {robo}: {pontos}") """


rotas_otimas_por_robo = {}

for robo, pontos in pontos_por_robo.items():
    ponto_inicial = robots_positions[robo]
    rota_otima = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, ponto_inicial, pontos)
    rotas_otimas_por_robo[robo] = rota_otima


PlotUtils.plot_rotas_grafo(Greduced_map,rotas_otimas_por_robo)
PlotUtils.plot_rotas_reais(G_mapa,rotas_otimas_por_robo,point_mission_positions, "Abordagem Professor Leonardo")

PlotUtils.plot_rotas_reais(G_mapa,rotas_por_robo_movns,point_mission_positions, "Abordagem MOVNS")


# PlotUtils.plot_comparacao_robo_a_robo(G_mapa, rotas_otimas_por_robo, rotas_por_robo_movns, point_mission_positions)

# Exibir resultados
""" for robo, rota in rotas_otimas_por_robo.items():
    print(f"\n🚗 Rota ótima para {robo}:")
    print(" -> ".join(rota)) """


tempo, distancia, balances, qtde_pontos = solution_priority_argo.calcular_custos_totais_solucao(rotas_otimas_por_robo, cache_astar)

















    # # Inicializa o planejador
    # method = "nearest" # "permutation"   #  "branch_bound" #  ou  or
    # planner = FixedTaskPlanner(grafo_mapa_dict, point_mission_positions, mission_times, mission_execution)
    #
    #
    # # Executa o planejamento de tarefas
    # plan, total_time, schedule = planner.distribute_and_schedule_tasks(
    #     robots_positions,
    #     fixed_tasks_per_robot,
    #     distributed_tasks,
    #     method="nearest" #"branch_bound"
    # )
    #
    # # Mostra o plano resultante
    # print(f"\n🔧 Tempo total de execução (makespan): {total_time:.2f} s")
    # for robot, tasks in plan.items():
    #     print(f"\n📦 Plano para {robot}:")
    #     for task in tasks:
    #         print(f"  ▶ Missão: {task['mission']}, Caminho: {task['path']}, "
    #               f"Deslocamento: {task['travel_time']}s, Execução: {task['execution_time']}s, "
    #               f"Início: {task['start_time']:.2f}, Fim: {task['end_time']:.2f}")
    #
    # # Geração de dicionário com os pontos atribuídos por robô
    # execution_points_per_robot = defaultdict(list)
    # for robot, tasks in plan.items():
    #     for t in tasks:
    #         execution_points_per_robot[robot].append(t["mission"])
    #
    # print("\n🗺️ Pontos de execução por robô:")
    # for robot, points in execution_points_per_robot.items():
    #     print(f"  {robot}: {points}")
