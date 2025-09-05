from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from collections import defaultdict
from tspOptimization import FixedTaskPlanner
from aabbutils import AABBUtils
from snakes.nets import PetriNet, Place, Transition, Value



def graph_to_petri_net(G, net_name="GrafoPetri"):
    petri_net = PetriNet(net_name)

    # Cria os lugares para cada nó do grafo
    for node in G.nodes():
        petri_net.add_place(Place(str(node)))  # lugares são identificados por string

    # Cria transições para cada aresta
    for i, (src, dst) in enumerate(G.edges()):
        trans_name = f"t_{src}_{dst}"  # ou usar apenas um índice
        petri_net.add_transition(Transition(trans_name))

        # Arcos de entrada e saída
        petri_net.add_input(str(src), trans_name, Value(1))
        petri_net.add_output(str(dst), trans_name, Value(1))

    return petri_net


if __name__ == "__main__":
    # Leitura do grafo do ambiente
    file_path = "./jsons/graph9_new.json"
    observation_points_json_path = "./jsons/obp_6.json"


    # Carregar pontos de observação por obstáculo
    observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(observation_points_json_path)

    # Missões por obstáculo
    missions = ['b_busip4', 'ef_reator1', 'ls_pr4']
    mission_positions = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo, missions)

    # Transformar pontos de observação em missões individuais
    point_mission_positions = {}
    for mission, points in mission_positions.items():
        for point in points:
            point_mission_positions[point] = point  # Cada ponto é sua própria posição alvo

    print("\n🔍 Missões individuais:")
    for k, v in point_mission_positions.items():
        print(f"{k} -> {v}")

    # Carrega o grafo do ambiente e reduzido para inspeção
    G_mapa = SegmentUtils.load_graph_json(file_path)



    # Define posições iniciais reais dos robôs com base em coordenadas (x, y)
    tx, ty = -165.9766, -77.6645
    tx2, ty2 = 87.9766, 30.6645

    label_posr1, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
    label_posr2, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx2, ty2)

    robots_positions = {"R1": label_posr1, "R2": label_posr2}

    Greduced_map = MultiGraphPlanner.build_inspection_graph([label_posr1,label_posr2], point_mission_positions, G_mapa)
    PlotUtils.plot_grafo_distance(Greduced_map)

    net = graph_to_petri_net(Greduced_map)
    print(net)

    # 🔄 Definição de execução permitida por missão principal
    mission_execution_config = {
        "b_busip4": ["R1", "R2"],
        "ef_reator1": ["R1", "R2"],
        "ls_pr4": ["R1","R2"]
    }

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

    print("\n📌 Clusterização balanceada dos pontos por robô:")
    for robo, pontos in pontos_por_robo.items():
        print(f"  {robo}: {pontos}")


    rotas_otimas_por_robo = {}

    for robo, pontos in pontos_por_robo.items():
        ponto_inicial = robots_positions[robo]
        rota_otima = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, ponto_inicial, pontos)
        rotas_otimas_por_robo[robo] = rota_otima


    PlotUtils.plot_rotas_grafo(Greduced_map,rotas_otimas_por_robo)
    PlotUtils.plot_rotas_reais(G_mapa,rotas_otimas_por_robo,point_mission_positions)

    # Exibir resultados
    for robo, rota in rotas_otimas_por_robo.items():
        print(f"\n🚗 Rota ótima para {robo}:")
        print(" -> ".join(rota))







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
