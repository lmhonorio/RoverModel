from queue import PriorityQueue
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation as nx_app
from networkx.drawing.nx_agraph import to_agraph
from baseclasses import *
from aplanner import *
from roverclass import ObstacleLoader
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from aabbutils import AABBUtils
from planning_functions import *

# Example usage:
if __name__ == "__main__":
    # Define instances for robots and locations


    # Exemplo de uso:
    file_path = "graph8.json"  # Altere para o caminho correto do arquivo
    g = SegmentUtils.load_graph_json(file_path)
    grafo_mapa = AABBUtils.convert_graph_to_dict(g)

    # g = Grafo()
    # G1 = g.xml_to_graph(grafo_mapa)
    # agraph1 = to_agraph(G1)
    # agraph1.layout(prog='dot')
    # agraph1.draw('graph_with_weights.png')  # Gerar o arquivo de imagem
    #
    # img = plt.imread('graph_with_weights.png')
    # plt.imshow(img)
    # plt.axis('off')  # Remover eixos
    # plt.show()

    robots = []
    posicoes =[]
    baterias = []
    chargePositions = []

    robots.append(Instance("Robot", "R1"))
    robots.append(Instance("Robot", "R2"))
    posicoes.append(Predicate("Em", robots[0], "PR11_2"))
    posicoes.append(Predicate("Em", robots[1], "PR12_7"))
    baterias.append(Predicate("Btry", 'R1', 20000))
    baterias.append(Predicate("Btry", 'R2', 20000))
    chargePositions.append(Predicate("ChrPos", 'PR11_6'))
    # Instance("Robot", f"R2")


    locations = [Instance("Place", i) for i in sorted(grafo_mapa['states'], key=str)]



    # Combine all predicates into the initial state
    initial_state = State(posicoes + baterias + chargePositions, 0)

   # print("Estados do grafo:", grafo_mapa['states'])


    # Define the list of operators
    operators = [
        Operator(
            "Mover(robo, posicao)",
            preconditions_func=move_preconditions,
            add_effects_func=move_add_effects,
            del_effects_func=move_del_effects,
            cost=1  # Assume all movements have equal cost
        ),
        Operator(
            "Carregar(robo)",
            preconditions_func=charge_preconditions,
            add_effects_func=charge_add_effects,
            del_effects_func=charge_del_effects,
            cost=1  # Assume all movements have equal cost
        )
    ]


    # def goal_predicates_func():
    #     # We want each position to have exactly one robot.
    #     goal_predicates = set()
    #     goal_predicates.add(Predicate("Full", "PR11_0"))
    #     #goal_predicates.add(Predicate("Full", "P5"))
    #     #goal_predicates.add(Predicate("Carga", Instance("Robot", "robo1"), 100))
    #     #goal_predicates.add(Predicate("Carga", Instance("Robot", "robo2"), 100))
    #     return goal_predicates

    goal_predicates_func = lambda: {Predicate("Full", "TPC1_0"), Predicate("Full", "TPC3_6") }



    # Create the planner
    planner = AStarPlanner(initial_state, goal_predicates_func, operators, robots, grafo_mapa, get_goal_state_predicates)

    # Find the optimal plan
    plan, final_state = planner.plan()
    if plan:
        print("Optimal Plan:")
        for action, robot, location in plan:
            print(f"{robot.name} executes {action.name} to {location.name}")
    else:
        print("No plan found")

    print("\nEstado Final:")

    for predicate in sorted(final_state.predicates, key=str):
        print(predicate)
