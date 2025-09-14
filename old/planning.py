# Import das classes e funções
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from planning_functions import *




def build_goal_predicates_func(mission_scenarios):
    """
    Gera predicados de objetivo a partir das missões.
    Cada missão concluída se torna `MissaoAtendida(local, True)`, e esse estado nunca muda.
    """
    goal_set = set()
    for local, _ in mission_scenarios:
        goal_set.add(Predicate("MissaoAtendida", local, True))  # O objetivo final é que todas as missões sejam atendidas

    def _goal_predicates_func():
        return goal_set

    return _goal_predicates_func


def is_mission_accomplished(state, goal_predicates):
    """
    Verifica se todas as missões foram atendidas **em algum momento**.
    Se um local foi marcado como `MissaoAtendida(local, True)`, então a missão daquele local foi concluída permanentemente.
    """
    mission_status = {p.args[0]: p.args[1] for p in state.predicates if p.name == "MissaoAtendida"}

    for gp in goal_predicates:
        if gp.name == "MissaoAtendida":
            local = gp.args[0]
            if mission_status.get(local, False) is not True:
                return False  # A missão ainda não foi atendida

    return True


# ==================== PLANO PRINCIPAL ==================== #

if __name__ == "__main__":
    # Carregar o grafo
    file_path = "../jsons/graph8.json"
    g = SegmentUtils.load_graph_json(file_path)
    grafo_mapa = AABBUtils.convert_graph_to_dict(g)

    # Definir robôs e posições iniciais
    robots = [
        Instance("Robot", "R1"),
        Instance("Robot", "R2"),
    ]
    pred_iniciais = [
        Predicate("Em", robots[0], "PR11_2"),
        Predicate("Em", robots[1], "PR12_7"),
    ]

    # Missões principais
    mission_scenarios = [
        ("TPC3_5", "ANY"),
        ("TPC2_6", "R2")
    ]

    # Construir predicados de objetivo
    goal_predicates_func = build_goal_predicates_func(mission_scenarios)

    # Adicionar `MissaoAtendida` inicial como False
    for local, _ in mission_scenarios:
        pred_iniciais.append(Predicate("MissaoAtendida", local, False))

    # Estado inicial
    initial_state = State(pred_iniciais, 0)

    # Definir operadores SEM bateria
    operators = [
        Operator(
            "Mover(robo, posicao)",
            preconditions_func=move_preconditions,
            add_effects_func=move_add_effects,
            del_effects_func=move_del_effects,
            cost=1
        ),
    ]

    # Criar um novo planejador que usa os predicados de missão
    class MissionPlanner(AStarPlanner):
        def is_goal(self, state):
            goal_preds = self.goal_predicates_func()
            return is_mission_accomplished(state, goal_preds)

    planner = MissionPlanner(
        initial_state,
        goal_predicates_func,
        operators,
        robots,
        grafo_mapa,
        get_goal_state_predicates=None
    )

    # Executar o plano
    path = planner.plan_graph_based()
    if path:
        print("✅ Caminho final:", path)
    else:
        print("❌ Nenhum caminho encontrado!")
