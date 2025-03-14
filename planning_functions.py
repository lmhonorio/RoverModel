from aplanner import *



def get_goal_state_predicates(current_state: State):
    goalpredicates = set()
    for p in current_state.predicates:
        if p.name == "Em":
            goalpredicates.add(Predicate("Full", p.args[1]))

    return goalpredicates


# def goal_predicates_func():
#     # We want each position to have exactly one robot.
#     goal_predicates = set()
#     goal_predicates.add(Predicate("Full", "PR11_7"))
#     # goal_predicates.add(Predicate("Full", "PR10_3"))
#     # goal_predicates.add(Predicate("Full", "P5"))
#     # goal_predicates.add(Predicate("Carga", Instance("Robot", "robo1"), 100))
#     # goal_predicates.add(Predicate("Carga", Instance("Robot", "robo2"), 100))
#     return goal_predicates


def charge_preconditions(state, robot: Instance, location: Instance, grafo_mapa):
    """The robot can move if the target location is 'livre'.
    :type grafo_mapa: object
    """
    custo = 10
    CondP = False
    for p in state.predicates:
        if p.name == "Em" and p.args[0] == robot:
            for pb in state.predicates:
                if pb.name == "Btry" and pb.args[0] == robot.name and pb.args[1] < 95:
                    for ppos in state.predicates:
                        if ppos.name == "ChrPos" and ppos.args[0] == p.args[1]:
                            return True, custo

    return False, custo


def move_preconditions(state, robot: Instance, location: Instance, grafo_mapa):
    """The robot can move if the target location is 'livre'.
    :type grafo_mapa: object
    """
    custo = 10000000
    CondP = False
    for p in state.predicates:
        if p.name == "Em" and p.args[0] == robot:
            for pb in state.predicates:
                if pb.name == "Btry" and pb.args[0] == robot.name and pb.args[1] < 10:
                    return False, custo
            current = p.args[1]
            for (current_state, target_state), peso in grafo_mapa['transitions'].items():
                if current == current_state and location.name == target_state:
                    custo = peso
                    CondP = True
                    return CondP, custo

    return CondP, custo


def charge_add_effects(robot: Instance, tolocation: Instance, state, pgrafo_mapa):
    """The robot can move if the target location is 'livre'.
    :type grafo_mapa: object
    """
    new_predicates = []

    for pb in state.predicates:
        if pb.name == "Btry" and pb.args[0] == robot.name:
            new_predicates.append(Predicate("Btry", robot.name, 20000))

    return new_predicates


def move_add_effects(robot: Instance, tolocation: Instance, state, pgrafo_mapa):
    """Move the robot to a new location."""

    new_predicates = [
        Predicate("Em", robot, tolocation.name)
        # Predicate("Carga", robot)
    ]

    bateria = 100

    for p in state.predicates:
        if p.name == "Em" and p.args[0] == robot:
            for pb in state.predicates:
                if pb.name == "Btry" and pb.args[0] == robot.name:
                    bateria = pb.args[1]
                    new_predicates.append(Predicate("Btry", robot.name, bateria - 1))

    return new_predicates


def charge_del_effects(robot: Instance, tolocation: Instance, state, grafo_mapa):
    """The robot can move if the target location is 'livre'.
    :type grafo_mapa: object
    """
    new_predicates = []

    for pb in state.predicates:
        if pb.name == "Btry" and pb.args[0] == robot.name:
            new_predicates.append(Predicate("Btry", robot.name, pb.args[1]))

    return new_predicates


def move_del_effects(robot: Instance, tolocation: Instance, state, grafo_mapa):
    """Remove the previous location predicate and mark it as 'livre'."""
    currentlocation = "Error"
    for p in state.predicates:
        if p.name == "Em" and p.args[0] == robot:
            currentlocation = p.args[1]

    effects = [Predicate("Em", robot, currentlocation)]

    for p in state.predicates:
        if p.name == "Em" and p.args[0] == robot:
            for pb in state.predicates:
                if pb.name == "Btry" and pb.args[0] == robot.name:
                    bateria = pb.args[1]
                    effects.append(Predicate("Btry", robot.name, bateria))

    return effects