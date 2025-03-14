from queue import PriorityQueue
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation as nx_app
from networkx.drawing.nx_agraph import to_agraph
import itertools
import matplotlib.colors as mcolors


class Grafo:
    def __init__(self):
        return

    def xml_to_graph(self, graphxml):
        # Criar um grafo não direcionado
        G = nx.Graph()

        # Adicionar estados
        G.add_nodes_from(graphxml['states'])

        # Adicionar transições
        for (current_state, target_state), nweight in graphxml['transitions'].items():
            G.add_edge(current_state, target_state, label=nweight, weight=nweight)

        # Definir estados finais
        if 'accepting_states' in graphxml and graphxml['accepting_states']:
            for state in graphxml['accepting_states']:
                G.nodes[state]['accepting_state'] = True
                G.nodes[state]['shape'] = 'doublecircle'

        if 'start' in graphxml and graphxml['start']:
            G.graph['start'] = graphxml['start']
            G.nodes[graphxml['start']]['color'] = 'red'

        # Agrupar nós com mesmo nome e índices diferentes
        node_groups = {}
        for node in G.nodes:
            if '_' in node:
                base_name = node.rsplit('_', 1)[0]  # Obtém o nome sem o índice
                if base_name not in node_groups:
                    node_groups[base_name] = []
                node_groups[base_name].append(node)

        # Gerar cores distintas para cada grupo
        color_palette = itertools.cycle(mcolors.TABLEAU_COLORS.values())
        node_colors = {}

        for base_name, nodes in node_groups.items():
            color = next(color_palette)
            for node in nodes:
                G.nodes[node]['color'] = color

        return G


class Instance:
    def __init__(self, instance_type, name):
        self.instance_type = instance_type
        self.name = name


    def __eq__(self, other):
        return isinstance(other, Instance) and self.instance_type == other.instance_type and self.name == other.name

    def __hash__(self):
        return hash((self.instance_type, self.name))

    def __repr__(self):
        return f"{self.name}"
        #return f"{self.instance_type}({self.name})"

class Predicate:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __eq__(self, other):
        return self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, self.args))

    def __repr__(self):
        return f"{self.name}({', '.join(str(arg) for arg in sorted(self.args,key=str))})"

class State:
    def __init__(self, predicates, robot_costs):
        self.predicates = set(sorted(predicates,key=str))
        self.robot_costs = robot_costs  # Dictionary: {robot: cost}

    def apply(self, operator, robot, target=None, state=None, state_map=None):
        """Apply an operator to the state, returning a new state with updated costs for the robot."""
        isapplicable, peso = operator.is_applicable(self, robot, target, state_map)
        if not isapplicable:
            return None

        # Apply the operator's effects
        new_set = list(operator.add_effects(robot, target, state, state_map))
        new_predicates = list(self.predicates)  # Convert set to list

        # Use extend() to add elements from new_set to new_predicates
        new_predicates.extend(new_set)

        # Get the predicates to be deleted and remove them from the list
        deleted_predicates = list(operator.del_effects(robot, target, state, state_map))
        for p in deleted_predicates:
            if p in new_predicates:
                new_predicates.remove(p)

        # Update the robot costs
        new_robot_costs = peso + self.robot_costs

        # Return a new State object with the updated predicates and costs
        return State(set(new_predicates), new_robot_costs)

    def total_cost(self):
        """Sum the costs of all robots"""
        return sum(self.robot_costs.values())

    def __eq__(self, other):
        return self.predicates == other.predicates and self.robot_costs == other.robot_costs

    def __hash__(self):
        return hash((frozenset(self.predicates), self.robot_costs))

    def __repr__(self):
        return f"State(predicates={self.predicates}, costs={self.robot_costs})"

    def __lt__(self, other):
        """Define a comparação entre estados pelo custo"""
        return self.robot_costs < other.robot_costs  # Ou qualquer outro critério adequado

class Operator:
    def __init__(self, name, preconditions_func, add_effects_func, del_effects_func, cost):
        self.name = name
        self.preconditions_func = preconditions_func
        self.add_effects_func = add_effects_func
        self.del_effects_func = del_effects_func
        self.cost = cost

    def is_applicable(self, state, robot, target=None, grafo_mapa=None):
        """Check if the operator can be applied with the given robot and target (e.g., location)"""
        return self.preconditions_func(state, robot, target, grafo_mapa)

    def add_effects(self, robot, target=None, state=None, grafo_mapa=None):
        return self.add_effects_func(robot, target, state, grafo_mapa)

    def del_effects(self, robot, target=None, state=None, grafo_mapa=None):
        return self.del_effects_func(robot, target, state, grafo_mapa)

    def __repr__(self):
        return f"{self.name} (cost: {self.cost})"
