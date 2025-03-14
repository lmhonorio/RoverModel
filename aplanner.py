import heapq
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation as nx_app
from networkx.drawing.nx_agraph import to_agraph
from baseclasses import *


class AStarPlanner:
    def __init__(self, initial_state, goal_predicates_func, operators, robots, grafo_mapa, get_goal_state_predicates):
        self.initial_state = initial_state
        self.goal_predicates_func = goal_predicates_func  # Function to generate goal conditions dynamically
        self.operators = operators
        self.robots = robots
        self.grafo_mapa = grafo_mapa
        self.get_goal_state_predicates = get_goal_state_predicates

    def heuristic(self, state):
        """Simple heuristic: number of unsatisfied goal predicates"""
        goal_predicates = self.goal_predicates_func()
        return len(goal_predicates - state.predicates)

    def is_goal(self, state):
        """Check if the state satisfies the goal conditions"""
        goal_predicates = self.goal_predicates_func()
        goal_state_predicates = self.get_goal_state_predicates(state)
        return goal_predicates.issubset(goal_state_predicates)

    def plan(self):
        newlocations = [Instance("Place", i) for i in sorted(self.grafo_mapa['states'], key=str)]

        open_set = []
        heapq.heappush(open_set, (0, self.initial_state))  # Priority Queue (min-heap)
        came_from = {}

        g_score = {self.initial_state: 0}  # Melhor custo conhecido até cada estado
        visited = set()  # Estados já processados

        while open_set:
            _, current = heapq.heappop(open_set)  # Obtém estado com menor custo

            # Se o estado é objetivo, reconstruir caminho
            if self.is_goal(current):
                print("Goal reached!")
                return self.reconstruct_path(came_from, current)

            visited.add(current)

            for operator in self.operators:
                for robot in self.robots:
                    for location in newlocations:
                        isapplicable, peso = operator.is_applicable(current, robot, location, self.grafo_mapa)
                        if not isapplicable:
                            continue

                        next_state = current.apply(operator, robot, location, state=current, state_map=self.grafo_mapa)
                        if next_state is None:
                            continue
                        if next_state in visited:
                            continue

                        # Calcular custo atualizado
                        tentative_g_score = g_score[current] + peso

                        # Atualiza somente se encontramos um caminho melhor
                        if next_state not in g_score or tentative_g_score < g_score[next_state]:
                            g_score[next_state] = tentative_g_score
                            priority = tentative_g_score + self.heuristic(next_state)
                            heapq.heappush(open_set, (priority, next_state))
                            came_from[next_state] = (current, operator, robot, location)

        print("No plan found")
        return None  # No plan found

    def reconstruct_path(self, came_from, current):
        path = []
        final = current
        while current in came_from:
            current, operator, robot, location = came_from[current]
            path.insert(0, (operator, robot, location))
        return path, final
