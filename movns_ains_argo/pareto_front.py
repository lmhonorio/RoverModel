import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Pareto_Front:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.solutions = []

    def update_pareto_front(self, new_solution):
        """
        Tenta adicionar uma nova solução ao pool de soluções de Pareto.
        Retorna True se a solução foi adicionada, False caso contrário.
        """
        dominated_indices = []

        # Verifica se a nova solução já existe no pool
        for sol in self.solutions:
            if self.equals(sol, new_solution):
                # print("SOLUÇOES IGUAIS")
                return False  # A solução já existe no pool

        # Verifica se a nova solução domina alguma solução no pool
        for i, existing_solution in enumerate(self.solutions):
            if self.dominates(new_solution, existing_solution):
                # print("NOVA SOLUÇÃO DOMINA")
                dominated_indices.append(i)  # Coleta o índice da solução dominada

            # Verifica se alguma solução existente domina a nova solução
            elif self.dominates(existing_solution, new_solution):
                return False  # A nova solução não será adicionada pois é dominada

        # Remove todas as soluções dominadas
        for index in sorted(dominated_indices, reverse=True):  # Remove em ordem reversa para evitar problemas de reindexação
            self.remove_solution(index)

        # for sol in self.solutions:
            # sol.print_solution_metrics()

        # Adiciona a nova solução ou lida com excesso de capacidade
        if len(self.solutions) < self.capacity:
            self.add_solution(new_solution)
        else:
            self.handle_over_capacity(new_solution)

        return True

    def handle_over_capacity(self, new_solution):
        # print("entrei no handle over capacity")
        """
        Lida com o excesso de capacidade da frente de Pareto.
        Remove soluções com base na distância de crowding.
        """
        all_solutions = self.solutions + [new_solution]
        metrics = np.array([sol.metrics for sol in all_solutions])

        # Calcula as frentes de Pareto
        fronts = self.calculate_fronts(metrics)
        survivors = []

        # Adiciona soluções das frentes até atingir a capacidade
        for front in fronts:
            if len(survivors) + len(front) <= self.capacity:
                survivors.extend(front)
            else:
                # Calcula a distância de crowding para o último front
                crowding_distances = self.calculate_crowding_distance(metrics, front)

                # Ordena pela distância de crowding em ordem decrescente
                sorted_by_crowding = sorted(front, key=lambda x: crowding_distances[x], reverse=True)

                # Adiciona as soluções restantes até atingir a capacidade
                survivors.extend(sorted_by_crowding[:self.capacity - len(survivors)])
                break

        # Atualiza as soluções com os sobreviventes
        self.solutions = [all_solutions[i] for i in survivors]
        """ if new_solution in self.solutions:
            print("NOVA SOLUCAO ENTROU") """

    def dominates(self, sol1, sol2):
        """
        Verifica se sol1 domina sol2 com base nos atributos de interesse.
        """
        return (
            sol1.distance <= sol2.distance and
            sol1.time <= sol2.time and
            sol1.balance_load <= sol2.balance_load and
            (
                sol1.distance < sol2.distance or
                sol1.time < sol2.time or
                sol1.balance_load < sol2.balance_load
            )
        )

    def add_solution(self, new_solution):
        """
        Adiciona uma nova solução ao pool.
        """
        self.solutions.append(new_solution)

    def remove_solution(self, index):
        """
        Remove uma solução do pool pelo índice.
        """
        del self.solutions[index]

    def equals(self, sol1, sol2):
        """
        Verifica se duas soluções são idênticas (baseado em distância, tempo e balanceamento).
        """
        return (
            sol1.distance == sol2.distance and
            sol1.time == sol2.time and
            sol1.balance_load == sol2.balance_load
        )

    def calculate_fronts(self, metrics):
        """
        Divide a população em frentes de Pareto.
        """
        non_dominated_sorting = NonDominatedSorting()
        fronts = non_dominated_sorting.do(metrics)
        return fronts

    def calculate_crowding_distance(self, metrics, front):
        """
        Calcula a distância de crowding para um conjunto de indivíduos.
        """
        n_objectives = metrics.shape[1]
        distances = np.zeros(len(front))

        for i in range(n_objectives):
            obj_values = metrics[front, i]
            sorted_indices = np.argsort(obj_values)
            sorted_front = np.array(front)[sorted_indices]

            # Distância infinita para os extremos
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Normalizar os valores do objetivo
            norm = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
            if norm == 0:
                continue

            for j in range(1, len(sorted_front) - 1):
                distances[sorted_indices[j]] += (
                    obj_values[sorted_indices[j + 1]] - obj_values[sorted_indices[j - 1]]
                ) / norm

        return {front[i]: distances[i] for i in range(len(front))}
