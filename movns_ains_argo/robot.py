import numpy as np


DISTANCE_METRIC = 'euclidean'

# Função auxiliar para calcular distâncias
def calculate_distance(point1, point2, metric='manhattan'):
    if metric == 'manhattan':
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    elif metric == 'euclidean':
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    else:
        raise ValueError("Invalid metric. Use 'euclidean' or 'manhattan'.")


class Robot:
    def __init__(self, id, battery_time, start_node_label, graph):
        self.id = id
        self.battery_time = battery_time
        self.initial_battery_time = battery_time
        self.start_node_label = start_node_label
        self.current_node_label = start_node_label
        self.allocations = []

        # Recupera posição (x, y) se houver no grafo
        node_data = graph.nodes[start_node_label]
        self.x, self.y = node_data.get("pos", (0, 0))
        self.start_node_coord_abs = (self.x, self.y)
        self.initial_position = start_node_label

    def __str__(self):
        return f"Robot {self.id} - Battery: {self.battery_time:.2f}, Start: {self.start_node_label}, (x, y): {self.x}, {self.y}"


    def can_allocate(self, task):
        return self.battery_time >= task.inspection_distance

    def allocate(self, task):
        if not self.allocations:
            last_task_id = None
        else:
            last_task_id = self.allocations[-1]

        if last_task_id is not None:
            distance_to_new_task = calculate_distance(last_task_id.coordinates, task.coordinates, DISTANCE_METRIC)

        else:
            # Se não há tarefas alocadas, considere a distância inicial como 0
            distance_to_new_task = 0

        if self.can_allocate(task) and self.battery_time >= (task.inspection_distance + distance_to_new_task):
            self.allocations.append(task)
            self.battery_time -= (task.inspection_distance + distance_to_new_task)
            return True
        
        return False
    
    def can_allocate_all(self):
        total_distance = 0
        total_inspection_distance = sum(task.inspection_distance for task in self.allocations)

        # Se o robô tem tarefas, calculamos as distâncias entre os pontos de entrada e saída
        if self.allocations:
            # Distância do robô até a primeira tarefa
            first_task = self.allocations[0]
            total_distance += calculate_distance(first_task.coordinates, (self.x, self.y), DISTANCE_METRIC)


            # Distância entre os pontos de saída e entrada das tarefas
            for i in range(len(self.allocations) - 1):
                current_task = self.allocations[i]
                next_task = self.allocations[i + 1]
                total_distance += calculate_distance(current_task.coordinates, next_task.coordinates, DISTANCE_METRIC)


            # Distância de volta para a posição inicial do robô
            last_task = self.allocations[-1]
            total_distance += calculate_distance(last_task.coordinates, (self.x, self.y), DISTANCE_METRIC)

        # Verifica se a soma do deslocamento e da inspeção está dentro da capacidade de bateria
        total_required_energy = total_inspection_distance + total_distance

        return total_required_energy <= self.battery_time
    
    def shallow_copy(self):
        return Robot(self.id, self.battery_time, self.x, self.y, self.initial_position)
    

