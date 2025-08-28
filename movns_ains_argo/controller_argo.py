#!/usr/bin/env python3

import rospy
from actionlib import SimpleActionClient
from ardupilot_gazebo.msg import MissionAction, MissionGoal
import calcular_lat_lon
from mavros_msgs.msg import State

class Controller:
    def __init__(self, shared_ros_context=False):
        if not shared_ros_context:
            rospy.init_node('mission_controller')  # Apenas inicializa se não for compartilhado
        # Configuração inicial como antes...
        self.robot_mapping = {
            'rover_1': 1,
            'rover_2': 2,
            'rover_3': 3
        }

        self.clients = {
            name: SimpleActionClient(f'/{name}/mission_server', MissionAction)
            for name in self.robot_mapping.keys()
        }

        self.states = {name: None for name in self.robot_mapping.keys()}
        for name in self.robot_mapping.keys():
            rospy.Subscriber(f'/{name}/mavros/state', State, self.state_callback, callback_args=name)

        self.missions = {}  # Inicialmente vazio



    def receive_mission(self, robot_id, waypoints):
        """
        Recebe missões de forma centralizada para um robô.

        Args:
            robot_id (int): ID do robô.
            waypoints (list of tuples): Lista de waypoints [(x1, y1), (x2, y2)].
        """
        robot_name = f'rover_{robot_id}'
        if robot_name not in self.robot_mapping:
            rospy.logwarn(f"Robô {robot_name} não encontrado no mapeamento.")
            return

        pontos_rota_gps = [
            calcular_lat_lon.calculate_relative_coordinates(x, y) for x, y in waypoints
        ]
        self.missions[robot_name] = pontos_rota_gps
        rospy.loginfo(f"Missão recebida para {robot_name}: {pontos_rota_gps}")




    def state_callback(self, msg, robot_name):
        """Callback para atualizar o estado do robô."""
        self.states[robot_name] = msg.mode  # Atualiza o modo atual do robô
        rospy.loginfo(f"Estado do {robot_name}: {msg.mode}")

    def check_robot_status(self, event):
        """Verifica se algum robô está pronto para receber uma nova missão."""
        for name, state in self.states.items():
            if state == 'MANUAL' and not self.missions[name].is_complete():
                rospy.loginfo(f"{name} está pronto para receber uma missão.")
                self.send_next_chunk(name)

    def send_next_chunk(self, robot_name):
        """Envia o próximo pacote de waypoints para o robô."""
        mission = self.missions[robot_name]
        if mission.is_complete():
            rospy.loginfo(f"Todas as missões para {robot_name} foram concluídas.")
            return

        chunk = mission.get_next_chunk()
        flattened_points = self.flatten_waypoints(chunk)
        goal = MissionGoal(robot_id=mission.robot_id, mission_points=flattened_points)

        rospy.loginfo(f"Enviando pacote {mission.current_chunk}/{mission.total_chunks} para {robot_name}.")

        self.clients[robot_name].send_goal(
            goal,
            done_cb=lambda status, result: self.mission_done_callback(robot_name, status, result),
            feedback_cb=lambda feedback: self.mission_feedback_callback(robot_name, feedback)
        )

    def mission_done_callback(self, robot_name, status, result):
        """Callback quando um pacote de missão é concluído."""
        if result.success:
            rospy.loginfo(f"Pacote concluído para {robot_name}.")
            self.send_next_chunk(robot_name)
        else:
            rospy.logwarn(f"Falha ao concluir pacote para {robot_name}: {result.message}")

    def mission_feedback_callback(self, robot_name, feedback):
        """Recebe feedback durante a execução da missão."""
        rospy.loginfo(f"Feedback do {robot_name}: Waypoint atual - {feedback.current_waypoint}")

    def flatten_waypoints(self, waypoints):
        """Converte uma lista de tuplas (x, y) para uma lista plana de floats."""
        return [float(coord) for point in waypoints for coord in point]

    def preempt_mission(self, robot_name):
        """Cancela a missão atual do robô."""
        if robot_name in self.clients:
            rospy.logwarn(f"Cancelando missão para {robot_name}.")
            self.clients[robot_name].cancel_goal()
        else:
            rospy.logerr(f"Robô {robot_name} não encontrado.")
