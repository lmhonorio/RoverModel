#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import NavSatFix
import time
import json
from threading import Lock, Thread
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MonitoraRobo:
    """Classe para monitorar a posição de cada robô"""

    def __init__(self, rotas_json_path):
        rospy.init_node('gps_listener_multirobot', anonymous=True)

        # Carrega rotas de referência do JSON
        with open(rotas_json_path, "r") as f:
            self.rotas_referencia = json.load(f)

        # Trajetória local de cada robô (atualizações em tempo real)
        self.trajetoria = {int(k.split("_")[1]): [] for k in self.rotas_referencia.keys()}

        # Controle de tempo para enviar 1 vez por segundo
        self.last_time = {int(k.split("_")[1]): 0 for k in self.rotas_referencia.keys()}

        # Lock para evitar race conditions
        self.lock = Lock()

        # Subscribers para cada robô
        for i in self.trajetoria.keys():
            rospy.Subscriber(
                f"/rover_argo_N1/Instance{i}/mavros/global_position/global",
                NavSatFix,
                self.gps_callback,
                callback_args=i
            )

    def gps_callback(self, msg, rover_id):
        """Callback de GPS para cada robô"""
        current_time = time.time()

        if current_time - self.last_time[rover_id] >= 1.0:
            self.last_time[rover_id] = current_time

            latitude = msg.latitude
            longitude = msg.longitude

            with self.lock:
                self.trajetoria[rover_id].append((latitude, longitude))
            # rospy.loginfo(f"robot_{rover_id} -> Lat: {latitude:.7f}, Lon: {longitude:.7f}")

    def listener(self):
        """Inicia o listener do ROS"""
        rospy.spin()


class VisualizadorRota:
    """Visualiza rotas planejadas e trajetórias reais"""

    def __init__(self, rotas_referencia, trajetoria):
        self.rotas_referencia = rotas_referencia
        self.trajetoria = trajetoria

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Comparação Rota Planejada x Real")
        self.ax.grid(True)
        self.ax.axis('equal')

        self.colors = ["blue", "green", "red", "purple", "orange", "cyan"]
        self.lines_ref = {}
        self.lines_real = {}

        # Plota rotas de referência
        for i, (robo, rota) in enumerate(self.rotas_referencia.items()):
            lat_ref = [p[0] for p in rota]
            lon_ref = [p[1] for p in rota]
            line_ref, = self.ax.plot(lon_ref, lat_ref, linestyle='--', marker='o',
                                     color=self.colors[i % len(self.colors)], label=f"{robo} ref")
            self.lines_ref[robo] = line_ref

            # Inicializa linha da trajetória real
            line_real, = self.ax.plot([], [], linestyle='-', marker='x',
                                      color=self.colors[i % len(self.colors)], label=f"{robo} real")
            self.lines_real[robo] = line_real

        self.ax.legend()
        self.anim = FuncAnimation(self.fig, self.update, interval=1000, blit=False)

    def update(self, frame):
        """Atualiza posições reais"""
        for robo, line in self.lines_real.items():
            robo_id = int(robo.split("_")[1])
            with monitora_robo.lock:
                traj = self.trajetoria.get(robo_id, [])
            if traj:
                lat_real = [p[0] for p in traj]
                lon_real = [p[1] for p in traj]
                line.set_data(lon_real, lat_real)
        return list(self.lines_real.values())


if __name__ == '__main__':
    ROTAS_JSON_PATH = "rotas_otimas_baseline.json"

    monitora_robo = MonitoraRobo(ROTAS_JSON_PATH)

    # ROS roda em thread separada
    ros_thread = Thread(target=monitora_robo.listener)
    ros_thread.daemon = True
    ros_thread.start()

    # Matplotlib roda na thread principal
    visualizador = VisualizadorRota(monitora_robo.rotas_referencia, monitora_robo.trajetoria)
    plt.show()
