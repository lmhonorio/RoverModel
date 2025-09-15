#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import NavSatFix
import time
import requests
from threading import Lock

class MonitoraRobo:
    """Classe para monitorar a posição de cada robô"""

    def __init__(self, servidor_url="http://127.0.0.1:5000"):

        rospy.init_node('gps_listener_multirobot', anonymous=True)

        self.servidor_url = servidor_url

        # Trajetória local de cada robô
        self.trajetoria = {1: [], 2: [], 3: []}

        # Controle de tempo para enviar 1 vez por segundo
        self.last_time = {1: 0, 2: 0, 3: 0}

        # Lock para evitar race conditions
        self.lock = Lock()

        # Subscribers para cada robô
        rospy.Subscriber("/rover_argo_N1/Instance1/mavros/global_position/global", NavSatFix, self.gps_callback, callback_args=1)
        # rospy.Subscriber("/rover_argo_N1/Instance2/mavros/global_position/global", NavSatFix, self.gps_callback, callback_args=2)
        # rospy.Subscriber("/rover_argo_N1/Instance3/mavros/global_position/global", NavSatFix, self.gps_callback, callback_args=3)

    def gps_callback(self, msg, rover_id):
        """Callback de GPS para cada robô"""

        current_time = time.time()

        # Salva apenas se passou mais de 1s desde a última vez
        if current_time - self.last_time[rover_id] >= 1.0:
            self.last_time[rover_id] = current_time

            # Extrai latitude e longitude
            latitude = msg.latitude
            longitude = msg.longitude

            # Armazena na lista
            with self.lock:
                self.trajetoria[rover_id].append((latitude, longitude))
            rospy.loginfo(f"Robo {rover_id} -> Lat: {latitude:.7f}, Lon: {longitude:.7f}")
            
            # Envia para o servidor Flask
            try:
                payload = {"robo": rover_id, "latitude": latitude, "longitude": longitude}
                response = requests.post(f"{self.servidor_url}/send_gps", json=payload, timeout=1)
                if response.status_code != 200:
                    rospy.logwarn(f"Falha ao enviar GPS do Robo {rover_id}: {response.text}")
            except requests.exceptions.RequestException as e:
                rospy.logwarn(f"Erro de conexão ao enviar GPS do Robo {rover_id}: {e}")

    def listener(self):
        """Inicia o listener do ROS"""
        rospy.spin()

if __name__ == '__main__':
    try:
        servidor = "http://127.0.0.1:5000"
        monitora_robo = MonitoraRobo(servidor)
        monitora_robo.listener()
    except rospy.ROSInterruptException:
        pass
