import sys
import time

# import movns_ains
sys.path.append('/home/milena/catkin_ws/src/ardupilot_gazebo/scripts/otimization')

from flask import Flask, request, jsonify, send_file  # Adicione send_file ao import
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from plotagem_server_argo import Mapa
import pickle
import send_mission_argo
from mavros_msgs.msg import WaypointReached
import rospy
from robot import Robot




# Função para mapear de volta da matriz para o mundo
def matrix_to_world(lista_pontos):
    pontos_mundo = []
    x_offset = 124
    y_offset = 172
    for x, y in lista_pontos:
        world_x = x - x_offset - 3
        world_y = y - y_offset - 25
        pontos_mundo.append((world_x, world_y))
    return pontos_mundo


""" def enviar_waypoints_para_controller(rotas_completas):
    rospy.wait_for_service('/send_waypoints')
    try:
        send_waypoints = rospy.ServiceProxy('/send_waypoints', SendWaypoints)

        waypoints_flattened = []
        waypoint_indices = []
        robot_ids = []

        current_index = 0
        for robot_id, waypoints in rotas_completas.items():
            robot_ids.append(robot_id)
            waypoints_flattened.extend([Point(x=wp[0], y=wp[1], z=0.0) for wp in waypoints])
            waypoint_indices.append(current_index)
            current_index += len(waypoints)
        print(f"waypoints_flattened: {waypoints_flattened}")
        response = send_waypoints(robot_ids=robot_ids, waypoints=waypoints_flattened, waypoint_indices=waypoint_indices)
        if response.success:
            rospy.loginfo(response.message)
        else:
            rospy.logerr(f"Erro ao enviar waypoints: {response.message}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Falha ao chamar o serviço: {e}") """


app = Flask(__name__)
# Configuração de CORS para o Flask
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

mapa = Mapa("/home/milena/catkin_ws/src/ardupilot_gazebo/scripts/Figuras/mapaNovo.png")

trajetoria = []
trajetoria_gps = []
lista_pontos_mundo = []
equip = []
quantidade_equipamentos = 0
qtde_pontos = 0
dist_percorrida = 0
tasks_dict = {}  # Dicionário para armazenar todas as tasks

plot_path = None  # Inicialize a variável plot_path

# Inicialize o nó ROS na thread principal
rospy.init_node('waypoint_monitor', anonymous=True)




def gerar_missao_tasks(robot_id, waypoints):
    global trajetoria_gps
    trajetoria_gps.clear()
    # trajetoria_gps = [calcular_lat_lon.calculate_relative_coordinates(lat, lon) for lat, lon in waypoints]
    print("calculei trajetoria gps")
    send_mission_argo.generate_mission(trajetoria_gps, robot_id)
    return f"Missão gerada para o Robô {robot_id}."

def sair():
    return "Saindo..."

# Função de monitoramento de waypoint alcançado para todos os robôs
def monitor_waypoint_reached():

    def waypoint_callback(data, robot_id):
        print(f"Waypoint alcançado para robô {robot_id}: ", data.wp_seq)
        socketio.emit(f'mission_update_{robot_id}', {'status': 'Executando Missão', 'current_wp': data.wp_seq})

    try:
        rospy.Subscriber('/rover_1/mavros/mission/reached', WaypointReached, waypoint_callback, callback_args='1')
        rospy.Subscriber('/rover_2/mavros/mission/reached', WaypointReached, waypoint_callback, callback_args='2')
        rospy.Subscriber('/rover_3/mavros/mission/reached', WaypointReached, waypoint_callback, callback_args='3')
        print("Subscrições aos tópicos realizadas com sucesso.")
        rospy.spin()  # Mantém o nó ativo
    except Exception as e:
        print("Erro no monitoramento de waypoints:", e)

    
@app.route('/get_plot_image', methods=['GET'])
def get_plot_image():
    global plot_path  # Acesso à variável global plot_path
    return send_file(plot_path, mimetype='image/png')  # Retorna o arquivo da imagem com o tipo MIME correto

@app.route('/execute', methods=['POST'])
def execute():
    option = request.json.get("option")

    switcher = {
        8: gerar_missao_tasks,
        10: sair,
    }

    func = switcher.get(option, lambda: "Opção inválida")
    output = func()
    if option == 5:
        return jsonify({'output': output, 'plotUrl': '/Figuras/plot.png', 'outputList': trajetoria})  # Inclua plotUrl no retorno
    elif option == 7:
        return jsonify({'output': output, 'outputList': trajetoria_gps})
    elif option == 1:
        return jsonify({'output': output, 'plotUrl': '/Figuras/plot.png', 'outputList': lista_pontos_mundo, "equipments_count": quantidade_equipamentos, "pontos_count": qtde_pontos, "dist_percorrida": dist_percorrida})
    else:
        return jsonify({'output': output, 'outputList': trajetoria})  # Apenas 'output' para outras opções

if __name__ == "__main__":
    socketio.start_background_task(target=monitor_waypoint_reached)
    app.run(port=5001)

