import sys
import time

# import movns_ains
sys.path.append('/home/milena/catkin_ws/src/ardupilot_gazebo/scripts/otimization')

from flask import Flask, request, jsonify, send_file  # Adicione send_file ao import
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import rotas
from plotagem_server_argo import Mapa
import pickle
import send_mission_argo
import calcular_lat_lon
from mavros_msgs.msg import WaypointReached
import rospy
from robot import Robot
import movns_ains_argo
from task_priority_argo import Task
from ardupilot_gazebo.srv import SendWaypoints
from geometry_msgs.msg import Point
from controller_argo import Controller
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carregando a lista de equipamentos a partir do arquivo
with open('equipamentos.pkl', 'rb') as f:
    equipamentos_carregados = pickle.load(f)

# Função para buscar o equipamento
def encontrar_equipamento_por_nome(nome, equipamentos):
    for equipamento in equipamentos:
        if nome in str(equipamento.nome):  # Busca pelo nome no string do equipamento
            return equipamento
    return None



def plot_points_with_background(data, background_image=None):
    """
    Plots points from the provided data and includes an optional background image.

    Args:
        data (dict): A dictionary where keys are labels (e.g., robot IDs) and values are lists of (x, y) tuples.
        background_image (str, optional): Path to a background image. Defaults to None.
    """
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the background image if provided
    if background_image:
        img = mpimg.imread(background_image)
        ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]], aspect='auto')

    # Plot points for each key in the data
    for key, points in data.items():
        x_coords, y_coords = zip(*points)  # Unzip the list of tuples
        ax.plot(x_coords, y_coords, marker='o', label=f'Robot {key}')

    # Add labels and legend
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.legend()
    ax.set_title('Points Plot with Background')

    # Show grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.savefig('output_plot.png')


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
rota = rotas.Rota()
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

def load_all_tasks():
    
    global tasks_dict
    if not tasks_dict:  # Carrega apenas uma vez
        # Suponha que as tarefas estão em um arquivo ou banco de dados. Por exemplo, em um arquivo 'tasks.pkl'
        try:
            with open('tasks_server.pkl', 'rb') as f:
                all_tasks = pickle.load(f)  # Carrega todas as tasks
                
                tasks_dict = {task.id: task for task in all_tasks}  # Converte para dicionário com nomes como chaves
            print("Todas as tarefas foram carregadas com sucesso.")
        except FileNotFoundError:
            print("Arquivo de tarefas 'tasks.pkl' não encontrado.")
    return tasks_dict

def get_task_by_name(task_name):
    """Retorna o objeto Task correspondente ao task_name."""
    # Garante que todas as tasks foram carregadas
    load_all_tasks()
    # Retorna a tarefa pelo nome, ou None se não for encontrada
    return tasks_dict.get(task_name, None)

def rota_regiao():
    trajetoria.clear()
    robot_number = request.json.get("robotNumber")
    regiao = request.json.get("regiao")
    pontos_foto_ordenados, qtde_equip, qtde_pontos, dist_percorrida = rota.rota_regiao(rota, regiao, robot_number)
    trajetoria.extend(pontos_foto_ordenados)
    lista_pontos_mundo.extend(matrix_to_world(pontos_foto_ordenados))
    print("Pontos mundo: ", trajetoria)
    return {"output": "Rota na região '{}' realizada.".format(regiao), "outputList": lista_pontos_mundo, "equipments_count": qtde_equip, "pontos_count": qtde_pontos, "dist_percorrida": dist_percorrida}


"""
def rota_completa():
    trajetoria.clear()
    # Recebe as tasks em formato JSON e converte para objetos
    tasks_data = request.json.get("tasks")
    robots_data = request.json.get("robots")
    return "Função rota_completa a ser implementada."

def rota_subregiao():
    trajetoria.clear()
    robot_number = request.json.get("robotNumber")
    regiao = request.json.get("regiao")
    subregiao = request.json.get("subregiao")
    pontos_foto_ordenados = rota.rota_subregiao(rota, regiao, subregiao, robot_number)
    trajetoria.extend(pontos_foto_ordenados)
    quantidade_equipamentos = len(trajetoria)
    return {"output": "Rota na subregião '{}' da região '{}' realizada.".format(subregiao, regiao), "outputList": trajetoria, "equipments_count": quantidade_equipamentos}

def ir_para_ponto():
    trajetoria.clear()
    
    nome = request.json.get("nome")
    regiao = request.json.get("regiao")
    subregiao = request.json.get("subregiao")
    for objeto in equipamentos_carregados:
        if (nome.lower() in objeto.nome.lower() and
            regiao.lower() in objeto.regiao.lower() and
            subregiao.lower() in objeto.subregiao.lower()):
            equipamento = objeto
            break
    ponto = equipamento.coordenadas
    caminho = rota.rota_ponto(rota, ponto)
    trajetoria.extend(caminho)
    return "Inspeção do equipamento '{}' na região '{}' realizada.".format(nome, regiao)
"""

def ir_para_equipamento():
    trajetoria.clear()
    equipment_name = request.json.get("equipment")
    robots = request.json.get("robots")
    print(f"equipment: {equipment_name}, robots: {robots} ")

    # Encontra o equipamento na lista
    equipamento = encontrar_equipamento_por_nome(equipment_name, equipamentos_carregados)
    # print(equipamento)

    if not equipamento:
        return {"success": False, "message": f"Equipamento {equipment_name} não encontrado."}

    rotas_robos_equipamento=  rota.rota_robos_equipamento(robots, equipamento)
    print(f"rotas robos equipamentos: {rotas_robos_equipamento}")

    for robot_id, waypoints in rotas_robos_equipamento.items():
        print(waypoints)
        gerar_missao_tasks(robot_id, waypoints)
        # Adiciona um intervalo de 15 segundos entre os envios
        time.sleep(15)

    # Retorna as rotas para o frontend
    return {
        "success": True,
        "message": "Rotas geradas com sucesso.",
        "routes": [
            {"robot": robot_id, "path": waypoints}
            for robot_id, waypoints in rotas_robos_equipamento.items()
        ]
    }

"""
def plotar_caminho():
    global plot_path  # Adicione esta linha para acessar a variável global plot_path
    plot_path = Mapa.desenhar_rota(mapa, trajetoria)
    print("Caminho do plot da rota:", plot_path)  # Adicione este print para depuração
    return {"output": 'Rota na região', 'plotUrl': 'Figuras/plot.png'}

def printar_caminho():
    return str(trajetoria)

def gerar_rota_georaferenciada():
    global trajetoria_gps
    trajetoria_gps = rota.gerar_missao(trajetoria)
    return str(trajetoria_gps)
"""

def gerar_missao_tasks(robot_id, waypoints):
    global trajetoria_gps
    trajetoria_gps.clear()
    trajetoria_gps = [calcular_lat_lon.calculate_relative_coordinates(lat, lon) for lat, lon in waypoints]
    print("calculei trajetoria gps")
    send_mission_argo.generate_mission(trajetoria_gps, robot_id)
    return f"Missão gerada para o Robô {robot_id}."

def rotear_tasks():
    # Limpa trajetórias antigas
    trajetoria.clear()
    # Recebe as tasks em formato JSON e converte para objetos
    tasks_data = request.json.get("tasks")
    robots_data = request.json.get("robots")
    if not tasks_data:
        return {"error": "Tasks not found in request"}, 400
    tasks = []

    for task_name in tasks_data:
        task = get_task_by_name(task_name)
        if task:
            tasks.append(task)
        else:
            print(f"Tarefa '{task_name}' não encontrada.")

    # Agora 'tasks' é uma lista de objetos Task que você pode manipular com o MRTA
    # Define a configuração dos robôs
    num_robots = 3  # Pode ser ajustado ou recebido de outro lugar
    initial_positions = [(130, 75), (135, 75), (140, 75)]
    battery_times = [15000, 15000, 15000]
    robots = [Robot(i, battery_times[i], x=initial_positions[i][0], y=initial_positions[i][1]) for i in range(num_robots)]

    final_population = movns_ains_argo.run_movns(robots, tasks)
    best_solution = min(final_population, key=lambda sol: sol.time)

    # Salvar a solução em um .pkl
    with open("/home/milena/ardupilot_gazebo/best_solution.pkl", "wb") as f:
        pickle.dump(best_solution, f)
    print("✅ Best solution salva com sucesso!")

    # Chamar a função para desenhar as tasks no mapa
    # teste_gerar_figura_artigo.desenhar_tasks_no_mapa(best_solution)

    """ rotas_completas = rota.calcular_rota_completa(best_solution)

    plot_points_with_background(rotas_completas, background_image=None)

    for robot_id, waypoints in rotas_completas.items():
        gerar_missao_tasks(robot_id, waypoints) """
        
    

    """ # Cria e retorna a trajetória dos robôs
    robot_paths = {robot.id: robot.allocations for robot in best_solution.robots}
    print(robot_paths)  # Para ver a alocação final
    return {"robot_paths": robot_paths} """

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
        1: rota_regiao,
        2: rota_completa,
        3: rota_subregiao,
        4: ir_para_ponto,
        5: plotar_caminho,
        6: printar_caminho,
        7: gerar_rota_georaferenciada,
        8: gerar_missao_tasks,
        9: rotear_tasks,
        10: sair,
        11: ir_para_equipamento
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

