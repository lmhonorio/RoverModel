import sys
sys.path.append('/home/milena/ardupilot/FuncoesPython')
import math
import numpy as np
import a_star
import calcular_lat_lon
import pickle
from task_priority_argo import Task
from pontos_inspecao_argo import Equipamento
from caminhos_pre_calculados_argo import CaminhosPreCalculados

# A FUNÇÃO EXECUTA_A_STAR É CHAMADA COM AJUSTES DE 1 OU DUAS UNIDADES NO DESTINO SOMENTE PARA 
# "CONCORDAR" COM A MATRIZ DE CUSTOS

# X = 197 / Y = 125 no mapa.txt
robos = {
    '1': {"x": -115 + 172 + 25, "y": 30 + 124  + 3},    #(82, 157)
    '2': {"x": -115 + 172 + 25, "y": 20 + 124 - + 3},   
    '3': {"x": -115 + 172 + 25, "y": 35 + 124  + 3},
}

# Carregando a lista de equipamentos a partir do arquivo
with open('equipamentos.pkl', 'rb') as f:
    equipamentos_carregados = pickle.load(f)

    # Registrar os equipamentos carregados
    Equipamento.equipamentos_registrados = {equip.nome: equip for equip in equipamentos_carregados}


caminhos_pre_calc = CaminhosPreCalculados('caminhos_pre_calculados_server.pkl')

def obter_coordenadas_robo(numero_robo):
    if numero_robo in robos:
        coordenadas = robos[numero_robo]
        return coordenadas["x"], coordenadas["y"]
    else:
        print("Número do robô inválido.")
        return None, None

def transform_image_to_gps(coordenadas):
    image_coords_homogeneous = np.hstack([coordenadas, 1])
    gps_coords = np.dot(image_coords_homogeneous, create_matrix_trans())
    return gps_coords

def create_matrix_trans():
        # Definir os pontos da imagem e os pontos GPS
        image_points = np.array([
            [841.3293330188126, 2510.5470425193134],
            [507.8458158872445, 1687.6656366102493],
            [572.8101374063813, 2116.430158636551]
        ])

        gps_points = np.array([
            [-41.7654929, -3.124288],  # Note que a ordem é [longitude, latitude]
            [-41.7658421, -3.1235114],
            [-41.7657578, -3.1239008]
        ])

        # Adicionar coluna de 1s para os pontos da imagem para representar coordenadas homogêneas
        image_points_homogeneous = np.hstack([image_points, np.ones((image_points.shape[0], 1))])

        # Calcular a matriz de transformação usando mínimos quadrados
        trans, residuals, rank, s = np.linalg.lstsq(image_points_homogeneous, gps_points, rcond=None)

        return trans


class Rota:
    def __init__(self):
        self.rota_completa = None

        pass

    # Função para ordenação alternada entre crescente e decrescente
    def ordenacao_alternada(index):
        # Retorna True para índices pares (crescente) e False para índices ímpares (decrescente)
        return index % 2 == 0
    
    def distancia_euclidiana(ponto1, ponto2):
        y1, x1 = ponto1
        y2, x2 = ponto2
        return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    def ponto_mais_proximo(self, pontos_regiao_ordenados, loc_atual_robo_x, loc_atual_robo_y):
        # Calcular a distância do primeiro ponto
        y_primeiro, x_primeiro = pontos_regiao_ordenados[0]
        distancia_primeiro = self.distancia_euclidiana((loc_atual_robo_y, loc_atual_robo_x), (y_primeiro + 2, x_primeiro + 2))

        # Calcular a distância do último ponto
        y_ultimo, x_ultimo = pontos_regiao_ordenados[-1]
        distancia_ultimo = self.distancia_euclidiana((loc_atual_robo_y, loc_atual_robo_x), (y_ultimo + 2, x_ultimo + 2))

        # Determinar o ponto mais próximo
        if distancia_primeiro < distancia_ultimo:
            return "primeiro"
        else:
            return "ultimo"
    
    def calcular_distancia_total(pontos):
        distancia_total = 0.0

        # Itera sobre a lista de pontos, calculando a distância entre pontos consecutivos
        for i in range(1, len(pontos)):
            x1, y1 = pontos[i - 1]
            x2, y2 = pontos[i]
            distancia = abs(x2 - x1) + abs(y2 - y1)
            distancia_total += distancia

        return distancia_total
    
    # DEFINE A ROTA EM UMA REGIAO. DETERMINA UMA ROTA EM ZIGUE-ZAGUE PASSANDO PELOS 4 PONTOS DE INTERESSE DE CADA OBJETO NA REGIÃO.
    # APÓS DEFINIR O PONTO INICIAL E FINAL DA ROTA VERIFICA-SE QUAL DELES ESTÁ MAIS PERTO DA LOCALIZAÇÃO ATUAL DO ROBÔ E CALCULA O A* ATÉ
    # AQUELE PONTO. FINALIZADA A INSPEÇÃO DA REGIÃO O ROBÔ RETORNA À BASE UTILIZANDO O A*
    def rota_regiao(self, regiao, robo):
        print("robo: ", robo)
        loc_atual_robo_x, loc_atual_robo_y = obter_coordenadas_robo(robo)

        pontos_da_rota = []
        for objeto in equipamentos_carregados:
            if getattr(objeto, 'regiao', '') == regiao:
                pontos_da_rota.extend(getattr(objeto, 'pontos_foto', []))
                print(f"Equipamento: {objeto.nome}, nos pontos: {objeto.pontos_foto}")

        qtde_equip = len(pontos_da_rota) / 4
        qtde_pontos = len(pontos_da_rota)
        # Agrupar os pontos de foto por coluna
        colunas = {}
        for ponto in pontos_da_rota:
            y, x = ponto
            if x not in colunas:
                colunas[x] = []
            colunas[x].append(ponto)

        # Ordenar os pontos de foto dentro de cada coluna
        for index, (coluna, pontos) in enumerate(colunas.items()):
            pontos.sort(key=lambda ponto: ponto[0], reverse=self.ordenacao_alternada(index))

        # Criar uma lista ordenada final
        pontos_regiao_ordenados = []
        pontos_rota_ordenados = []

        for x in sorted(colunas.keys()):
            pontos_regiao_ordenados.extend(colunas[x])
        
        ponto_proximo = self.ponto_mais_proximo(self, pontos_regiao_ordenados, loc_atual_robo_x, loc_atual_robo_y)

        if ponto_proximo == "primeiro":
            y, x = pontos_regiao_ordenados[0]
            pontos_rota_ordenados, custo = a_star.executa_a_star(loc_atual_robo_y, loc_atual_robo_x, y, x)
            pontos_rota_ordenados.extend(pontos_regiao_ordenados)
        else:
            y, x = pontos_regiao_ordenados[-1]
            pontos_rota_ordenados, custo = a_star.executa_a_star(loc_atual_robo_y, loc_atual_robo_x, y, x)
            pontos_rota_ordenados.extend(pontos_regiao_ordenados[::-1])  # Extendendo a lista invertida

        dist_percorrida = self.calcular_distancia_total(pontos_rota_ordenados)

        return pontos_rota_ordenados, qtde_equip, qtde_pontos, dist_percorrida
    
    def rota_subregiao(self, regiao, subregiao, robo):
        loc_atual_robo_x, loc_atual_robo_y = obter_coordenadas_robo(robo)
        pontos_subregiao = []
        for objeto in equipamentos_carregados:
            if getattr(objeto, 'regiao', '') == regiao and getattr(objeto, 'subregiao', '') == subregiao:
                pontos_subregiao.extend(getattr(objeto, 'pontos_foto', []))


        # Agrupar os pontos de foto por coluna
        colunas = {}
        for ponto in pontos_subregiao:
            y, x = ponto
            if x not in colunas:
                colunas[x] = []
            colunas[x].append(ponto)

        # Ordenar os pontos de foto dentro de cada coluna
        for index, (coluna, pontos) in enumerate(colunas.items()):
            pontos.sort(key=lambda ponto: ponto[0], reverse=self.ordenacao_alternada(index))

        # Criar uma lista ordenada final
        pontos_subregiao_ordenados = []
        pontos_rota_ordenados = []

        for x in sorted(colunas.keys()):
            pontos_subregiao_ordenados.extend(colunas[x])

        ponto_proximo = self.ponto_mais_proximo(self, pontos_subregiao_ordenados)
        if ponto_proximo == "primeiro":
            y, x = pontos_subregiao_ordenados[0]
            pontos_rota_ordenados, custo = a_star.executa_a_star(loc_atual_robo_y, loc_atual_robo_x, y, x)
            pontos_rota_ordenados.extend(pontos_subregiao_ordenados)
        else:
            y, x = pontos_subregiao_ordenados[-1]
            pontos_rota_ordenados, custo = a_star.executa_a_star(loc_atual_robo_y, loc_atual_robo_x, y, x)
            pontos_rota_ordenados.extend(pontos_subregiao_ordenados[::-1])  # Extendendo a lista invertida
        return pontos_rota_ordenados
    
    def rota_ponto(self, ponto):
        y, x = ponto
        pontos_rota_ordenados, custo = a_star.executa_a_star(120, 80, y, x)
        return pontos_rota_ordenados
    
    #ENVIA UM OU MAIS ROBÕS PARA INSPECIONAR UM EQUIPAMENTO
    def rota_robos_equipamento(self, robos, equipamento):
        rota_cooperativa_equipamento = {}
        caminho_robo = []
        coordenadas_equipamento = (equipamento.coordenadas[0]+1, equipamento.coordenadas[1]+1)
        loc_robos = [(130, 75), (135, 75)]
        for robo in robos:
            # loc_atual_robo_x, loc_atual_robo_y = obter_coordenadas_robo(robo)
            loc_atual_robo_x, loc_atual_robo_y = loc_robos[0]
            caminho_ate_task = caminhos_pre_calc.obter_caminho((loc_atual_robo_x, loc_atual_robo_y), coordenadas_equipamento)
                # Calcular o caminho usando A* se ele não existir
            if not caminho_ate_task:     
                caminho_ate_task = caminhos_pre_calc.calcular_e_armazenar_caminho(
                    a_star.executa_a_star, (loc_atual_robo_x, loc_atual_robo_y), coordenadas_equipamento)
            
            caminho_robo.extend(caminho_ate_task)  # Adicionar o caminho A* ao caminho completo

            # 4. Calcular o caminho de volta ao ponto inicial
            caminho_de_volta = caminhos_pre_calc.obter_caminho(coordenadas_equipamento, (loc_atual_robo_x, loc_atual_robo_y))
            
            if not caminho_de_volta:
                caminho_de_volta = caminhos_pre_calc.calcular_e_armazenar_caminho(
                    a_star.executa_a_star, coordenadas_equipamento, (loc_atual_robo_x, loc_atual_robo_y))
                
            caminho_robo.extend(caminho_de_volta)  # Adicionar o caminho de volta ao caminho completo
            rota_cooperativa_equipamento[robo] = caminho_robo

        return rota_cooperativa_equipamento



    
    def gerar_missao(pontos):
        pontos_rota_gps = [calcular_lat_lon.calculate_relative_coordinates(lat, lon) for lat, lon in pontos]
        return [(ponto_lat, ponto_lon) for ponto_lon, ponto_lat in pontos_rota_gps]
    
    def calcular_rota_task_zigue_zague(self, task):
        """
        Calcula a rota dentro de uma task usando o padrão de zigue-zague, começando no ponto de entrada (entry_point).

        :param task: Objeto task com as informações dos pontos de inspeção (a partir dos equipamentos).
        :param equipment_data: Dicionário com os dados de todos os equipamentos, acessíveis pelos IDs.
        :return: Lista com os pontos da rota em zigue-zague dentro da task.
        """
        pontos_da_rota = []  # Lista para armazenar o percurso


        # Buscar os pontos de foto dinamicamente a partir dos equipamentos associados (via equipment_ids)
        for equipment_id in task.equipment_ids:
            equipment = next(equip for equip in equipamentos_carregados if equip.nome == equipment_id)
            if equipment:
                pontos_da_rota.extend(equipment.pontos_foto)  # Adiciona os pontos de inspeção


        # Verificar se existem pontos suficientes para calcular a rota
        if not pontos_da_rota:
            return []  # Se não houver pontos, retorna uma lista vazia

        # Agrupar os pontos por colunas (ordenados pelo eixo X)
        colunas = {}
        for ponto in pontos_da_rota:
            y, x = ponto
            if x not in colunas:
                colunas[x] = []
            colunas[x].append(ponto)

        # Ordenar as colunas de acordo com o entry_point
        pontos_rota_ordenados = []
        entry_x = task.entry_point[1]

        if entry_x == min(colunas.keys()):
            # Entry point está na primeira coluna - Colunas em ordem crescente
            colunas_ordenadas = dict(sorted(colunas.items()))
        else:
            # Entry point está na última coluna - Colunas em ordem decrescente
            colunas_ordenadas = dict(sorted(colunas.items(), reverse=True))

        entry_y = task.entry_point[0]

        # Verificar se o entry_point é o menor ou maior Y da primeira ou última coluna
        if entry_x == min(colunas.keys()) or entry_x == max(colunas.keys()):
            entry_coluna = colunas[entry_x]
            if entry_y == max(entry_coluna, key=lambda ponto: ponto[0])[0]:
                # Entry point é o maior Y, começar descendo
                for index, (coluna, pontos) in enumerate(colunas_ordenadas.items()):
                    pontos.sort(key=lambda ponto: ponto[0], reverse=(index % 2 == 0))  # Alterna subindo/descendo
                    pontos_rota_ordenados.extend(pontos)
            else:
                # Entry point é o menor Y, começar subindo
                for index, (coluna, pontos) in enumerate(colunas_ordenadas.items()):
                    pontos.sort(key=lambda ponto: ponto[0], reverse=(index % 2 != 0))  # Alterna descendo/subindo
                    pontos_rota_ordenados.extend(pontos)

        return pontos_rota_ordenados  # Retorna a lista de pontos ordenados em zigue-zague




    def calcular_rota_completa(self, solution):
        """
        Calcula a rota completa de cada robô, incluindo o caminho entre tasks com A* e o percurso em zigue-zague dentro de cada task.
        Se o caminho já existir no arquivo .pkl, ele será carregado, caso contrário será calculado e salvo.

        :param solution: Objeto solution contendo a alocação de tasks e os robôs.
        :param caminhos_pre_calculados: Objeto da classe CaminhosPreCalculados para armazenar e carregar caminhos.
        :return: Dicionário contendo as rotas completas de todos os robôs.
        """
        rotas_completas = {}  # Dicionário para armazenar a rota de cada robô
        # Loop para calcular e armazenar as rotas de todos os robôs
        for robot_id in range(len(solution.robots)):
            robot = solution.robots[robot_id]
            caminho_completo = []  # Caminho completo do robô
            current_position = (robot.x, robot.y)  # Posição inicial do robô
            print(f"robot: {robot}")
            print(f"robot_id: {robot_id}")
            # Percorrer as tasks alocadas ao robô
            for i, task in enumerate(solution.allocations[robot_id]):
                # Definir as chaves para verificar no dicionário                
                # Verificar se o caminho já existe no dicionário
                caminho_ate_task = caminhos_pre_calc.obter_caminho(current_position, task.entry_point)
                # Calcular o caminho usando A* se ele não existir
                if not caminho_ate_task:     
                    caminho_ate_task = caminhos_pre_calc.calcular_e_armazenar_caminho(
                        a_star.executa_a_star, current_position, task.entry_point)
                
                caminho_completo.extend(caminho_ate_task)  # Adicionar o caminho A* ao caminho completo
                
                # 2. Executar a inspeção na task em zigue-zague (rota dentro da task)
                rota_task = self.calcular_rota_task_zigue_zague(task)
                caminho_completo.extend(rota_task)  # Adicionar a rota da task ao caminho completo
                
                # 3. Atualizar a posição atual do robô para o ponto de saída da task
                current_position = task.exit_point

            # 4. Calcular o caminho de volta ao ponto inicial
            caminho_de_volta = caminhos_pre_calc.obter_caminho(current_position, (robot.x, robot.y))
            
            if not caminho_de_volta:
                caminho_de_volta = caminhos_pre_calc.calcular_e_armazenar_caminho(
                    a_star.executa_a_star, current_position, (robot.x, robot.y))
                
            caminho_completo.extend(caminho_de_volta)  # Adicionar o caminho de volta ao caminho completo

            # Armazenar a rota completa no robô
            robot.rota_completa = caminho_completo
            
            # Adicionar ao dicionário de rotas
            rotas_completas[robot_id] = caminho_completo

        return rotas_completas