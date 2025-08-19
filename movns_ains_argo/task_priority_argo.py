from collections import defaultdict
import math

def distancia(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def encontrar_idx_mais_proximo(rota, alvo):
    return min(range(len(rota)), key=lambda i: distancia(rota[i]['coord_abs'], alvo))

class Task:
    def __init__(self, id, x, y, inspection_time, inspection_distance, entry_point, exit_point, observation_points, aabb_info):
        self.id = id
        self.x = x
        self.y = y
        self.inspection_time = inspection_time
        self.inspection_distance = inspection_distance
        self.entry_point = entry_point
        self.exit_point = exit_point
        self.coordinates = entry_point
        self.possible_entry_points = []
        # self.equipments = []  # Lista de equipamentos associados a essa Task
        self.equipment_ids = []  # Em vez de salvar os equipamentos, salve os IDs ou referências
        self.id_raw = self.extract_id_raw()  # Chama o método ao criar a Task
        self.observation_points = observation_points
        
        # 🔁 GERA A ROTA COM BASE NOS PONTOS
        self.rota_task(aabb_info)

    def extract_id_raw(self):
        """
        Extrai o número do ID da task, assumindo que o ID tem o formato 'task_X',
        onde X é um número.
        """
        try:
            return int(self.id.split('_')[-1])-1
        except (ValueError, AttributeError):
            raise ValueError(f"O id '{self.id}' não está no formato esperado 'task_X'.")
    
    def add_equipment(self, equipment):
        self.equipments.append(equipment)


    def get_associated_equipments(self, equipment_data):
        # Quando precisar dos equipamentos, carregue-os dinamicamente a partir dos IDs
        return [equipment_data[id] for id in self.equipment_ids]
    
    def calcular_distancia_total(self, pontos):
        distancia_total = 0.0

        # Itera sobre a lista de pontos, calculando a distância Manhattan entre pontos consecutivos
        for i in range(1, len(pontos)):
            x1, y1 = pontos[i - 1]
            x2, y2 = pontos[i]
            distancia = abs(x2 - x1) + abs(y2 - y1)  # Distância Manhattan
            distancia_total += distancia

        return distancia_total


    def rota_task(self, aabb_info=None):

        if not self.observation_points:
            return 0, 0

        # Mapeia os pontos por coordenadas para facilitar acesso posterior
        coord_to_point = {tuple(pt["coord_abs"]): pt for pt in self.observation_points}
        coords = list(coord_to_point.keys())

        # Se temos info da AABB com corners reais, adicionamos os pontos deles também
        if aabb_info and all(k in aabb_info for k in ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]):
            corner_keys = ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]
            for key in corner_keys:
                corner_info = aabb_info[key]
                corner_coord = (corner_info["x"], corner_info["y"])
                if corner_coord not in coord_to_point:
                    coord_to_point[corner_coord] = {
                        "coord_abs": corner_coord,
                        "coord_gps": None,
                        "label": corner_info.get("label", f"corner_{corner_coord}")
                    }
                    coords.append(corner_coord)

        # Define limites da AABB
        min_x = aabb_info["x_min"] if aabb_info else min(x for x, _ in coords)
        max_x = aabb_info["x_max"] if aabb_info else max(x for x, _ in coords)
        min_y = aabb_info["y_min"] if aabb_info else min(y for _, y in coords)
        max_y = aabb_info["y_max"] if aabb_info else max(y for _, y in coords)

        # Agrupamento dos pontos por lado da AABB
        esquerda = sorted([pt for pt in coords if math.isclose(pt[0], min_x, abs_tol=1e-3)], key=lambda p: p[1])
        topo     = sorted([pt for pt in coords if math.isclose(pt[1], max_y, abs_tol=1e-3)], key=lambda p: p[0])
        direita  = sorted([pt for pt in coords if math.isclose(pt[0], max_x, abs_tol=1e-3)], key=lambda p: p[1], reverse=True)
        base     = sorted([pt for pt in coords if math.isclose(pt[1], min_y, abs_tol=1e-3)], key=lambda p: p[0], reverse=True)

        # Constrói a rota completa no sentido horário
        rota_coords = esquerda + topo + direita + base

        # Remove duplicatas mantendo a ordem
        rota_unica = []
        seen = set()
        for pt in rota_coords:
            if pt not in seen:
                seen.add(pt)
                rota_unica.append(pt)

        # Calcula distância total
        distancia = sum(math.hypot(x2 - x1, y2 - y1)
                        for (x1, y1), (x2, y2) in zip(rota_unica[:-1], rota_unica[1:]))

        qtde_pontos = len(rota_unica)
        tempo = distancia + qtde_pontos * 15

        # Monta a rota com pontos completos (label, coord_abs, etc)
        self.rota = [coord_to_point[pt] for pt in rota_unica]
        self.entry_point = self.rota[0]
        self.exit_point = self.rota[-1]
        self.possible_entry_points = [
            self.rota[0],
            self.rota[-1],
            coord_to_point.get(esquerda[-1]) if esquerda else self.rota[0],
            coord_to_point.get(direita[0]) if direita else self.rota[-1]
        ]

        self.inspection_distance = distancia
        self.inspection_time = tempo

        # print(self.rota)

        return distancia, tempo





        

    def calcular_tempo_inspecao(self, dist_percorrida, qtde_pontos):
        # Fórmula simples para calcular o tempo de inspeção baseado na distância e quantidade de pontos
        tempo_base_por_ponto = 0.5  # Tempo base por ponto (ajuste conforme necessário)
        tempo_total = dist_percorrida * 0.5 + qtde_pontos * tempo_base_por_ponto
        return tempo_total

    def ordenacao_alternada(self, index):
        # Alterna a ordem de acordo com o índice da coluna (zigue-zague)
        return index % 2 == 0
    



    def melhor_rota_para_task(self, ponto_anterior, ponto_seguinte):

        # POR ENQUANTO SÃO COMPARADAS TRÊS TIPOS DE ROTA PARA EXECUTAR A TASK, UMA NO SENTIDO HORÁRIO CONTORNANDO A MESMA, UMA NO ANTI-HORÁRIO E A DO VIZINHO MAIS PRÓXIMO

        rota = self.rota  # já ordenada no sentido horário
        # print(f"ponto anterior: {ponto_anterior}, ponto seuginte: {ponto_seguinte}, rota: {rota}")
        rota_inv = rota[::-1]
        resultados = []

        # Estratégia 1: volta completa sentido horário
        idx_entrada = encontrar_idx_mais_proximo(rota, ponto_anterior)
        rota1 = rota[idx_entrada:] + rota[:idx_entrada]
        custo1 = (
            distancia(ponto_anterior, rota1[0]['coord_abs']) +
            sum(distancia(rota1[i]['coord_abs'], rota1[i+1]['coord_abs']) for i in range(len(rota1)-1)) +
            distancia(rota1[-1]['coord_abs'], ponto_seguinte)
        )
        resultados.append(("volta_horario", custo1, rota1))

        # Estratégia 2: volta completa sentido anti-horário
        idx_entrada_inv = encontrar_idx_mais_proximo(rota_inv, ponto_anterior)
        rota2 = rota_inv[idx_entrada_inv:] + rota_inv[:idx_entrada_inv]
        custo2 = (
            distancia(ponto_anterior, rota2[0]['coord_abs']) +
            sum(distancia(rota2[i]['coord_abs'], rota2[i+1]['coord_abs']) for i in range(len(rota2)-1)) +
            distancia(rota2[-1]['coord_abs'], ponto_seguinte)
        )
        resultados.append(("volta_antihorario", custo2, rota2))

        # Estratégia 3: vizinho mais próximo
        visitados = []
        nao_visitados = rota.copy()
        atual = min(nao_visitados, key=lambda pt: distancia(ponto_anterior, pt['coord_abs']))
        visitados.append(atual)
        nao_visitados.remove(atual)

        while nao_visitados:
            proximo = min(nao_visitados, key=lambda pt: distancia(atual['coord_abs'], pt['coord_abs']))
            visitados.append(proximo)
            nao_visitados.remove(proximo)
            atual = proximo

        rota3 = visitados
        custo3 = (
            distancia(ponto_anterior, rota3[0]['coord_abs']) +
            sum(distancia(rota3[i]['coord_abs'], rota3[i+1]['coord_abs']) for i in range(len(rota3)-1)) +
            distancia(rota3[-1]['coord_abs'], ponto_seguinte)
        )
        resultados.append(("vizinho_mais_proximo", custo3, rota3))

        """ # Estratégia 3: meia-lua iniciando horário
        idx_ini = encontrar_idx_mais_proximo(rota, ponto_anterior)
        idx_fim = encontrar_idx_mais_proximo(rota, ponto_seguinte)
        if idx_ini <= idx_fim:
            ida = rota[idx_ini:idx_fim]
        else:
            ida = rota[idx_ini:] + rota[:idx_fim]
        volta = ida[::-1] + [rota[idx_fim]]
        rota3 = ida + volta[1:]  # evitar repetir o ponto intermediário
        custo3 = (
            distancia(ponto_anterior, rota3[0]['coord_abs']) +
            sum(distancia(rota3[i]['coord_abs'], rota3[i+1]['coord_abs']) for i in range(len(rota3)-1)) +
            distancia(rota3[-1]['coord_abs'], ponto_seguinte)
        )
        resultados.append(("meia_lua_horario", custo3, rota3))

        # Estratégia 4: meia-lua iniciando anti-horário
        idx_ini = encontrar_idx_mais_proximo(rota_inv, ponto_anterior)
        idx_fim = encontrar_idx_mais_proximo(rota_inv, ponto_seguinte)
        if idx_ini <= idx_fim:
            ida = rota_inv[idx_ini:idx_fim]
        else:
            ida = rota_inv[idx_ini:] + rota_inv[:idx_fim]
        volta = ida[::-1] + [rota_inv[idx_fim]]
        rota4 = ida + volta[1:]
        custo4 = (
            distancia(ponto_anterior, rota4[0]['coord_abs']) +
            sum(distancia(rota4[i]['coord_abs'], rota4[i+1]['coord_abs']) for i in range(len(rota4)-1)) +
            distancia(rota4[-1]['coord_abs'], ponto_seguinte)
        )
        resultados.append(("meia_lua_antihorario", custo4, rota4)) """

        melhor = min(resultados, key=lambda x: x[1])
        return melhor  # (nome_estrategia, custo_total, rota_escolhida)



    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "inspection_time": self.inspection_time,
            "inspection_distance": self.inspection_distance,
            "coordinates": self.coordinates
        }
    
    def __str__(self):
        info = [
            f"🟡 TASK {self.id}",
            f"📍 Posição média: ({self.x:.6f}, {self.y:.6f})",
            f"⏱️ Tempo de inspeção: {self.inspection_time:.2f}",
            f"📏 Distância de inspeção: {self.inspection_distance:.2f}",
            f"🔁 Entry Point: {self.entry_point}",
            f"⛔ Exit Point: {self.exit_point}",
            f"🧭 Coordenadas (px, py): {self.coordinates}",
            f"📦 Equipamentos associados: {', '.join(self.equipment_ids) if self.equipment_ids else 'Nenhum'}",
            f"🔎 Pontos de observação ({len(self.observation_points)}):"
        ]
        for i, pt in enumerate(self.observation_points):
            info.append(f"    {i+1}. {pt}")

        return "\n".join(info)
    
################################################################ FUNÇÕES CRIADAS PARA GERAR TASKS ##########################################################

def agrupar_missions_por_aabb(missions, equipamento_aabb):
    aabb_dict = defaultdict(list)
    for equip in missions:
        aabb = equipamento_aabb.get(equip)
        if aabb is not None:
            aabb_dict[aabb].append(equip)
        else:
            print(f"[AVISO] Equipamento '{equip}' não encontrado em equipamento_aabb.")
    return list(aabb_dict.values())  # Cada lista interna será uma Task

def gerar_tasks(missions, obs_points, equipamento_aabb, equipamento_data, aabb_info_data):
    grouped_missions = agrupar_missions_por_aabb(missions, equipamento_aabb)
    # print(f"🔹 Missões agrupadas por AABB: {grouped_missions}")
    
    tasks = []

    for task_index, equipment_ids in enumerate(grouped_missions):
        pontos_task = []
        pxs, pys = [], []

        aabb_idx = equipamento_aabb.get(equipment_ids[0])
        aabb_key = f"aabb_{aabb_idx}"
        aabb_info = aabb_info_data.get(aabb_key)


        for equipamento_id in equipment_ids:
            equipamento = equipamento_data.get(equipamento_id)
            if not equipamento:
                print(f"[AVISO] Equipamento '{equipamento_id}' não encontrado no JSON de dados.")
                continue

            pontos_foto = equipamento.get("pontos_foto", [])
            for ponto in pontos_foto:
                coord_abs = ponto.get("coord_abs")
                coord_gps = ponto.get("coord_gps")
                
                if coord_abs:  # Priorize a coord_abs para rota (x, y)
                    pontos_task.append({
                        "coord_abs": (coord_abs[0], coord_abs[1]),  # (x, y)
                        "coord_gps": tuple(coord_gps) if coord_gps else None,
                        "label": ponto.get("label")
                    })


            pxs.append(equipamento.get("px", 0.0))
            pys.append(equipamento.get("py", 0.0))

        if not pontos_task:
            print(f"[INFO] Nenhum ponto de observação encontrado para task_{task_index + 1}")
            continue

        # Coordenadas médias da task
        x_medio = sum(pxs) / len(pxs)
        y_medio = sum(pys) / len(pys)

        task = Task(
            id=f"task_{task_index + 1}",
            x=x_medio,
            y=y_medio,
            inspection_time=0,
            inspection_distance=0,
            entry_point=None,
            exit_point=None,
            observation_points=pontos_task,
            aabb_info=aabb_info
        )
        task.equipment_ids = equipment_ids
        task.coordinates = (x_medio, y_medio)

        tasks.append(task)

    print(f"✅ {len(tasks)} tasks criadas com sucesso.")
    return tasks