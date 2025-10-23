import pandas as pd
import math
import re
import json
import matplotlib.pyplot as plt

class AjustePlanilha:
    def __init__(self, sheet_name="Parnaiba3"):
        self.min_lat = None
        self.max_lat = None
        self.med_lat = None
        self.min_lon = None
        self.max_lon = None
        self.med_lon = None
        self.sheet_name = sheet_name

    def inverter_coordenadas(self, lista):
        if isinstance(lista, str):
            # lista = ast.literal_eval(lista)
            lista = json.loads(lista)
        return [(y, x) for x, y in lista]

    def posicao_em_metros_lat(self, lat, lat_ref):
        return (lat - lat_ref) * 111132.0

    def posicao_em_metros_lon(self, lon, lon_ref, lat_ref):
        return (lon - lon_ref) * (111320.0 * math.cos(math.radians(lat_ref)))

    def separar_altura(self, lista):
        daltura = [lat for lat, lon in lista]
        return 2 * abs(daltura[0] - daltura[1]) * 111132.0

    def separar_largura(self, lista):
        dlargura = [lon for lat, lon in lista]
        return 2 * abs(dlargura[0] - dlargura[1]) * 111320.0 * math.cos(math.radians(self.med_lat))

    def salvar_parametros_conversao(self, writer):
        df_parametros = pd.DataFrame({
            "Parametro": ["Latitude Min", "Latitude Max", "Latitude Média",
                          "Longitude Min", "Longitude Max", "Longitude Média"],
            "Valor": [self.min_lat, self.max_lat, self.med_lat,
                      self.min_lon, self.max_lon, self.med_lon]
        })
        df_parametros.to_excel(writer, sheet_name="ParametrosConversao", index=False)

    def processar(self, file_path, output_path):
        df = pd.read_excel(file_path, sheet_name=self.sheet_name)

        df["Vx"] = df["Vx"].apply(self.inverter_coordenadas)
        df["Vy"] = df["Vy"].apply(self.inverter_coordenadas)

        self.min_lat = df["Latitude"].min()
        self.max_lat = df["Latitude"].max()
        self.med_lat = (self.min_lat + self.max_lat) / 2.0
        self.min_lon = df["Longitude"].min()
        self.max_lon = df["Longitude"].max()
        self.med_lon = (self.min_lon + self.max_lon) / 2.0

        df["Py"] = df["Latitude"].apply(lambda x: self.posicao_em_metros_lat(x, self.med_lat))
        df["Px"] = df["Longitude"].apply(lambda x: self.posicao_em_metros_lon(x, self.med_lon, self.med_lat))

        def gerar_id_simplificado(full_name):
            if not isinstance(full_name, str) or "::" not in full_name:
                return full_name

            partes = full_name.split("::")
            if len(partes) == 1:
                return full_name

            n = len(partes)-2

            secao = partes[n]
            equipamento = partes[-1]

            # Simplifica a seção pegando só a primeira letra de cada parte separada por "_"
            sigla_secao = ''.join(p[0] for p in secao.lower().split('_') if p.isalnum())

            # Remove caracteres especiais do equipamento
            equipamento_limpo = re.sub(r'[^a-zA-Z0-9]', '', equipamento).lower()

            return f"{sigla_secao}_{equipamento_limpo}"

        df["ID"] = df["Model Name"].apply(gerar_id_simplificado)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=self.sheet_name + "_Transformado", index=False)
            self.salvar_parametros_conversao(writer)

        print(f"✅ Conversão concluída e salva em '{output_path}'.")

        return df

    def plotar_coordenadas_labels(self, df):
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df["Px"], df["Py"], c='blue', marker='o', label='Coordenadas')
        for i, row in df.iterrows():
            plt.text(row["Px"], row["Py"], row["ID"], fontsize=8, ha='right')
        plt.xlabel("Posição X (metros)")
        plt.ylabel("Posição Y (metros)")
        plt.title("Coordenadas dos Equipamentos")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def metros_para_geocoordenadas(lista_metros, file_path_parametros=None):
        """
        Converte lista de coordenadas em metros para coordenadas geográficas (lat, lon),
        usando valores padrão de conversão.

        Parâmetros:
            - lista_metros: lista de [x, y] ou [(x1, y1), (x2, y2), ...]
            - file_path_parametros: (ignorado) mantido para compatibilidade

        Retorna:
            Lista de (latitude, longitude)
        """
        # Valores padrão de conversão
        med_lat = -3.123139
        med_lon = -41.764440

        resultado = []
        for x, y in lista_metros:
            lon = x / (111320.0 * math.cos(math.radians(med_lat))) + med_lon
            lat = y / 111132.0 + med_lat
            resultado.append((lat, lon))
        return resultado
    
if __name__ == "__main__":

    conversor = AjustePlanilha()
    # conversor.processar("./planilhas/obstaculos.xlsx", "./planilhas/obstaculos_processado6.xlsx")
    df = conversor.processar("./planilhas/equipment.xlsx", "./planilhas/equipment_processado.xlsx")
    conversor.plotar_coordenadas_labels(df)
