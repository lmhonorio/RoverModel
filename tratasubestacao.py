import ast
import numpy as np
import pandas as pd



# Função para ajustar as coordenadas para a nova origem e converter para metros
def ajustar_para_metros(lista, min_valor, fator_conversao=111320):  # Aproximação para metros por grau
    return [(valor - min_valor) * fator_conversao for valor in lista]

# Função para inverter X e Y nas listas de tuplas
def inverter_coordenadas(lista):
    if isinstance(lista, str):  # Se os dados forem string, converter para lista
        lista = ast.literal_eval(lista)
    return [(y, x) for x, y in lista]

def posicao_em_metros(valor, valor_zero, fator_conversao=111320):
    posicao = (valor-valor_zero)*fator_conversao
    return posicao

# Função para separar largura e altura
def separar_altura(lista, fator_conversao=111320):
    #dlargura = [x for x, y in lista]
    daltura = [y for x, y in lista]
    #largura = abs(dlargura[0]-dlargura[1])*fator_conversao
    altura = 2*abs(daltura[0]-daltura[1])*fator_conversao
    return  altura

def separar_largura(lista, fator_conversao=111320):
    dlargura = [x for x, y in lista]
    #daltura = [y for x, y in lista]
    largura = 2*abs(dlargura[0]-dlargura[1])*fator_conversao
    #altura = abs(daltura[0]-daltura[1])*fator_conversao
    return largura

def separar_coordenadas(lista, fator_conversao=111320):
    dlargura = [x for x, y in lista]
    daltura = [y for x, y in lista]
    largura = abs(dlargura[0]-dlargura[1])*fator_conversao
    altura = abs(daltura[0]-daltura[1])*fator_conversao
    return largura, altura


# Caminho do arquivo atualizado
file_path = "obstaculos.xlsx"

# Verificar as planilhas disponíveis no arquivo
xls = pd.ExcelFile(file_path)

# Carregar a planilha "Parnaiba3"
df = pd.read_excel(xls, sheet_name="Parnaiba3")

# Exibir as primeiras linhas para entender a estrutura
print(df.head())

# Aplicar a inversão nas colunas Vx e Vy
df["Vx"] = df["Vx"].apply(inverter_coordenadas)
df["Vy"] = df["Vy"].apply(inverter_coordenadas)

min_latitude = df["Latitude"].min()
min_longitude = df["Longitude"].min()
max_latitude = df["Latitude"].max()
max_longitude = df["Longitude"].max()

med_latitude = (max_latitude+min_latitude)/2.0
med_longitude = (max_longitude+min_longitude)/2.0

df["Px"] = df["Latitude"].apply(lambda x: posicao_em_metros(x, med_latitude))
df["Py"] = df["Longitude"].apply(lambda x: posicao_em_metros(x, med_longitude))

# Criar novas colunas
df["Vx_altura"] = df["Vx"].apply(separar_altura)
df["Vy_largura"]= df["Vy"].apply(separar_largura)

# Converter as coordenadas em arrays NumPy para facilitar os cálculos
#vx_largura_total = df["Vx_largura"].values
#vx_altura_total = df["Vx_altura"].values

# Determinar o ponto mais inferior e mais à esquerda
#min_x = vx_largura_total.min()
#min_y = vx_altura_total.min()

# Ajustar todas as coordenadas para a nova origem e converter para metros
# df["Vx_largura"] = df["Vx_largura"].apply(lambda lista: ajustar_para_metros(lista, min_x))
# df["Vx_altura"] = df["Vx_altura"].apply(lambda lista: ajustar_para_metros(lista, min_y))
# df["Vy_largura"] = df["Vy_largura"].apply(lambda lista: ajustar_para_metros(lista, min_x))
# df["Vy_altura"] = df["Vy_altura"].apply(lambda lista: ajustar_para_metros(lista, min_y))

# Salvar a nova planilha
output_path = "obstaculos_processado2.xlsx"
df.to_excel(output_path, sheet_name="Parnaiba3_Transformado", index=False)


