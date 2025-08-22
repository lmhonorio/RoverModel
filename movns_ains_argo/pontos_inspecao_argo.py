import numpy as np


### NÃO ESTÁ SENDO USADA ESSA FUNÇÃO
""" def create_matrix_trans():
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

        return trans """

class Equipamento:


    def __init__(self, nome, regiao, subregiao, coordenadas, dimensoes = None, GPS=None):
        self.nome = nome
        self.regiao = regiao
        self.subregiao = subregiao
        self.coordenadas = coordenadas
        self.dimensoes = dimensoes
        self.calcular_pontos_foto()
        # self.matrix_image2gps = create_matrix_trans()
        self.GPS = GPS

    def calcular_pontos_foto(self):
        y, x = self.coordenadas
        distancia_horizontal = (self.dimensoes[1] // 2) + 2
        distancia_vertical = (self.dimensoes[0] // 2) + 2
        self.pontos_foto = [
            (y - distancia_vertical, x - distancia_horizontal),
            (y + distancia_vertical, x - distancia_horizontal),
            (y - distancia_vertical, x + distancia_horizontal),
            (y + distancia_vertical, x + distancia_horizontal),
        ]

    def transform_image_to_gps(self):
        image_coords_homogeneous = np.hstack([self.coordenadas, 1])
        gps_coords = np.dot(image_coords_homogeneous, self.matrix_image2gps)
        return gps_coords

    def __repr__(self):
        return f"{self.__class__.__name__}(regiao={self.regiao}, coordenadas={self.coordenadas})"

class ParaRaio(Equipamento):
    pass


class Tcp(Equipamento):
    pass

        
class Estrutura(Equipamento):
    distancia_horizontal = 2  # Distância vertical em metros
    distancia_vertical = 1


class Reator(Equipamento):
    distancia_horizontal = 4  # Distância vertical em metros
    distancia_vertical = 3

class IsoladorPedestal(Equipamento):
    pass

class SeccionadoraHorizontal(Equipamento):
    distancia_horizontal = 3  # Distância vertical em metros
    distancia_vertical = 1

class linhas_verticais(Equipamento):
    pass

class Disjuntor(Equipamento):
    distancia_horizontal = 3  # Distância vertical em metros
    distancia_vertical = 1

class TransformadorCorrente(Equipamento):
    distancia_horizontal = 2  # Distância vertical em metros
    distancia_vertical = 2
