from flask import Flask, jsonify, request, send_file
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import MultipleLocator

class Mapa:
    def __init__(self, image_path, d_cel=1, factor=10):
        self.image_path = image_path
        self.d_cel = d_cel
        self.factor = factor
        self.coordenadas = []

        self.image_cv2 = cv2.imread(self.image_path)
        (self.h, self.w) = self.image_cv2.shape[:2]

        self.nL = int(self.h / self.factor)
        self.nC = int(self.w / self.factor)

        self.image_plt = plt.imread("/home/milena/ardupilot/FuncoesPython/Figuras/mapaNovo.png") # Carrega imagem para usar a biblioteca do matplotlib

    def desenhar_rota(self, rota, cor='g-', espessura_linha=2, save_path="/home/milena/interface_argo/teste.app/src/Figuras/plot.png"):
        # PLOTAGEM
        # Plotar o caminho
        y_trajetoria, x_trajetoria = zip(*rota)
        plt.plot(x_trajetoria, y_trajetoria, marker='o', color='green', linestyle='-')

        # Ajustar limites do gráfico
        plt.xlim([min(x_trajetoria) - 1, max(x_trajetoria) + 1])
        plt.ylim([min(y_trajetoria) - 1, max(y_trajetoria) + 1])

        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title('Rota Região')
        plt.imshow(self.image_plt, origin='upper', extent=[0, self.nC, 0, self.nL]) # Define a origem no canto superior esquerdo
        plt.xlim([0, self.nC])
        plt.ylim([0, self.nL])

        plt.grid(True)
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))  # Ajusta para aparecer a cada uma unidade
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

        if save_path:
            plt.savefig(save_path)  # Salva a imagem no caminho especificado

        plt.close()  # Fecha o plot para liberar memória

        return save_path  # Retorna o caminho da imagem salva
