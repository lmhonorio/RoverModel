#!/usr/bin/env python3
"""
Método alternativo para salvar pontos de observação em KML usando CSV
Substitui o método original que dependia de arquivos Excel
"""

import os
import math
import pandas as pd
from xml.dom.minidom import Document

def metros_para_geocoordenadas_csv(lista_metros, csv_path):
    """
    Converte lista de coordenadas em metros para coordenadas geográficas (lat, lon),
    usando as coordenadas de referência do arquivo CSV.
    
    Parâmetros:
        - lista_metros: lista de [x, y] ou [(x1, y1), (x2, y2), ...]
        - csv_path: caminho do arquivo CSV com coordenadas GPS
    
    Retorna:
        Lista de (latitude, longitude)
    """
    try:
        # Lê o CSV para obter as coordenadas de referência
        df = pd.read_csv(csv_path)
        
        # Usa as coordenadas de referência do Gazebo2CSV.py
        lat_ref = -3.123199
        lon_ref = -41.764537
        radius_of_earth = 6378100.0  # metros
        
        resultado = []
        for x, y in lista_metros:
            # Aplica a mesma transformação de rotação do Gazebo2CSV.py
            x_rotated = y  # X original vira Y após rotação de -90°
            y_rotated = -x  # Y original vira -X após rotação de -90°
            
            # Converte metros para graus usando constantes do ArduPilot
            lat = y_rotated / (radius_of_earth * math.pi / 180.0) + lat_ref
            lon = x_rotated / (radius_of_earth * math.cos(math.radians(lat_ref)) * math.pi / 180.0) + lon_ref
            
            resultado.append((lat, lon))
        return resultado
        
    except Exception as e:
        raise ValueError(f"Erro ao ler parâmetros de conversão do CSV: {e}")

def save_observation_points_to_kml_csv(obstacles, perimeter_points, threshold, csv_path, output_folder, offset_lat_meters=0.0, offset_lon_meters=0.0):
    """
    Salva pontos de observação em arquivos KML usando CSV como fonte de coordenadas
    """
    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Converter pontos (x, y) para (lat, lon)
    pontos_xy = [(px, py) for px, py, _ in perimeter_points]
    pontos_xy_offset = [(px+offset_lon_meters, py+offset_lat_meters) for px, py, _ in perimeter_points]
    gps_coords = metros_para_geocoordenadas_csv(pontos_xy_offset, csv_path)
    coord_map = dict(zip(pontos_xy, gps_coords))

    for obs in obstacles:
        ox, oy = obs["pos"]
        label = obs["label"]
        pontos_obs = []

        for px, py, _ in perimeter_points:
            if distance((ox, oy), (px, py)) <= threshold:
                lat, lon = coord_map[(px, py)]
                pontos_obs.append((lat, lon))

        # Criar documento KML
        doc = Document()
        kml = doc.createElement("kml")
        kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
        doc.appendChild(kml)

        document = doc.createElement("Document")
        kml.appendChild(document)

        for lat, lon in pontos_obs:
            placemark = doc.createElement("Placemark")

            point = doc.createElement("Point")
            coordinates = doc.createElement("coordinates")
            coordinates.appendChild(doc.createTextNode(f"{lon},{lat},0"))

            point.appendChild(coordinates)
            placemark.appendChild(point)
            document.appendChild(placemark)

        filename = os.path.join(output_folder, f"{label}.kml")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(doc.toprettyxml(indent="  "))

        print(f"✅ KML salvo com deslocamento: {filename}")

if __name__ == "__main__":
    print("🔧 Módulo de conversão CSV para KML carregado!")
