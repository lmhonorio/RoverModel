"""
csv_kml_helper.py - Helper module for CSV and KML operations
Provides convenience functions for saving observation points to CSV and KML formats
"""

from segmentutils import SegmentUtils


def save_observation_points_to_kml_csv(obstacles, observation_points, threshold, 
                                       xlsx_path, output_folder, offset_lat_meters=0.0, 
                                       offset_lon_meters=0.0):
    """
    Salva pontos de observação em formato KML e CSV
    
    Args:
        obstacles: Lista de obstáculos
        observation_points: Pontos de observação
        threshold: Limiar de distância
        xlsx_path: Caminho do arquivo XLSX
        output_folder: Pasta de saída
        offset_lat_meters: Deslocamento em latitude (metros)
        offset_lon_meters: Deslocamento em longitude (metros)
    """
    # Chamar a função do SegmentUtils que já existe
    SegmentUtils.save_observation_points_to_kml(
        obstacles, 
        observation_points, 
        threshold, 
        xlsx_path, 
        output_folder, 
        offset_lat_meters=offset_lat_meters,
        offset_lon_meters=offset_lon_meters
    )


__all__ = ['save_observation_points_to_kml_csv']
