from geographiclib.geodesic import Geodesic
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Dict, Union
import math
from math import radians, sin, cos, atan2, sqrt
import re

class TreatSpreadsheet:
    """ Treat data about equipment dimensions and coordinates from a spreadsheet.
    Obs: we use as standard the longitude as x and the latitude as y.
    """

    # Constantes
    ORIGINAL_FILE_PATH = "planilhas/coordenadas_visita.xlsx"
    TRANSFORMED_FILE_PATH = "planilhas/equipment.xlsx"
    LAT0, LON0 = -3.123199, -41.764537  # Base for conversion gazebo→geodésico
    GEOLOCATION_FROM_GAZEBO = True

    # Standard dimensions for each equipment type in meters
    # The dimensions are defined as (width, height)
    DIMENSIONS = {
        "reator": (2*1.534546, 2*3.054527),
        "pr": (2*0.6871033, 2*0.6181564),
        "tpc": (2*0.864502, 2*0.7777786),
        "ip": (2*0.7239075, 2*0.5235214),
        "sech": (2*3.303741, 2*0.5302429),
        "tc": (2*0.8567124, 2*0.7042313),
        "secv": (2*1.088993, 2*0.54982),
        "disjuntor": (2*2.458675, 2*0.7382889),
        "buscsb": (2*0.7805481, 2*0.54982),
        "busip": (2*0.7805519, 2*0.54982),
        "bombeiro": (1, 1),
        "caixa": (1,1),
        "estrutura": (5.48, 2.6),
        "torre": (5.48, 2.6),
        "transformador": (2*1.534546, 2*3.054527),
        "obstaculo": (1,1),
        "cercado": (11.41, 6.8),
        "diversos": (69, 48),
        "svc": (44.9, 82.5)
    }

    def __init__(self):

        self.df = self.LoadFileToDataframe()
        self.equipment_lat = self.df['Latitude']
        self.equipment_lon = self.df['Longitude']
        self.model_name = self.df['Model Name']
        self.convertCanaletasTaludes()

    def LoadFileToDataframe(self) -> pd.DataFrame:
        """
        Load file to pandas dataframe.
        """
        try:
            return pd.read_excel(self.ORIGINAL_FILE_PATH)
        except Exception as e:
            print(f"Erro ao carregar arquivo: {e}")
            return pd.DataFrame()

    def convertCanaletasTaludes(self):
        
        eq_name = ["canaleta", "talude"]
        self.points = {}

        for i in range(len(self.equipment_lat)):

            for eq in eq_name:

                if eq in self.model_name[i]:

                    match = re.match(r"^(.*?\d+)(.*)$", self.model_name[i])
                    model_base = match.group(1)
                    model_part = match.group(2)

                    if model_base not in self.points:
                        self.points[model_base] = {}
                    if model_part not in self.points[model_base]:
                        self.points[model_base][model_part] = {}

                    self.points[model_base][model_part] = [self.equipment_lat[i], self.equipment_lon[i]]
                    self.df = self.df[self.df['Model Name'] != self.model_name[i]]

        for model_base in self.points:

            if "left" in self.points[model_base] and "right" in self.points[model_base]:

                width = self.haversine(self.points[model_base]["left"][0], self.points[model_base]["left"][1], self.points[model_base]["right"][0], self.points[model_base]["right"][1])
                mid_width_lat = (self.points[model_base]["left"][0] + self.points[model_base]["right"][0]) / 2
                mid_width_lon = (self.points[model_base]["left"][1] + self.points[model_base]["right"][1]) / 2
                new_line1 = self.newLine(
                    name=model_base + "deitado",
                    lat=mid_width_lat,
                    lon=mid_width_lon,
                    alt=77
                )
                self.df = pd.concat([self.df, pd.DataFrame([new_line1])], ignore_index=True)
                self.DIMENSIONS[model_base + "deitado"] = (width, 1)

            if "top" in self.points[model_base] and "bottom" in self.points[model_base]:

                length = self.haversine(self.points[model_base]["top"][0], self.points[model_base]["top"][1], self.points[model_base]["bottom"][0], self.points[model_base]["bottom"][1])
                mid_length_lat = (self.points[model_base]["top"][0] + self.points[model_base]["bottom"][0]) / 2
                mid_length_lon = (self.points[model_base]["top"][1] + self.points[model_base]["bottom"][1]) / 2
                new_line2 = self.newLine(
                    name=model_base + "empe",
                    lat=mid_length_lat,
                    lon=mid_length_lon,
                    alt=77
                )
                self.df = pd.concat([self.df, pd.DataFrame([new_line2])], ignore_index=True)
                self.DIMENSIONS[model_base + "empe"] = (1, length)

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points on the Earth's surface.
        """
        R = 6371000.0  # raio da Terra em metros
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    @staticmethod
    def SafeJsonLoad(val: Union[str, float, list]) -> Union[list, dict]:
        """
        Safely load a JSON string into a Python object.
        """
        if isinstance(val, str):
            return json.loads(val)
        if pd.isna(val):
            return []
        return val

    @staticmethod
    def CartesianToGeodesic(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
        """
        Convert Cartesian coordinates (x, y) to geodesic coordinates (latitude, longitude)
        using the WGS84 ellipsoid model.
        """
        geod = Geodesic.WGS84
        azimuth = math.degrees(math.atan2(y, x))
        distance = math.hypot(x, y)
        result = geod.Direct(lat0, lon0, azimuth, distance)
        return result['lat2'], result['lon2']

    @staticmethod
    def CalculateVectors(lat: float, lon: float, midpoint: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Calculate the vectors x and y from the midpoint.
        """
        vx = [(midpoint[1], midpoint[0]), (lon, midpoint[0])]
        vy = [(midpoint[1], midpoint[0]), (midpoint[1], lat)]
        return vx, vy
    
    def newLine(self, name: str, lat: float, lon: float, alt: float) -> dict:
        """Cria nova linha para o DataFrame."""
        return {
            'Model Name': name,
            'Latitude': lat,
            'Longitude': lon,
            'Altitude': alt,
        }

    def PlotEquipamentCoordinates(self, dx: pd.Series, dy: pd.Series, centers: pd.Series) -> None:
        """
        Plot equipament coordinates and collect points to the mission.
        """
        fig, ax = plt.subplots()
        for latlon, w, h in zip(centers, dx, dy):
            if isinstance(latlon, list) and all(isinstance(v, float) for v in latlon) and len(latlon) == 2:
                lon, lat = latlon
                w_deg = w / (40075000 * np.cos(np.radians(lat)) / 360)
                h_deg = h / 111320
                ax.scatter(lon, lat, color='blue')
                rect = patches.Rectangle((lon - w_deg/2, lat - h_deg/2), w_deg, h_deg,
                                         linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Equipamentos em Coordenadas Geográficas")
        ax.legend(["Centro do Equipamento"])
        ax.grid(True)
        plt.axis('equal')
        plt.show()

    def main(self) -> None:
        """
        Main function to process the data and plot the equipment dimensions.
        """

        self.equipment_lat = self.df['Latitude']
        self.equipment_lon = self.df['Longitude']
        self.model_name = self.df['Model Name']

        for eq_name, dimension in self.DIMENSIONS.items():

            for i in range(len(self.equipment_lat)):

                if eq_name in self.model_name[i]:

                    midpoint = (self.equipment_lat[i], self.equipment_lon[i])
                    dx, dy = dimension
                    lat, lon = self.CartesianToGeodesic(dx, dy, *midpoint)
                    vx, vy = self.CalculateVectors(lat, lon, midpoint)

                    self.df.loc[i, 'LatLonCentral'] = json.dumps([midpoint[1], midpoint[0]])
                    self.df.loc[i, 'Vx'] = json.dumps(vx)
                    self.df.loc[i, 'Vy'] = json.dumps(vy)
                    self.df.loc[i, 'Vx_largura'] = dx
                    self.df.loc[i, 'Vy_altura'] = dy

        # Save the transformed DataFrame to an Excel file
        self.df['Vx_largura'] = self.df['Vx_largura'].apply(self.SafeJsonLoad)
        self.df['Vy_altura'] = self.df['Vy_altura'].apply(self.SafeJsonLoad)

        # Width and height of the rectangle [meters]
        dx = self.df['Vx_largura'] 
        dy = self.df['Vy_altura'] 

        # Save the transformed DataFrame to an Excel file
        self.df['Vx'] = self.df['Vx'].apply(self.SafeJsonLoad)
        self.df['Vy'] = self.df['Vy'].apply(self.SafeJsonLoad)
        lat_lon_central = self.df['LatLonCentral'].apply(self.SafeJsonLoad)

        self.df.to_excel(self.TRANSFORMED_FILE_PATH, index=False)
        self.PlotEquipamentCoordinates(dx, dy, lat_lon_central)


if __name__ == "__main__":
    TreatSpreadsheet().main()
