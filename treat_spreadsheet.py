from geographiclib.geodesic import Geodesic
import math
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

class TreatSpreadsheet:
    """ Treat data about equipment dimensions and coordinates from a spreadsheet.

    Obs: we use as standard the longitude as x and the latitude as y.
    """

    def __init__(self):

        # Spreadsheet file path
        self.original_file_path = "planilhas/models.xlsx"
        self.transformed_file_path = "planilhas/equipment.xlsx"

        self.geolocation_from_gazebo = True

        self.df = self.LoadFileToDataframe()

        self.equipment_lat = self.df['Latitude']
        self.equipment_lon = self.df['Longitude']
        self.model_name = self.df['Model Name']

        # Equipment dimensions, HALF of width and height [meters]
        # Data from gazebo dae file
        dimension_REATOR = (1.534546, 3.054527) 
        dimension_PR = (0.6871033, 0.6181564) 
        dimension_TPC = (0.864502, 0.7777786) 
        dimension_IP = (0.7239075, 0.5235214) 
        dimension_SECH = (3.303741, 0.5302429) 
        dimension_TC = (0.8567124, 0.7042313) 
        dimension_SECV = (1.088993, 0.54982) 
        dimension_DISJUNTOR = (2.458675, 0.7382889) 
        dimension_BUSCSB = (0.7805481, 0.54982) 
        dimension_BUSIP = (0.7805519, 0.54982) 
        dimension_ESTRUTURA2 = (2.74, 1.3, -2.74, -1.3)
        dimension_ESTRUTURA3 = (2.74, 1.3, -2.74, -1.3)
        dimension_canaletas_horizontal = (11, 2) # Não é usado SE GAZEBO, só tá por causa da estrutura do for principal

        if self.geolocation_from_gazebo:

            # width, height and x, y from gazebo global frame
            self.dimension_canaletas_taludes = {
                "canaleta1": (11, 2, -0.459704, -123.217343),
                "canaleta2": (2, 23.6, -13.222777, -118.792624),
                "canaleta3": (2, 16.62, 8.387840, -118.721732),
                "canaleta4": (101, 2, 17, -68.997061),
                "canaleta5": (2, 83, 58.553385, -98.998900),
                "canaleta6": (101, 2, -87.351506, -19.399625),
                "canaleta7": (2, 123, -24.635673, -21),
                "canaleta8": (202, 2, 38.059014, 27),
                "canaleta9": (103, 2, -28, -21),
                "talude1": (2, 50, -95, -74.099800),
                "talude2": (180, 2, -72, 15), 
                "talude3": (250, 2, 22.256600, 10),
                "talude4": (2, 100, -35, 120),
                "talude5": (2, 75, 59.464300, -67.093500)
                }

            # Base for the transformation, cartesian to geodesic
            lat0, lon0 = -3.123199, -41.764537 

            # Convert the canaletas and taludes coordinates from cartesian to geodesic
            for name, canaleta in self.dimension_canaletas_taludes.items():
                novo_lat, novo_lon = self.CartesianToGeodesic(-canaleta[2], canaleta[3], lat0, lon0) # Tem algum problema no sinal do x no gazebo, não mudar isso
                self.dimension_canaletas_taludes[name] = (canaleta[0], canaleta[1], novo_lat, novo_lon)

        # Other objects        
        if self.geolocation_from_gazebo:

            self.dimensions = [dimension_REATOR, dimension_PR, dimension_TPC, dimension_IP, dimension_SECH, dimension_TC, dimension_SECV, dimension_DISJUNTOR, dimension_BUSCSB, dimension_BUSIP, dimension_ESTRUTURA2, dimension_ESTRUTURA3, dimension_canaletas_horizontal] 
            self.equipment_name = ["REATOR", "PR", "TPC", "IP", "SECH", "TC", "SECV", "DISJUNTOR", "BUSCSB", "BUSIP", "ESTRUTURA2", "ESTRUTURA3", "canaletas"]
        
        else:

            self.equipment_name = ["reator", "pr", "tpc", "ip", "sech", "tc", "secv", "disjuntor", "buscsb", "busip", "estrutura2", "estrutura3", "canaleta", "talude", "obstaculo", "bombeiro", "torre", "transformador", "caixa"]

    def LoadFileToDataframe(self):
        """
        Load file to pandas dataframe.

        Returns:
            pd.DataFrame: Dataframe
        """
        try:
            df = pd.read_excel(self.original_file_path)
            return df
        except Exception as e:
            print(f"Could not load the file: {e}")
            return None
        
    def CalculateVectors(self, lat, lon, ponto_medio):
        """
        Calculate the vectors x and y from the midpoint.

        Args:
            lat (float): Latitude of the point
            lon (float): Longitude of the point
            ponto_medio (tuple): Midpoint coordinates (lat, lon)
        
        Returns:
            tuple: Vectors x and y
        """
        vetores_x = []
        vetores_y = []
        
        vetores_x =[(ponto_medio[1], ponto_medio[0]), (lon, ponto_medio[0])]
        vetores_y = [(ponto_medio[1], ponto_medio[0]), (ponto_medio[1],lat)]
            
        return vetores_x, vetores_y
    
    def PlotEquipamentCoordinates(self, vetores_x, vetores_y):
        """
        Plot equipament coordinates and collect points to the mission.

        Args:

            vetores_x (list): List of x coordinates
            vetores_y (list): List of y coordinates
            equipment_lat (list): List of equipment latitudes
            equipment_lon (list): List of equipment longitudes

        Returns:
            None
        """

        fig, ax = plt.subplots()
        ax.scatter(self.equipment_lon, self.equipment_lat, picker=True, label='Equipment')
        ax.scatter(vetores_x[1][0], vetores_x[1][1], picker=True, color='red', label='Robot Position')
        ax.scatter(vetores_y[1][0], vetores_y[1][1], picker=True, color='red', label='Robot Position')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.show()

    def PlotEquipamentDimensions(self, dx, dy, lat_lon_central):
        """
        Plot equipment coordinates as rectangles with given dimensions (dx and dy in meters),
        centered at (equipment_lon, equipment_lat). Convert meters to degrees of latitude and longitude,
        1 degree of latitude = ~111.32 km and 1 degree of longitude depends on the latitude

        Args:
            lat_lon_central (list): List of tuples (lon, lat)
            dx (float): Width of the rectangle in meters
            dy (float): Height of the rectangle in meters
        
        Returns:
            None
        """
        fig, ax = plt.subplots()

        # Plot rectangles
        for latlon, w, h in zip(lat_lon_central, dx, dy):

            if all(isinstance(i, float) for i in latlon) and len(latlon) != 0:
                lon = latlon[0]
                lat = latlon[1]

                w_deg = w / (40075000 * np.cos(np.radians(lat)) / 360)  # longitude
                h_deg = h / 111320  # latitude

                ax.scatter(lon, lat, color='blue')
                rect = patches.Rectangle((lon - w_deg/2, lat - h_deg/2), w_deg, h_deg,
                                        linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            # Estruturas case
            elif all(isinstance(i, list) for i in latlon) and len(latlon) != 0:
                for latlons in latlon:
                    lon = latlons[0]
                    lat = latlons[1]

                    w_deg = w / (40075000 * np.cos(np.radians(lat)) / 360)
                    h_deg = h / 111320

                    ax.scatter(lon, lat, color='blue')
                    rect = patches.Rectangle((lon - w_deg/2, lat - h_deg/2), w_deg, h_deg,
                                            linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

            else:
                continue

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Equipments in Geo Coordinates")
        ax.legend(["Equipment Center"])
        ax.grid(True)
        plt.axis('equal')
        plt.show()

    def SafeJsonLoad(self, val):
        """
        Safely load a JSON string into a Python object.
        
        Args:
            val (str): JSON string
        
        Returns:
            dict or None: Loaded JSON object or None if the string is not valid JSON
        """
        if isinstance(val, str):
            return json.loads(val)
        if pd.isna(val):
            return []
        return val 

    def CartesianToGeodesic(self, x, y, lat0, lon0):
        """
        Convert Cartesian coordinates (x, y) to geodesic coordinates (latitude, longitude)
        using the WGS84 ellipsoid model.
        
        Args:
            x (float): X coordinate in meters
            y (float): Y coordinate in meters
            lat0 (float): Initial latitude in degrees
            lon0 (float): Initial longitude in degrees

        Returns:
            tuple: Latitude and longitude in degrees
        """

        # Create the Geodesic object with the WGS84 model
        geod = Geodesic.WGS84

        # Calculate the azimuth and distance from the Cartesian displacements
        azimuth = math.atan2(y, x) * 180 / math.pi  # Convert to degrees
        distance = math.sqrt(x**2 + y**2)  # Distance in meters

        # Use the Direct method to calculate the new latitude and longitude
        result = geod.Direct(lat0, lon0, azimuth, distance)

        # Latitude, longitude
        lat = result['lat2']
        lon = result['lon2']

        return lat, lon
    
    def NewLine(self, name, lat, lon, alt, vx, vy, dx, dy):
        """
        Create a new line with the given parameters.
        Args:
            name (str): Name of the line
            lat (float): Latitude of the line
            lon (float): Longitude of the line
            alt (float): Altitude of the line
            vx (list): Vx coordinates
            vy (list): Vy coordinates
            dx (float): Width of the line in meters
            dy (float): Height of the line in meters
        Returns:
            dict: New line with the given parameters
        """

        nova_linha = {
            'Model Name': name,
            'Latitude': lat,
            'Longitude': lon,
            'Altitude': alt,
            'LatLonCentral': json.dumps([lon, lat]),
            'Vx': json.dumps(vx),
            'Vy': json.dumps(vy),
            'Vx_largura': dx,
            'Vy_altura': dy
        }
        return nova_linha

    def ProcessCanaletaTalude(self, i):
        """
        Process the canaleta and talude dimensions and coordinates.
        Args:
            i (int): Index of the equipment
        """
        
        for name, data in self.dimension_canaletas_taludes.items():
            ponto_medio = (data[2], data[3]) # lat, lon
            dx, dy = data[0], data[1] # central cartesian coordinates
            lat, lon = self.CartesianToGeodesic(dx, dy, data[0], data[1]) # transformation
            vetores_x, vetores_y = self.CalculateVectors(lat, lon, ponto_medio)

            # Add new line
            nova_linha = self.NewLine(f'ARGO_PARNAIBAIII_V2_LD::BASE::{name}', ponto_medio[0], ponto_medio[1], 77, vetores_x, vetores_y, dx, dy)
            self.df = pd.concat([self.df, pd.DataFrame([nova_linha])], ignore_index=True)

    def ProcessOtherEquipment(self, i, dimension):
        """
        Process the other equipment dimensions and coordinates.
        Args:
            i (int): Index of the equipment
            dimension (tuple): Equipment dimensions (width, height)
        """

        ponto_medio = (self.equipment_lat[i], self.equipment_lon[i])
        dx, dy = dimension
        lat, lon = self.CartesianToGeodesic(dx, dy, ponto_medio[0], ponto_medio[1])
        vetores_x, vetores_y = self.CalculateVectors(lat, lon, ponto_medio)

        self.df.loc[i, 'LatLonCentral'] = json.dumps([ponto_medio[1], ponto_medio[0]])
        self.df.loc[i, 'Vx'] = json.dumps(vetores_x)
        self.df.loc[i, 'Vy'] = json.dumps(vetores_y)
        self.df.loc[i, 'Vx_largura'] = dx * 2
        self.df.loc[i, 'Vy_altura'] = dy * 2

        # Test vectors position
        # self.PlotEquipamentCoordinates(vetores_x, vetores_y)

    def ProcessEstrutura(self, i, dimension):
        """
        Process the structure dimensions and coordinates.
        Args:
            i (int): Index of the equipment
            vertice (tuple): Equipment dimensions (width, height)
        """

        if "ESTRUTURA2" in self.model_name[i]:
            coord = [14.2, 0, -14.2, 0]
        else:
            coord = [28.2, 0, -28.2, 0]
            ponto_medio = (self.equipment_lat[i], self.equipment_lon[i])

            #------------ Parte central da estrutura --------------
            # Parâmetros iniciais
            lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
            lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)

            # Deslocamentos em metros (valores de exemplo)
            dx = dimension[0]  # deslocamento no eixo X (longitude)
            dy = dimension[1]   # deslocamento no eixo Y (latitude)

            lat, lon = self.CartesianToGeodesic(dx, dy, lat0, lon0)

            # Calcular os vetores
            vx, vy = self.CalculateVectors(lat, lon, ponto_medio)

            # Add new line
            nova_linha = self.NewLine(f'{self.model_name[i]}middle', ponto_medio[0], ponto_medio[1], self.df['Altitude'][i], vx, vy, dimension[0], dimension[1])
            self.df = pd.concat([self.df, pd.DataFrame([nova_linha])], ignore_index=True)

        #--------------------------------------------------------------------
        # Ponto médio para a estrutura como um todo
        ponto_medio_ambos = (self.equipment_lat[i], self.equipment_lon[i])

        # Parâmetros iniciais
        lat0 = ponto_medio_ambos[0] # Latitude inicial (ponto médio)
        lon0 = ponto_medio_ambos[1]  # Longitude inicial (ponto médio)

        #---------------------- Topo da estrutura ---------------------------
        # Deslocamentos em metros (valores de exemplo)
        x = coord[0]  # deslocamento no eixo X (longitude)
        y = coord[1]   # deslocamento no eixo Y (latitude)
        
        # Novo ponto médio para cima
        lat1, lon1 = self.CartesianToGeodesic(x, y, lat0, lon0)

        # Ponto médio em cima
        ponto_medio1 = (lat1, lon1)

        # Deslocamentos em metros 
        dx = dimension[0]  # deslocamento no eixo X (longitude)
        dy = dimension[1]   # deslocamento no eixo Y (latitude)
        
        lat1, lon1 = self.CartesianToGeodesic(dx, dy, lat1, lon1)
        
        # Calcular os vetores
        vx1, vy1 = self.CalculateVectors(lat1, lon1, ponto_medio1)

        # Add new line
        nova_linha = self.NewLine(f'{self.model_name[i]}top', ponto_medio1[0], ponto_medio1[1], self.df['Altitude'][i], vx1, vy1, dimension[0], dimension[1])
        self.df = pd.concat([self.df, pd.DataFrame([nova_linha])], ignore_index=True)

        #------------------- Parte inferior da estrutura --------------------
        x = coord[2]   # deslocamento no eixo X (longitude)
        y = coord[3] # deslocamento no eixo Y (latitude)
        
        # Novo ponto médio para baixo
        lat2, lon2 = self.CartesianToGeodesic(x, y, lat0, lon0)

        # Ponto médio em baixo
        ponto_medio2 = (lat2, lon2)
        
        # Deslocamentos em metros 
        dx = dimension[2]  # deslocamento no eixo X (longitude)
        dy = dimension[3]   # deslocamento no eixo Y (latitude)
        
        lat2, lon2 = self.CartesianToGeodesic(dx, dy, lat2, lon2)

        # Calcular os vetores
        vx2, vy2 = self.CalculateVectors(lat2, lon2, ponto_medio2)

        nova_linha = self.NewLine(f'{self.model_name[i]}bottom', ponto_medio2[0], ponto_medio2[1], self.df['Altitude'][i], vx2, vy2, dimension[0], dimension[1])
        self.df = pd.concat([self.df, pd.DataFrame([nova_linha])], ignore_index=True)
    
    def main(self):
        """
        Main function to process the data and plot the equipment dimensions.
        """

        # Process each equipment based on its model name and dimensions
        for k in range(len(self.dimensions)):
            dimension = self.dimensions[k]

            for i in range(self.equipment_lat.size):
                
                if self.equipment_name[k] in self.model_name[i]:

                    if self.geolocation_from_gazebo and ("ESTRUTURA2" in self.model_name[i] or "ESTRUTURA3" in self.model_name[i]):
                        self.ProcessEstrutura(i, dimension)
                        self.df.to_excel(self.transformed_file_path, index=False)

                    elif "canaletas" in self.model_name[i]:
                        self.ProcessCanaletaTalude(i)
                        self.df.to_excel(self.transformed_file_path, index=False)
                    else:
                        self.ProcessOtherEquipment(i, dimension)
                        self.df.to_excel(self.transformed_file_path, index=False)

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

        # Plot the equipment dimensions
        self.PlotEquipamentDimensions(dx, dy, lat_lon_central)

if __name__ == "__main__":  

    treat_data = TreatSpreadsheet()
    treat_data.main()