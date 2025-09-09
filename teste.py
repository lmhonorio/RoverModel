import matplotlib.pyplot as plt
import numpy as np
import simplekml

# Rota fornecida
rota = [[-3.123211593420434, -41.765399197300724], [-3.123211593420434, -41.7655257434927], [-3.123247586653706, -41.7655257434927], [-3.123272281611054, -41.76550715497968], [-3.123272281611054, -41.76547542711749], [-3.1232835798869782, -41.76547542711749], [-3.1233082748443266, -41.76547542711749], [-3.1233082748443266, -41.76550715497968], [-3.1233082748443266, -41.765543140872666], [-3.1233195731202508, -41.76557912676565], [-3.1233624671561717, -41.76557912676565], [-3.123389462081126, -41.76557912676565], [-3.12341645700608, -41.76557912676565], [-3.123417996616633, -41.765579229325446], [-3.123417996616633, -41.76561511265864], [-3.123463546053339, -41.76561511265864], [-3.123463546053339, -41.76564796597964], [-3.123494639707732, -41.76564796597964], [-3.123494639707732, -41.76565109855162], [-3.123697026958929, -41.76565109855162], [-3.123697026958929, -41.76568708444461], [-3.123697026958929, -41.76572307033759], [-3.123697026958929, -41.76572425607277], [-3.12374741748551, -41.76572425607277], [-3.12374741748551, -41.76572307033759], [-3.1239498047367067, -41.76572307033759], [-3.1239498047367067, -41.76572425607277], [-3.124000195263288, -41.76572425607277]]

# Converte para array NumPy
coords = np.array(rota)

# Cria a figura
plt.figure(figsize=(8, 8))

# Plota a rota
plt.plot(coords[:, 1], coords[:, 0], linestyle='-', marker='o', color='blue', markersize=5)

# Destaca início e fim
plt.plot(coords[0, 1], coords[0, 0], marker='s', color='green', markersize=10, label='Início')
plt.plot(coords[-1, 1], coords[-1, 0], marker='X', color='red', markersize=10, label='Fim')

# Ajustes do gráfico
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Caminho da Rota")
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()

# Cria objeto KML
kml = simplekml.Kml()

# Converte rota para formato (longitude, latitude)
linha = [(lon, lat) for lat, lon in rota]

# Adiciona a rota como LineString
ls = kml.newlinestring(name="Minha Rota", coords=linha)
ls.style.linestyle.color = simplekml.Color.blue
ls.style.linestyle.width = 3

# Adiciona início e fim
kml.newpoint(name="Início", coords=[linha[0]]).style.iconstyle.color = simplekml.Color.green
kml.newpoint(name="Fim", coords=[linha[-1]]).style.iconstyle.color = simplekml.Color.red

# Salva arquivo
kml.save("rota.kml")
