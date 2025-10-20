#!/usr/bin/env python3
"""
Programa para validar o cálculo GPS do Gazebo2CSV.py
comparando com as posições GPS simuladas pelo ArduPilot no Gazebo
"""

import rospy
import math
import time
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Pose
import pandas as pd
from typing import Dict, List, Tuple

class GPSValidator:
    def __init__(self):
        """Inicializa o validador GPS"""
        rospy.init_node('gps_validator', anonymous=True)
        
        # Coordenadas de referência (mesmas do Gazebo2CSV.py)
        self.lat_ref = -3.123199
        self.lon_ref = -41.764537
        
        # Dados coletados
        self.gps_data = {}  # {model_name: {'gps': (lat, lon), 'pose': (x, y, z)}}
        self.model_poses = {}
        
        # Subscribers
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.gps_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.gps_callback)
        
        print("🔍 Validador GPS inicializado!")
        print(f"📍 Referência: Lat={self.lat_ref:.6f}, Lon={self.lon_ref:.6f}")
        print("⏳ Coletando dados por 10 segundos...")
        
    def model_states_callback(self, msg):
        """Callback para posições dos modelos no Gazebo"""
        for i, name in enumerate(msg.name):
            if i < len(msg.pose):
                pose = msg.pose[i]
                self.model_poses[name] = {
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z
                }
    
    def gps_callback(self, msg):
        """Callback para dados GPS do ArduPilot"""
        if msg.status.status >= 0:  # GPS válido
            # Armazena o GPS mais recente
            self.current_gps = {
                'lat': msg.latitude,
                'lon': msg.longitude,
                'alt': msg.altitude,
                'timestamp': time.time()
            }
    
    def gazebo_to_gps_calculation(self, x: float, y: float) -> Tuple[float, float]:
        """Cálculo GPS usando a fórmula corrigida do Gazebo2CSV.py"""
        radius_of_earth = 6378100.0  # metros
        
        # CORREÇÃO: Considera a rotação do modelo ARGO_PARNAIBAIII_V3 (-90° no eixo Z)
        # A rotação de -1.570796 radianos (90° negativo) significa:
        # X_gazebo -> Y_real (coordenada Norte-Sul)
        # Y_gazebo -> -X_real (coordenada Leste-Oeste, invertida)
        
        # Aplica a transformação de rotação
        x_rotated = y  # X original vira Y após rotação de -90°
        y_rotated = -x  # Y original vira -X após rotação de -90°
        
        # Converte metros para graus usando constantes do ArduPilot
        # Latitude: variação Norte-Sul (Y após rotação)
        lat = y_rotated / (radius_of_earth * math.pi / 180.0) + self.lat_ref
        # Longitude: variação Leste-Oeste (X após rotação)
        lon = x_rotated / (radius_of_earth * math.cos(math.radians(self.lat_ref)) * math.pi / 180.0) + self.lon_ref
        
        return lat, lon
    
    def calculate_distance_gps(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distância entre duas coordenadas GPS usando fórmula de Haversine"""
        R = 6371000  # Raio da Terra em metros
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def collect_data(self, duration: int = 10):
        """Coleta dados por um período especificado"""
        start_time = time.time()
        
        while time.time() - start_time < duration and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
            # Se temos GPS e poses de modelos, armazena
            if hasattr(self, 'current_gps') and self.model_poses:
                for model_name, pose in self.model_poses.items():
                    if model_name not in self.gps_data:
                        self.gps_data[model_name] = {
                            'gps_ardupilot': (self.current_gps['lat'], self.current_gps['lon']),
                            'pose_gazebo': (pose['x'], pose['y'], pose['z']),
                            'gps_calculado': self.gazebo_to_gps_calculation(pose['x'], pose['y'])
                        }
    
    def analyze_results(self):
        """Analisa e exibe os resultados da validação"""
        if not self.gps_data:
            print("❌ Nenhum dado coletado!")
            return
        
        print("\n" + "="*60)
        print("📊 ANÁLISE DE VALIDAÇÃO GPS")
        print("="*60)
        
        total_models = len(self.gps_data)
        total_error = 0
        max_error = 0
        min_error = float('inf')
        
        print(f"🔢 Total de modelos analisados: {total_models}")
        print()
        
        for model_name, data in self.gps_data.items():
            gps_ardupilot = data['gps_ardupilot']
            gps_calculado = data['gps_calculado']
            pose = data['pose_gazebo']
            
            # Calcula erro em metros
            error_meters = self.calculate_distance_gps(
                gps_ardupilot[0], gps_ardupilot[1],
                gps_calculado[0], gps_calculado[1]
            )
            
            total_error += error_meters
            max_error = max(max_error, error_meters)
            min_error = min(min_error, error_meters)
            
            print(f"🤖 Modelo: {model_name}")
            print(f"   📍 Posição Gazebo: X={pose[0]:.3f}m, Y={pose[1]:.3f}m, Z={pose[2]:.3f}m")
            print(f"   🛰️  GPS ArduPilot: Lat={gps_ardupilot[0]:.8f}, Lon={gps_ardupilot[1]:.8f}")
            print(f"   🧮 GPS Calculado:  Lat={gps_calculado[0]:.8f}, Lon={gps_calculado[1]:.8f}")
            print(f"   📏 Erro: {error_meters:.3f}m")
            print()
        
        # Estatísticas gerais
        avg_error = total_error / total_models if total_models > 0 else 0
        
        print("📈 ESTATÍSTICAS GERAIS:")
        print(f"   📊 Erro médio: {avg_error:.3f}m")
        print(f"   📊 Erro máximo: {max_error:.3f}m")
        print(f"   📊 Erro mínimo: {min_error:.3f}m")
        print()
        
        # Avaliação
        if avg_error < 1.0:
            print("✅ EXCELENTE! Cálculo GPS está muito preciso (< 1m de erro)")
        elif avg_error < 5.0:
            print("✅ BOM! Cálculo GPS está preciso (< 5m de erro)")
        elif avg_error < 10.0:
            print("⚠️  ACEITÁVEL! Cálculo GPS tem precisão moderada (< 10m de erro)")
        else:
            print("❌ PROBLEMA! Cálculo GPS precisa de ajustes (> 10m de erro)")
        
        return {
            'total_models': total_models,
            'avg_error': avg_error,
            'max_error': max_error,
            'min_error': min_error
        }
    
    def save_results_to_csv(self, filename: str = "validacao_gps.csv"):
        """Salva os resultados em um arquivo CSV"""
        if not self.gps_data:
            print("❌ Nenhum dado para salvar!")
            return
        
        data_rows = []
        for model_name, data in self.gps_data.items():
            gps_ardupilot = data['gps_ardupilot']
            gps_calculado = data['gps_calculado']
            pose = data['pose_gazebo']
            
            error_meters = self.calculate_distance_gps(
                gps_ardupilot[0], gps_ardupilot[1],
                gps_calculado[0], gps_calculado[1]
            )
            
            data_rows.append({
                'Modelo': model_name,
                'X_Gazebo': pose[0],
                'Y_Gazebo': pose[1],
                'Z_Gazebo': pose[2],
                'Lat_ArduPilot': gps_ardupilot[0],
                'Lon_ArduPilot': gps_ardupilot[1],
                'Lat_Calculado': gps_calculado[0],
                'Lon_Calculado': gps_calculado[1],
                'Erro_Metros': error_meters
            })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"💾 Resultados salvos em: {filename}")
    
    def run_validation(self, duration: int = 10):
        """Executa a validação completa"""
        try:
            print("🚀 Iniciando validação GPS...")
            self.collect_data(duration)
            
            if self.gps_data:
                stats = self.analyze_results()
                self.save_results_to_csv()
                return stats
            else:
                print("❌ Nenhum dado GPS coletado!")
                print("💡 Verifique se:")
                print("   - O Gazebo está rodando")
                print("   - O ArduPilot SITL está conectado")
                print("   - O tópico /mavros/global_position/global está ativo")
                return None
                
        except KeyboardInterrupt:
            print("\n⏹️  Validação interrompida pelo usuário")
        except Exception as e:
            print(f"❌ Erro durante validação: {e}")

def main():
    """Função principal"""
    print("🔍 VALIDADOR GPS - ArduPilot vs Gazebo2CSV.py")
    print("=" * 50)
    
    validator = GPSValidator()
    
    # Aguarda um pouco para inicializar
    rospy.sleep(2)
    
    # Executa validação por 10 segundos
    stats = validator.run_validation(duration=10)
    
    if stats:
        print(f"\n🎯 Validação concluída!")
        print(f"   Modelos analisados: {stats['total_models']}")
        print(f"   Erro médio: {stats['avg_error']:.3f}m")
    else:
        print("\n❌ Validação falhou!")

if __name__ == "__main__":
    main()

