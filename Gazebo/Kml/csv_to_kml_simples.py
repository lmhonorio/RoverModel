#!/usr/bin/env python3
"""
Conversor CSV para KML Simplificado
Versão mais simples e otimizada para visualização rápida
"""

import pandas as pd
import os
import ast
from typing import List, Tuple

class SimpleCSVToKML:
    def __init__(self, csv_file: str = "../todos_pontos_gps.csv"):
        """Inicializa o conversor simplificado"""
        self.csv_file = csv_file
        self.gazebo_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.gazebo_dir, csv_file)
        
        print("🚀 Conversor CSV para KML Simplificado")
        print(f"📁 Diretório: {self.gazebo_dir}")
    
    def load_csv_data(self) -> pd.DataFrame:
        """Carrega os dados do arquivo CSV"""
        try:
            if not os.path.exists(self.csv_path):
                print(f"❌ Arquivo não encontrado: {self.csv_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.csv_path)
            print(f"✅ CSV carregado: {len(df)} objetos")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar CSV: {e}")
            return pd.DataFrame()
    
    def create_simple_kml(self, df: pd.DataFrame) -> str:
        """Cria KML simplificado"""
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Objetos Gazebo - Parnaíba III</name>
    <description>Objetos extraídos do mundo Gazebo</description>
    
    <!-- Estilos -->
    <Style id="reactor">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.5</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
    
    <Style id="tpc">
      <IconStyle>
        <color>ffff0000</color>
        <scale>1.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/square.png</href></Icon>
      </IconStyle>
    </Style>
    
    <Style id="other">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
    
    <!-- Objetos -->
'''
        
        # Adiciona cada objeto
        for _, row in df.iterrows():
            model_name = row['Model Name']
            latitude = row['Latitude']
            longitude = row['Longitude']
            altitude = row['Altitude']
            
            # Determina estilo baseado no nome
            if 'REATOR' in model_name.upper():
                style = "reactor"
            elif 'TPC' in model_name.upper():
                style = "tpc"
            else:
                style = "other"
            
            # Adiciona Placemark
            kml_content += f'''    <Placemark>
      <name>{model_name}</name>
      <description>ID: {row['ID']}
Largura: {row['Vx_largura']:.2f}m
Altura: {row['Vy_altura']:.2f}m</description>
      <styleUrl>#{style}</styleUrl>
      <Point>
        <coordinates>{longitude},{latitude},{altitude}</coordinates>
      </Point>
    </Placemark>
'''
        
        kml_content += '''  </Document>
</kml>'''
        
        return kml_content
    
    def save_kml(self, kml_content: str, output_file: str = "objetos_gazebo_simples.kml"):
        """Salva o KML em arquivo"""
        try:
            output_path = os.path.join(self.gazebo_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            
            print(f"💾 KML salvo: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar KML: {e}")
            return None
    
    def convert(self, output_file: str = "objetos_gazebo_simples.kml"):
        """Executa a conversão"""
        print("\n🔄 Convertendo CSV para KML...")
        
        # Carrega dados
        df = self.load_csv_data()
        if df.empty:
            return None
        
        # Cria KML
        print("📝 Criando KML...")
        kml_content = self.create_simple_kml(df)
        
        # Salva arquivo
        print("💾 Salvando arquivo...")
        output_path = self.save_kml(kml_content, output_file)
        
        if output_path:
            print(f"\n✅ Conversão concluída!")
            print(f"📄 Arquivo: {output_path}")
            print(f"📊 Objetos: {len(df)}")
            
            # Estatísticas
            reatores = df[df['Model Name'].str.contains('REATOR', case=False, na=False)]
            tpc = df[df['Model Name'].str.contains('TPC', case=False, na=False)]
            outros = df[~df['Model Name'].str.contains('REATOR|TPC', case=False, na=False)]
            
            print(f"   🔴 Reatores: {len(reatores)}")
            print(f"   🔵 TPC: {len(tpc)}")
            print(f"   🟢 Outros: {len(outros)}")
            
            return output_path
        else:
            print("❌ Falha na conversão!")
            return None

def main():
    """Função principal"""
    print("🚀 CONVERSOR CSV PARA KML SIMPLIFICADO")
    print("=" * 50)
    
    # Cria conversor
    converter = SimpleCSVToKML("../todos_pontos_gps.csv")
    
    # Executa conversão
    output_file = converter.convert("objetos_gazebo_simples.kml")
    
    if output_file:
        print(f"\n🎯 Para visualizar:")
        print(f"   1. Abra o Google Earth")
        print(f"   2. File > Open")
        print(f"   3. Selecione: {output_file}")
        print(f"   4. Os objetos aparecerão com cores diferentes!")
        print(f"\n📋 Cores:")
        print(f"   🔴 Vermelho: Reatores")
        print(f"   🔵 Azul: TPC (Transformadores)")
        print(f"   🟢 Verde: Outros objetos")
    else:
        print("\n❌ Conversão falhou!")

if __name__ == "__main__":
    main()
