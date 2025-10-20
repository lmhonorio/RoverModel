#!/usr/bin/env python3
"""
Exemplo de uso do Rover GPS Controller

Este script demonstra como usar o RoverGPSController de forma personalizada.
"""

import sys
import os
from rover_gps_controller import RoverGPSController

def exemplo_basico():
    """Exemplo básico de uso"""
    print("🎯 Exemplo Básico - Rover GPS Controller")
    print("=" * 50)
    
    # Cria controlador com configurações padrão
    controller = RoverGPSController()
    
    try:
        # Conecta ao rover
        if not controller.connect():
            print("❌ Falha na conexão")
            return False
        
        # Executa comparação completa
        success = controller.run_gps_comparison()
        
        if success:
            # Salva resultados
            controller.save_results("resultados_basico.csv")
            
            # Exibe resumo
            controller.print_summary()
        
        return success
        
    finally:
        controller.disconnect()

def exemplo_personalizado():
    """Exemplo com configurações personalizadas"""
    print("🎯 Exemplo Personalizado - Rover GPS Controller")
    print("=" * 50)
    
    # Cria controlador com configurações personalizadas
    controller = RoverGPSController(
        udp_channel="udp:127.0.0.1:14552",  # Canal diferente
        source_system=2                      # Sistema diferente
    )
    
    try:
        # Conecta com mais tentativas
        if not controller.connect(max_attempts=30):
            print("❌ Falha na conexão")
            return False
        
        # Carrega pontos manualmente
        points = controller.load_csv_points("todos_pontos_gps.csv")
        if not points:
            print("❌ Falha ao carregar pontos")
            return False
        
        # Processa apenas os primeiros 3 pontos
        points_limitados = points[:3]
        print(f"📋 Processando apenas {len(points_limitados)} pontos")
        
        # Aguarda fix GPS
        if not controller.wait_for_gps_fix(timeout=60.0):
            print("❌ Falha ao obter fix GPS")
            return False
        
        # Cria missão
        if not controller.create_mission_from_points(points_limitados):
            print("❌ Falha ao criar missão")
            return False
        
        # Inicia missão
        if not controller.start_mission():
            print("❌ Falha ao iniciar missão")
            return False
        
        # Processa cada ponto manualmente
        for i, point in enumerate(points_limitados):
            print(f"\n📍 Processando {point['Model Name']}...")
            
            # Aguarda waypoint
            if controller.wait_for_waypoint_reached(i, timeout=120.0):
                # Aguarda estabilização
                import time
                time.sleep(10.0)  # Mais tempo para estabilizar
                
                # Obtém posição
                current_pos = controller.get_current_position()
                if current_pos:
                    # Compara
                    comparison = controller.compare_positions(
                        point['Latitude'], point['Longitude'],
                        current_pos['lat'], current_pos['lon']
                    )
                    
                    # Exibe resultado detalhado
                    print(f"📊 {point['Model Name']}:")
                    print(f"   Erro total: {comparison['distance_m']:.3f}m")
                    print(f"   Erro Lat: {comparison['lat_diff_m']:+.3f}m")
                    print(f"   Erro Lon: {comparison['lon_diff_m']:+.3f}m")
                    print(f"   GPS Esperado: ({point['Latitude']:.8f}, {point['Longitude']:.8f})")
                    print(f"   GPS Atual: ({current_pos['lat']:.8f}, {current_pos['lon']:.8f})")
                    
                    # Armazena resultado
                    comparison.update({
                        'point_name': point['Model Name'],
                        'point_id': point['ID'],
                        'point_index': i + 1,
                        'timestamp': time.time()
                    })
                    controller.results.append(comparison)
        
        # Salva resultados personalizados
        controller.save_results("resultados_personalizado.csv")
        controller.print_summary()
        
        return True
        
    finally:
        controller.disconnect()

def exemplo_apenas_telemetria():
    """Exemplo que apenas monitora posição sem enviar missão"""
    print("🎯 Exemplo Telemetria - Monitoramento de Posição")
    print("=" * 50)
    
    controller = RoverGPSController()
    
    try:
        if not controller.connect():
            print("❌ Falha na conexão")
            return False
        
        if not controller.wait_for_gps_fix():
            print("❌ Falha ao obter fix GPS")
            return False
        
        print("📡 Monitorando posição do rover (Ctrl+C para parar)...")
        
        import time
        for i in range(30):  # Monitora por 30 segundos
            pos = controller.get_current_position()
            if pos:
                print(f"📍 Posição {i+1:2d}: "
                      f"Lat={pos['lat']:.8f}, "
                      f"Lon={pos['lon']:.8f}, "
                      f"Alt={pos['alt']:.1f}m, "
                      f"HDG={pos['heading']:.1f}°")
            else:
                print(f"⚠️ Posição {i+1:2d}: Não disponível")
            
            time.sleep(1.0)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
        return True
    finally:
        controller.disconnect()

def main():
    """Menu principal"""
    print("🚀 Exemplos de Uso - Rover GPS Controller")
    print("=" * 60)
    print("Escolha um exemplo:")
    print("1. Exemplo Básico (completo)")
    print("2. Exemplo Personalizado (primeiros 3 pontos)")
    print("3. Exemplo Telemetria (apenas monitoramento)")
    print("4. Sair")
    
    while True:
        try:
            escolha = input("\nDigite sua escolha (1-4): ").strip()
            
            if escolha == '1':
                return exemplo_basico()
            elif escolha == '2':
                return exemplo_personalizado()
            elif escolha == '3':
                return exemplo_apenas_telemetria()
            elif escolha == '4':
                print("👋 Saindo...")
                return True
            else:
                print("❌ Opção inválida! Digite 1, 2, 3 ou 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
