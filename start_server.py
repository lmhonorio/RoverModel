#!/usr/bin/env python3
"""
Script de inicialização para o servidor de planejamento de missões.
Verifica dependências e inicia o servidor Flask.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Verifica se as dependências estão instaladas."""
    try:
        import flask
        import flask_cors
        print("✅ Flask e Flask-CORS instalados")
        return True
    except ImportError:
        print("❌ Dependências não encontradas")
        print("📦 Instalando dependências...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependências instaladas com sucesso")
            return True
        except subprocess.CalledProcessError:
            print("❌ Falha ao instalar dependências")
            return False

def check_files():
    """Verifica se os arquivos necessários existem."""
    required_files = [
        './jsons/graph9_new.json',
        './jsons/obp_6.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Arquivos necessários não encontrados:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 O servidor tentará executar mesmo assim, mas pode haver erros.")
        return False
    
    print("✅ Todos os arquivos necessários encontrados")
    return True

if __name__ == '__main__':
    print("🚀 Iniciando servidor de planejamento de missões...")
    print("="*60)
    
    # Verificar dependências
    if not check_dependencies():
        print("❌ Não foi possível instalar as dependências necessárias")
        sys.exit(1)
    
    # Verificar arquivos
    check_files()
    
    print("\n🌐 Iniciando servidor Flask...")
    print("📍 URL: http://localhost:5000")
    print("🔄 Para parar o servidor, pressione Ctrl+C")
    print("="*60)
    
    # Importar e executar o servidor
    try:
        from mission_server import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"\n💥 Erro ao iniciar servidor: {e}")
        sys.exit(1)
