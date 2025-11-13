#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Programa para marcar vãos em pontos GPS
Permite selecionar uma área no plot e adicionar o nome do vão aos modelos
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import os
import sys


class MarcadorVao:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.selected_points = []
        self.pending_selections = []  # Seleções sem nome ainda
        self.all_selections = []  # Lista de tuplas (indices, nome_regiao)
        self.fig = None
        self.ax = None
        self.selector = None
        self.nome_vao = None
        self.finalized = False
        
    def load_data(self):
        """Carrega os dados do CSV"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"\n✓ Arquivo carregado: {self.csv_file}")
            print(f"  Total de pontos: {len(self.df)}")
            print(f"  Colunas: {', '.join(self.df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Erro ao carregar arquivo: {e}")
            return False
    
    def on_select(self, eclick, erelease):
        """Callback quando uma área é selecionada"""
        # Coordenadas da seleção
        lon1, lat1 = eclick.xdata, eclick.ydata
        lon2, lat2 = erelease.xdata, erelease.ydata
        
        # Garantir que temos min e max corretos
        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
        
        # Encontrar pontos dentro da área selecionada
        mask = (
            (self.df['Longitude'] >= lon_min) & 
            (self.df['Longitude'] <= lon_max) &
            (self.df['Latitude'] >= lat_min) & 
            (self.df['Latitude'] <= lat_max)
        )
        
        selected_indices = self.df[mask].index.tolist()
        
        if len(selected_indices) > 0:
            # Adicionar à lista de seleções pendentes
            self.pending_selections.append(selected_indices)
            
            print(f"\n✓ Região #{len(self.pending_selections)} selecionada:")
            print(f"  Longitude: {lon_min:.6f} a {lon_max:.6f}")
            print(f"  Latitude: {lat_min:.6f} a {lat_max:.6f}")
            print(f"  Pontos: {len(selected_indices)}")
            print(f"\n  Exemplos de modelos:")
            for i, idx in enumerate(selected_indices[:3]):
                print(f"    - {self.df.loc[idx, 'Model Name']}")
            if len(selected_indices) > 3:
                print(f"    ... e mais {len(selected_indices) - 3} modelos")
            
            # Atualizar o plot para mostrar todas as seleções
            self.ax.clear()
            self.plot_points(highlight_pending=True)
            self.fig.canvas.draw()
        else:
            print("\n⚠ Nenhum ponto na área selecionada")
    
    def on_key_press(self, event):
        """Callback quando uma tecla é pressionada"""
        if event.key.lower() == 'f':
            print("\n" + "="*60)
            print("✓ Finalizando seleções...")
            print("="*60)
            self.finalized = True
            plt.close(self.fig)
    
    def plot_points(self, highlight_selected=False, highlight_all_selections=False, highlight_pending=False):
        """Plota os pontos GPS"""
        # Plotar todos os pontos
        self.ax.scatter(
            self.df['Longitude'], 
            self.df['Latitude'],
            c='blue',
            alpha=0.5,
            s=20,
            label='Todos os pontos'
        )
        
        # Destacar seleções pendentes (sem nome ainda)
        if highlight_pending and len(self.pending_selections) > 0:
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
            for i, indices in enumerate(self.pending_selections):
                selected_df = self.df.loc[indices]
                color = colors[i % len(colors)]
                self.ax.scatter(
                    selected_df['Longitude'],
                    selected_df['Latitude'],
                    c=color,
                    alpha=0.8,
                    s=50,
                    label=f'Região #{i+1} ({len(indices)} pts)',
                    marker='*'
                )
        
        # Destacar pontos selecionados com nomes
        elif highlight_all_selections and len(self.all_selections) > 0:
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, (indices, nome_regiao) in enumerate(self.all_selections):
                selected_df = self.df.loc[indices]
                color = colors[i % len(colors)]
                self.ax.scatter(
                    selected_df['Longitude'],
                    selected_df['Latitude'],
                    c=color,
                    alpha=0.8,
                    s=50,
                    label=f'{nome_regiao} ({len(indices)})',
                    marker='*'
                )
        
        # Destacar pontos selecionados atualmente
        elif highlight_selected and len(self.selected_points) > 0:
            selected_df = self.df.loc[self.selected_points]
            self.ax.scatter(
                selected_df['Longitude'],
                selected_df['Latitude'],
                c='red',
                alpha=0.8,
                s=50,
                label=f'Selecionados ({len(self.selected_points)})',
                marker='*'
            )
        
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        
        # Título dinâmico
        if len(self.pending_selections) > 0:
            self.ax.set_title(f'Selecione mais áreas ou pressione F para finalizar\n({len(self.pending_selections)} região(ões) selecionada(s))')
        else:
            self.ax.set_title('Selecione áreas com o mouse (clique e arraste)\nPressione F quando terminar')
        
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
    
    def select_area(self):
        """Permite selecionar uma área no plot"""
        print("\n" + "="*60)
        print("INSTRUÇÕES:")
        print("="*60)
        print("1. Use o mouse para selecionar áreas (clique e arraste)")
        print("2. Você pode fazer múltiplas seleções")
        print("3. Pressione a tecla F quando terminar de selecionar")
        print("4. Depois, digite o nome de cada região em ordem")
        print("="*60 + "\n")
        
        # Criar figura
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.plot_points()
        
        # Adicionar listener de teclado
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Configurar seletor de retângulo
        try:
            # Tentar versão mais nova do matplotlib (com props)
            self.selector = RectangleSelector(
                self.ax,
                self.on_select,
                useblit=True,
                button=[1],  # Botão esquerdo do mouse
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True,
                props=dict(facecolor='red', alpha=0.3)
            )
        except TypeError:
            # Versão mais antiga do matplotlib (sem props)
            self.selector = RectangleSelector(
                self.ax,
                self.on_select,
                drawtype='box',
                useblit=True,
                button=[1],  # Botão esquerdo do mouse
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True,
                rectprops=dict(facecolor='red', alpha=0.3, fill=True)
            )
        
        plt.tight_layout()
        plt.show()
    
    def ask_region_names(self):
        """Pergunta os nomes de cada região selecionada"""
        if not self.pending_selections:
            print("\n⚠ Nenhuma região foi selecionada!")
            return False
        
        print("\n" + "="*60)
        print(f"Digite o nome de cada região ({len(self.pending_selections)} regiões selecionadas)")
        print("="*60)
        
        for i, indices in enumerate(self.pending_selections):
            print(f"\n► REGIÃO #{i+1}:")
            print(f"  Pontos: {len(indices)}")
            print(f"  Exemplos de modelos:")
            for j, idx in enumerate(indices[:3]):
                print(f"    - {self.df.loc[idx, 'Model Name']}")
            if len(indices) > 3:
                print(f"    ... e mais {len(indices) - 3} modelos")
            
            while True:
                nome_regiao = input(f"\n  Nome da Região #{i+1} (ex: ECHO): ").strip().upper()
                if nome_regiao:
                    self.all_selections.append((indices, nome_regiao))
                    print(f"  ✓ Região '{nome_regiao}' definida")
                    break
                else:
                    print("  ✗ Nome vazio! Por favor, digite um nome válido.")
        
        print("\n" + "="*60)
        print(f"✓ Todos os nomes definidos!")
        print("="*60)
        return True
    
    def modify_names(self):
        """Modifica os nomes dos modelos de todas as seleções"""
        if not self.all_selections:
            print("\n✗ Nenhuma região foi selecionada!")
            return False
        
        print(f"\n" + "="*60)
        print(f"Modificando modelos de {len(self.all_selections)} região(ões)...")
        print("="*60)
        
        total_modified = 0
        
        for indices, nome_regiao in self.all_selections:
            print(f"\n► Processando região: {nome_regiao}")
            print(f"  Pontos: {len(indices)}")
            
            modified_count = 0
            for idx in indices:
                original_name = self.df.loc[idx, 'Model Name']
                
                # Verificar se já não tem o nome da região
                if f"::{nome_regiao}::" in original_name:
                    continue
                
                # Dividir o nome em partes (formato: PREFIXO::NOME)
                parts = original_name.split("::")
                
                if len(parts) >= 2:
                    # Inserir o nome da região entre o prefixo e o resto
                    new_name = f"{parts[0]}::{nome_regiao}::{('::'.join(parts[1:]))}"
                else:
                    # Se não tiver ::, apenas adicionar a região no final
                    new_name = f"{original_name}::{nome_regiao}"
                
                self.df.loc[idx, 'Model Name'] = new_name
                modified_count += 1
                
                if modified_count <= 3:
                    print(f"  ✓ {original_name}")
                    print(f"    → {new_name}")
            
            if modified_count > 3:
                print(f"  ... e mais {modified_count - 3} modelos modificados")
            
            print(f"  Total: {modified_count} modelos modificados")
            total_modified += modified_count
        
        print(f"\n" + "="*60)
        print(f"✓ TOTAL GERAL: {total_modified} modelos modificados")
        print("="*60)
        return True
    
    def save_modified_csv(self):
        """Salva o CSV modificado"""
        # Gerar nome do arquivo de saída
        base_name = os.path.splitext(self.csv_file)[0]
        output_file = f"{base_name}_modificado.csv"
        
        try:
            self.df.to_csv(output_file, index=False)
            print(f"\n✓ Arquivo salvo: {output_file}")
            print(f"  Total de linhas: {len(self.df)}")
            return True
        except Exception as e:
            print(f"\n✗ Erro ao salvar arquivo: {e}")
            return False
    
    def run(self):
        """Executa o programa completo"""
        print("\n" + "="*60)
        print("MARCADOR DE REGIÕES/VÃOS - PONTOS GPS")
        print("="*60)
        
        # Carregar dados
        if not self.load_data():
            return False
        
        # Selecionar áreas (múltiplas seleções em uma única janela)
        self.select_area()
        
        # Verificar se usuário finalizou com F
        if not self.finalized:
            print("\n⚠ Operação cancelada (janela fechada sem pressionar F)")
            return False
        
        # Verificar se houve alguma seleção
        if not self.pending_selections:
            print("\n⚠ Nenhuma região foi selecionada!")
            print("Operação cancelada.")
            return False
        
        # Perguntar nomes das regiões
        if not self.ask_region_names():
            return False
        
        # Modificar nomes de todos os pontos selecionados
        if not self.modify_names():
            return False
        
        # Salvar CSV modificado
        if not self.save_modified_csv():
            return False
        
        print("\n" + "="*60)
        print("✓ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*60 + "\n")
        return True


def main():
    """Função principal"""
    # Verificar argumentos
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Usar arquivo padrão
        csv_file = "todos_pontos_gps.csv"
    
    # Verificar se arquivo existe
    if not os.path.exists(csv_file):
        print(f"\n✗ Arquivo não encontrado: {csv_file}")
        print(f"\nUso: python {sys.argv[0]} [arquivo.csv]")
        print(f"  Se nenhum arquivo for especificado, usa 'todos_pontos_gps.csv'")
        return 1
    
    # Executar marcador
    marcador = MarcadorVao(csv_file)
    success = marcador.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

