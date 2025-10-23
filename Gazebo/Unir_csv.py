#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unir_csv.py - Script para unir dados de Taludes_marker.csv em todos_pontos_gps
Adiciona os taludes ao arquivo CSV e XLSX mantendo os dados existentes
"""

import os
import pandas as pd
from openpyxl import load_workbook


class UnirCSV:
    def __init__(self):
        # Obter o diretório do script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Caminhos dos arquivos
        self.taludes_csv = os.path.join(self.script_dir, 'Taludes_marker.csv')
        self.todos_csv = os.path.join(self.script_dir, 'todos_pontos_gps.csv')
        self.todos_xlsx = os.path.join(self.script_dir, 'todos_pontos_gps.xlsx')
        
    def verificar_arquivos(self):
        """Verifica se os arquivos necessários existem"""
        if not os.path.exists(self.taludes_csv):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.taludes_csv}")
        
        if not os.path.exists(self.todos_csv):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.todos_csv}")
        
        if not os.path.exists(self.todos_xlsx):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.todos_xlsx}")
        
        print("✅ Todos os arquivos necessários foram encontrados")
    
    def unir_csv(self):
        """Une o CSV de taludes com o CSV de todos os pontos"""
        print("\n📝 Unindo arquivos CSV...")
        
        try:
            # Ler os arquivos
            df_taludes = pd.read_csv(self.taludes_csv)
            df_todos = pd.read_csv(self.todos_csv)
            
            print(f"  📊 Taludes: {len(df_taludes)} registros")
            print(f"  📊 Todos os pontos: {len(df_todos)} registros")
            
            # Remover linhas vazias de taludes
            df_taludes = df_taludes.dropna(how='all')
            
            # Verificar se os taludes já existem
            ids_existentes = set(df_todos['ID'].values)
            df_taludes_novos = df_taludes[~df_taludes['ID'].isin(ids_existentes)]
            
            if len(df_taludes_novos) == 0:
                print("  ⚠️  Todos os taludes já existem no arquivo de pontos")
                return False
            
            print(f"  ➕ Adicionando {len(df_taludes_novos)} novo(s) talude(s)")
            
            # Unir os DataFrames
            df_unido = pd.concat([df_todos, df_taludes_novos], ignore_index=True)
            
            # Salvar o arquivo unido
            df_unido.to_csv(self.todos_csv, index=False)
            print(f"  ✅ Arquivo CSV salvo com {len(df_unido)} registros")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Erro ao unir CSV: {e}")
            raise
    
    def unir_xlsx(self):
        """Une o CSV de taludes com o XLSX de todos os pontos"""
        print("\n📊 Unindo arquivos XLSX...")
        
        try:
            # Ler os arquivos
            df_taludes = pd.read_csv(self.taludes_csv)
            df_todos = pd.read_excel(self.todos_xlsx, sheet_name='Sheet1', engine='openpyxl')
            
            print(f"  📊 Taludes: {len(df_taludes)} registros")
            print(f"  📊 Todos os pontos (XLSX): {len(df_todos)} registros")
            
            # Remover linhas vazias de taludes
            df_taludes = df_taludes.dropna(how='all')
            
            # Verificar se os taludes já existem
            ids_existentes = set(df_todos['ID'].values)
            df_taludes_novos = df_taludes[~df_taludes['ID'].isin(ids_existentes)]
            
            if len(df_taludes_novos) == 0:
                print("  ⚠️  Todos os taludes já existem no arquivo XLSX")
                return False
            
            print(f"  ➕ Adicionando {len(df_taludes_novos)} novo(s) talude(s)")
            
            # Unir os DataFrames
            df_unido = pd.concat([df_todos, df_taludes_novos], ignore_index=True)
            
            # Carregar o workbook
            wb = load_workbook(self.todos_xlsx)
            
            # Remover a aba Sheet1 existente
            if 'Sheet1' in wb.sheetnames:
                del wb['Sheet1']
            
            # Criar nova aba com os dados unidos
            ws = wb.create_sheet('Sheet1', 0)
            
            # Escrever headers
            for col_idx, col_name in enumerate(df_unido.columns, 1):
                ws.cell(row=1, column=col_idx, value=col_name)
            
            # Escrever dados
            for row_idx, row in enumerate(df_unido.itertuples(index=False), 2):
                for col_idx, value in enumerate(row, 1):
                    ws.cell(row=row_idx, column=col_idx, value=value)
            
            # Manter a aba de parâmetros
            if 'ParametrosConversao' not in wb.sheetnames:
                print("  ⚠️  Aba ParametrosConversao não encontrada, ela será mantida se existir")
            
            # Salvar o workbook
            wb.save(self.todos_xlsx)
            print(f"  ✅ Arquivo XLSX salvo com {len(df_unido)} registros")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Erro ao unir XLSX: {e}")
            raise
    
    def run(self):
        """Função principal"""
        print("=" * 60)
        print("🔄 UNINDO DADOS DE TALUDES COM PONTOS GPS")
        print("=" * 60)
        
        try:
            # Verificar arquivos
            self.verificar_arquivos()
            
            # Unir CSV
            csv_unido = self.unir_csv()
            
            # Unir XLSX
            xlsx_unido = self.unir_xlsx()
            
            # Resumo
            print("\n" + "=" * 60)
            if csv_unido or xlsx_unido:
                print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
                print("=" * 60)
                print(f"📁 CSV: {self.todos_csv}")
                print(f"📁 XLSX: {self.todos_xlsx}")
            else:
                print("⚠️  NENHUM DADO NOVO FOI ADICIONADO")
                print("=" * 60)
                
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            return False
        
        return True


def main():
    try:
        unir = UnirCSV()
        sucesso = unir.run()
        exit(0 if sucesso else 1)
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        exit(1)


if __name__ == '__main__':
    main()
