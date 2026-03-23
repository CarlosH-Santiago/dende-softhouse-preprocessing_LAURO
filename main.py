import csv
from dende_preprocessing import Preprocessing

def carregar_csv(caminho):
    dataset = {}
    try:
        with open(caminho, mode='r', encoding='utf-8') as arquivo:
            leitor = csv.DictReader(arquivo)
            if not leitor.fieldnames: return {}
            
            for coluna in leitor.fieldnames:
                dataset[coluna] = []
                
            for linha in leitor:
                for coluna in leitor.fieldnames:
                    valor = linha[coluna]
                    if valor == "" or valor is None: valor = None
                    else:
                        try:
                            val_float = float(valor)
                            valor = int(val_float) if val_float.is_integer() else val_float
                        except ValueError: pass
                    dataset[coluna].append(valor)
        return dataset
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        return None

def main():
    caminho = 'spotify_data clean.csv'
    
    print("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ---\n")
    dataset_spotify = carregar_csv(caminho)
    
    if dataset_spotify:
        try:        
            prep = Preprocessing(dataset_spotify)
            
            # Etapa 1: Duplicatas
            print("[Etapa 1]: Removendo duplicatas...")
            prep.drop_duplicates()
            
            # Etapa 2: Nulos
            # Lembra da nossa varredura? Os nulos estão em artist_name e artist_genres
            print("[Etapa 2]: Tratando valores ausentes (NaN)...")
            prep.fillna(columns={'artist_name', 'artist_genres'}, value='Desconhecido')
            
            # Etapa 3: Scalers
            print("[Etapa 3]: Aplicando Transformadores de Escala...")
            prep.scale(columns={'artist_followers', 'track_popularity'}, method='minMax')
            prep.scale(columns={'track_duration_min'}, method='standard')
            
            # Etapa 4: Encoders
            print("[Etapa 4]: Aplicando Encoders (Textos para Números)...")
            prep.encode(columns={'explicit'}, method='label') # True/False vira 1 e 0
            prep.encode(columns={'album_type'}, method='oneHot') # Cria colunas binárias para Album, Single, etc.
            
            print("\n✅ Pipeline Completo executado com sucesso!")
            print(f"\n📊 [Resumo]: O dataset entrou com {len(dataset_spotify.keys())} colunas e saiu com {len(prep.dataset.keys())} colunas (devido ao One-Hot Encoding).")
            print(f"📊 [Resumo]: Linhas finais prontas para o Machine Learning: {len(prep.dataset['track_name'])}.")
            
            # Printa a primeira linha do dataset final para provar que tudo virou número/texto limpo
            print("\n--- AMOSTRA DO DATASET FINAL (Linha 0) ---")
            for chave in prep.dataset.keys():
                print(f"{chave}: {prep.dataset[chave][0]}")
                
        except Exception as e:
            print(f"Erro na execução: {e}")

if __name__ == "__main__":
    main()
