def main():
    caminho = 'spotify_data clean.csv' # Atenção ao nome exato do arquivo
    
    print("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ---\n")
    dataset_spotify = carregar_dados_spotify(caminho)
    
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
