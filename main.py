import csv
from dende_preprocessing import Preprocessing

def carregar_dados_spotify(caminho_arquivo):
    print(f"Carregando dados de: {caminho_arquivo}")
    
    # Estrutura base (colunas viram chaves, linhas viram listas)
    dados = {
        'track_name': [], 'artist_name': [], 'artist_genres': [],
        'explicit': [], 'album_type': [], 'track_popularity': [], 
        'artist_popularity': [], 'artist_followers': [], 
        'album_total_tracks': [], 'track_duration_min': []
    }
    
    try:   
        with open(caminho_arquivo, mode='r', encoding='utf-8') as file:
            leitor = csv.DictReader(file)
            contagem = 0
            
            for linha in leitor:
                # Dados categóricos
                dados['track_name'].append(linha['track_name'])
                dados['artist_name'].append(linha['artist_name'])
                dados['artist_genres'].append(linha['artist_genres'])
                dados['explicit'].append(linha['explicit'])
                dados['album_type'].append(linha['album_type'])

                # Conversões numéricas. 
                # Se der erro (ex: campo vazio), salvamos como None para os dados serem tratados pelas funções do Preprocessing 
                try: dados['track_popularity'].append(int(linha['track_popularity']))
                except ValueError: dados['track_popularity'].append(None)
                
                try: dados['artist_popularity'].append(int(linha['artist_popularity']))
                except ValueError: dados['artist_popularity'].append(None) 
                
                try: dados['artist_followers'].append(int(linha['artist_followers']))
                except ValueError: dados['artist_followers'].append(None) 

                try: dados['album_total_tracks'].append(int(linha['album_total_tracks']))
                except ValueError: dados['album_total_tracks'].append(None) 
                
                try: dados['track_duration_min'].append(float(linha['track_duration_min']))
                except ValueError: dados['track_duration_min'].append(None) 
                    
                contagem += 1
                    
        print(f"Leitura concluída! {contagem} linhas processadas.\n")
        return dados
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho_arquivo} não encontrado na pasta.")
        return {}

def main():
    caminho = 'spotify_data_clean.csv'
    
    print("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ---\n")
    dataset_spotify = carregar_dados_spotify(caminho)
    
    if dataset_spotify:
        try:        
            prep = Preprocessing(dataset_spotify)
            
            # Inicia a limpeza passando a vassoura nas linhas idênticas
            print("\n[Etapa 1]: Removendo duplicatas...")
            prep.drop_duplicates()
            
            print("\nPipeline Base executado com sucesso!")
        except Exception as e:
            print(f"Erro na execução: {e}")


if __name__ == "__main__":
    main()
