from dende_statistics import Statistics
from typing import Dict, List, Set, Any

class MissingValueProcessor:
    # Classe responsável por achar e tratar os buracos (None) nos dados
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def _get_num_rows(self) -> int:
        # da mesma forma como vamos saber quantas colunas tem, é improtante saber quantas linhas também tem
        if not self.dataset:
            return 0
        # Pega a primeira coluna do dicionário e verifica o tamanho da lista dela
        primeira_coluna = next(iter(self.dataset.values()))
        return len(primeira_coluna)

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        # 1. Definir quais colunas vamos olhar
        target_cols = self._get_target_columns(columns)
        num_rows = self._get_num_rows()

        novo_dataset = {col: [] for col in self.dataset}

        for i in range(num_rows):
            # vai verificar se tem pelo menos um none
            tem_nulo = any(self.dataset[col][i] is None for col in target_cols)

            # se achou um nulo vai copiar a coluna e mandar para o novo dataset
            if tem_nulo:
                for col in self.dataset:
                    novo_dataset[col].append(self.dataset[col][i])

        return novo_dataset

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        num_rows = self._get_num_rows()

        novo_dataset = {col: [] for col in self.dataset}

        for i in range(num_rows):
            tem_nulo = any(self.dataset[col][i] is None for col in target_cols)

            # mesma coisa da função anterior porém se não tem nulo, manda para a nova tabela
            if not tem_nulo:
                for col in self.dataset:
                    novo_dataset[col].append(self.dataset[col][i])

        return novo_dataset

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)

        # Iteramos apenas pelas colunas que queremos modificar
        for col in target_cols:
            for i in range(len(self.dataset[col])):
                if self.dataset[col][i] is None:
                    self.dataset[col][i] = value

        # vai retornar o dataset modificado
        return self.dataset

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        dataset_limpo = self.notna(columns)

        for col in self.dataset:
            self.dataset[col] = dataset_limpo[col]

        return self.dataset

class Scaler:
    # Classe que ajusta as escalas matemáticas (como normalizar notas de 0 a 1)
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]: pass
    def standard_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]: pass

class Encoder:
    # Classe que transforma textos (categorias) em números para o modelo entender
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]) -> Dict[str, List[Any]]: pass
    def oneHot_encode(self, columns: Set[str]) -> Dict[str, List[Any]]: pass

class Preprocessing:
    # Essa é a classe chefe. Ela recebe os dados brutos e distribui para as classes especialistas.
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        # Trava de segurança: garante que o código não vai rodar se as colunas tiverem tamanhos diferentes
        if not isinstance(self.dataset, dict):
            raise ValueError("O dataset deve ser um dicionário (mapa).")
            
        lengths = [len(v) for v in self.dataset.values() if isinstance(v, list)]
        if lengths and len(set(lengths)) > 1: 
            raise ValueError("Todas as colunas devem possuir o mesmo tamanho.")

    def drop_duplicates(self) -> Dict[str, List[Any]]:
        """Varre o dicionário e apaga as linhas que são cópias exatas umas das outras."""
        colunas = list(self.dataset.keys())
        if not colunas:
            return self.dataset
            
        num_linhas = len(self.dataset[colunas[0]])
        linhas_vistas = set() 
        dataset_limpo = {coluna: [] for coluna in colunas}
        
        for i in range(num_linhas):
            # empacotamos a linha inteira em uma tupla.
            # Como tuplas são imutáveis, o Python consegue checar se ela já existe no 'set' de forma super rápida.
            linha_atual = tuple(self.dataset[col][i] for col in colunas)
            
            if linha_atual not in linhas_vistas:
                linhas_vistas.add(linha_atual) # Registra a linha como "já vista"
                
                # Como a linha é inédita, salvamos cada pedaço dela nas colunas do novo dataset
                for col in colunas:
                    dataset_limpo[col].append(self.dataset[col][i])
                    
        linhas_removidas = num_linhas - len(linhas_vistas)
        print(f"[Drop Duplicates]: {linhas_removidas} linhas duplicadas foram para o lixo.")
        
        # atualizamos o dataset das outras classes também. 
        # assim garantimos que as análises futuras serão feitas com os dados limpos.
        self.dataset = dataset_limpo
        self.statistics.dataset = self.dataset
        self.missing_values.dataset = self.dataset
        self.scaler.dataset = self.dataset
        self.encoder.dataset = self.dataset
        
        return self.dataset

    # Atalhos práticos: em vez de chamar prep.missing_values.isna(), chamamos direto prep.isna()
    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.isna(columns)
    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.notna(columns)
    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]: return self.missing_values.fillna(columns, value)
    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.dropna(columns)

    def scale(self, columns: Set[str] = None, method: str = 'minMax') -> Dict[str, List[Any]]:
        if method == 'minMax': return self.scaler.minMax_scaler(columns)
        elif method == 'standard': return self.scaler.standard_scaler(columns)
        else: raise ValueError(f"Método de escalonamento '{method}' não suportado.")

    def encode(self, columns: Set[str], method: str = 'label') -> Dict[str, List[Any]]:
        if method == 'label': return self.encoder.label_encode(columns)
        elif method == 'oneHot': return self.encoder.oneHot_encode(columns)
        else: raise ValueError(f"Método de codificação '{method}' não suportado.")
