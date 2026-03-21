from dende_statistics import Statistics
from typing import Dict, List, Set, Any

class MissingValueProcessor:
    # Classe responsável por achar e tratar os buracos (None) nos dados
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: pass
    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: pass
    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]: pass
    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: pass

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
