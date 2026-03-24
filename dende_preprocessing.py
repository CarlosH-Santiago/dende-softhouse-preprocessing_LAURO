from dende_statistics import Statistics
from typing import Dict, List, Set, Any
import math

class MissingValueProcessor:
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def _get_num_rows(self) -> int:
        if not self.dataset: return 0
        return len(next(iter(self.dataset.values())))

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        n_rows = self._get_num_rows()
        novo_dataset = {col: [] for col in self.dataset}
        
        for i in range(n_rows):
            if any(self.dataset[col][i] is None for col in target_cols):
                for col in self.dataset:
                    novo_dataset[col].append(self.dataset[col][i])
        return novo_dataset

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        n_rows = self._get_num_rows()
        novo_dataset = {col: [] for col in self.dataset}
        
        for i in range(n_rows):
            if not any(self.dataset[col][i] is None for col in target_cols):
                for col in self.dataset:
                    novo_dataset[col].append(self.dataset[col][i])
        return novo_dataset

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            for i in range(len(self.dataset[col])):
                if self.dataset[col][i] is None:
                    self.dataset[col][i] = value
        return self.dataset

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        dataset_limpo = self.notna(columns)
        for col in self.dataset:
            self.dataset[col] = dataset_limpo[col]
        return self.dataset


class Scaler:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            valid_values = [v for v in self.dataset[col] if isinstance(v, (int, float))]
            if not valid_values: continue
                
            v_min, v_max = min(valid_values), max(valid_values)
            for i in range(len(self.dataset[col])):
                v = self.dataset[col][i]
                if isinstance(v, (int, float)):
                    if v_max == v_min:
                        self.dataset[col][i] = 0.0
                    else:
                        self.dataset[col][i] = (v - v_min) / (v_max - v_min)
        return self.dataset

    def standard_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            valid_values = [v for v in self.dataset[col] if isinstance(v, (int, float))]
            n = len(valid_values)
            if n == 0: continue
                
            mean = sum(valid_values) / n
            variance = sum((v - mean) ** 2 for v in valid_values) / n
            std_dev = math.sqrt(variance)
            
            for i in range(len(self.dataset[col])):
                v = self.dataset[col][i]
                if isinstance(v, (int, float)):
                    if std_dev == 0:
                        self.dataset[col][i] = 0.0
                    else:
                        self.dataset[col][i] = (v - mean) / std_dev
        return self.dataset


class Encoder:
    
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        target_cols = list(columns) if columns else list(self.dataset.keys())
        for col in target_cols:
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.dataset[col] = [mapping[v] if v is not None else None for v in self.dataset[col]]
        return self.dataset

    def oneHot_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        target_cols = list(columns) if columns else []
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        for col in target_cols:
            if col not in self.dataset: continue
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            for val in unique_vals:
                new_col_name = f"{col}_{val}"
                self.dataset[new_col_name] = [1 if self.dataset[col][i] == val else 0 for i in range(n_rows)]
            del self.dataset[col]
        return self.dataset


class Preprocessing:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        if not isinstance(self.dataset, dict):
            raise ValueError("O dataset deve ser um dicionário (mapa).")
        lengths = [len(v) for v in self.dataset.values() if isinstance(v, list)]
        if lengths and len(set(lengths)) > 1: 
            raise ValueError("Todas as colunas devem possuir o mesmo tamanho.")

    def drop_duplicates(self) -> Dict[str, List[Any]]:
        colunas = list(self.dataset.keys())
        if not colunas: return self.dataset
            
        num_linhas = len(self.dataset[colunas[0]])
        linhas_vistas = set() 
        dataset_limpo = {coluna: [] for coluna in colunas}
        
        for i in range(num_linhas):
            linha_atual = tuple(self.dataset[col][i] for col in colunas)
            if linha_atual not in linhas_vistas:
                linhas_vistas.add(linha_atual) 
                for col in colunas:
                    dataset_limpo[col].append(self.dataset[col][i])
                    
        linhas_removidas = num_linhas - len(linhas_vistas)
        print(f"[Drop Duplicates]: {linhas_removidas} linhas duplicadas foram para o lixo.")
        
        self.dataset = dataset_limpo
        self.statistics.dataset = self.dataset
        self.missing_values.dataset = self.dataset
        self.scaler.dataset = self.dataset
        self.encoder.dataset = self.dataset
        return self.dataset

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.isna(columns)
    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.notna(columns)
    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]: return self.missing_values.fillna(columns, value)
    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]: return self.missing_values.dropna(columns)

    def scale(self, columns: Set[str] = None, method: str = 'minMax') -> Dict[str, List[Any]]:
        if method == 'minMax': return self.scaler.minMax_scaler(columns)
        elif method == 'standard': return self.scaler.standard_scaler(columns)
        else: raise ValueError(f"Método '{method}' não suportado.")

    def encode(self, columns: Set[str], method: str = 'label') -> Dict[str, List[Any]]:
        if method == 'label': return self.encoder.label_encode(columns)
        elif method == 'oneHot': return self.encoder.oneHot_encode(columns)
        else: raise ValueError(f"Método '{method}' não suportado.")
