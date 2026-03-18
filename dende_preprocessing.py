from dende_statistics import Statistics
from typing import Dict, List, Set, Any
import math

class MissingValueProcessor:
    """
    Processa valores ausentes (representados como None) no dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """Retorna as colunas a serem processadas. Se 'columns' for vazio, retorna todas as colunas."""
        return list(columns) if columns else list(self.dataset.keys())

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Retorna um novo dataset contendo apenas as linhas que possuem
        pelo menos um valor nulo (None) em uma das colunas especificadas.

        Args:
            columns (Set[str]): Um conjunto de nomes de colunas a serem verificadas.
                            Se vazio, todas as colunas são consideradas.

        Returns:
            Dict[str, List[Any]]: Um dicionário representando as linhas com valores nulos.
        """
        target_cols = self._get_target_columns(columns)
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        
        result = {col: [] for col in self.dataset}
        for i in range(n_rows):
            if any(self.dataset[col][i] is None for col in target_cols):
                for col in self.dataset:
                    result[col].append(self.dataset[col][i])
        return result

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Retorna um novo dataset contendo apenas as linhas que não possuem
        valores nulos (None) em nenhuma das colunas especificadas.

        Args:
            columns (Set[str]): Um conjunto de nomes de colunas a serem verificadas.
                               Se vazio, todas as colunas são consideradas.

        Returns:
            Dict[str, List[Any]]: Um dicionário representando as linhas sem valores nulos.
        """
        target_cols = self._get_target_columns(columns)
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        
        result = {col: [] for col in self.dataset}
        for i in range(n_rows):
            if all(self.dataset[col][i] is not None for col in target_cols):
                for col in self.dataset:
                    result[col].append(self.dataset[col][i])
        return result

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        """
        Preenche valores nulos (None) nas colunas especificadas com um valor fixo.
        Modifica o dataset da classe.

        Args:
            columns (Set[str]): Colunas onde o preenchimento será aplicado. 
                               Se vazio, aplica a todas as colunas do dataset.
            value (Any): Valor a ser inserido no lugar de None.

        Returns:
            Preprocessing: A própria instância (self) para permitir encadeamento.
        """
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            self.dataset[col] = [
                value if v is None else v 
                for v in self.dataset[col]
            ]
        return self.dataset

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Remove as linhas que contêm valores nulos (None) nas colunas especificadas.
        Modifica o dataset da classe.

        Args:
            columns (Set[str]): Colunas a serem verificadas para valores nulos. Se vazio, todas as colunas são verificadas.
        """
        target_cols = self._get_target_columns(columns)
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        
        indices_to_keep = []
        for i in range(n_rows):
            if all(self.dataset[col][i] is not None for col in target_cols):
                indices_to_keep.append(i)
        
        for col in self.dataset:
            self.dataset[col] = [self.dataset[col][i] for i in indices_to_keep]
            
        return self.dataset


class Scaler:
    """
    Aplica transformações de escala em colunas numéricas do dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Aplica a normalização Min-Max ($X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$)
        nas colunas especificadas. Modifica o dataset.

        Args:
            columns (Set[str]): Colunas para aplicar o scaler. Se vazio, tenta aplicar a todas.
        """
        target_cols = self._get_target_columns(columns)
        
        for col in target_cols:
            valid_values = [v for v in self.dataset[col] if v is not None]
            if not valid_values:
                continue
                
            v_min, v_max = min(valid_values), max(valid_values)
            
            if v_max == v_min:
                self.dataset[col] = [
                    0.0 if v is not None else None 
                    for v in self.dataset[col]
                ]
            else:
                self.dataset[col] = [
                    (v - v_min) / (v_max - v_min) if v is not None else None
                    for v in self.dataset[col]
                ]
                
        return self.dataset

    def standard_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Aplica a padronização Z-score ($X_{std} = \frac{X - \mu}{\sigma}$)
        nas colunas especificadas. Modifica o dataset.

        Args:
            columns (Set[str]): Colunas para aplicar o scaler. Se vazio, tenta aplicar a todas.
        """
        target_cols = self._get_target_columns(columns)
        
        for col in target_cols:
            valid_values = [v for v in self.dataset[col] if v is not None]
            n = len(valid_values)
            if n == 0:
                continue
                
            mean = sum(valid_values) / n
            variance = sum((v - mean) ** 2 for v in valid_values) / n
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                self.dataset[col] = [
                    0.0 if v is not None else None 
                    for v in self.dataset[col]
                ]
            else:
                self.dataset[col] = [
                    (v - mean) / std_dev if v is not None else None
                    for v in self.dataset[col]
                ]
                
        return self.dataset

class Encoder:
    """
    Aplica codificação em colunas categóricas.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        """
        Converte cada categoria em uma coluna em um número inteiro.
        Modifica o dataset.

        Args:
            columns (Set[str]): Colunas categóricas para codificar.
        """
        target_cols = list(columns) if columns else list(self.dataset.keys())
        
        for col in target_cols:
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            
            self.dataset[col] = [
                mapping[v] if v is not None else None
                for v in self.dataset[col]
            ]
            
        return self.dataset

    def oneHot_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        """
        Cria novas colunas binárias para cada categoria nas colunas especificadas (One-Hot Encoding).
        Modifica o dataset adicionando e removendo colunas.

        Args:
            columns (Set[str]): Colunas categóricas para codificar.
        """
        target_cols = list(columns) if columns else []
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        
        for col in target_cols:
            if col not in self.dataset:
                continue
                
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            
            for val in unique_vals:
                new_col_name = f"{col}_{val}"
                self.dataset[new_col_name] = [
                    1 if self.dataset[col][i] == val else 0
                    for i in range(n_rows)
                ]
                
            del self.dataset[col]
            
        return self.dataset


class Preprocessing:
    """
    Classe principal que orquestra as operações de pré-processamento de dados.
    Nota: Todos os métodos retornam o dicionário de dados (dataset), 
    o que encerra a possibilidade de encadeamento de métodos da classe.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        """
        Valida se todas as listas (colunas) no dicionário do dataset
        têm o mesmo comprimento.
        """
        if not self.dataset:
            return
            
        lengths = {len(col_data) for col_data in self.dataset.values()}
        if len(lengths) > 1:
            raise ValueError("O dataset possui colunas de tamanhos diferentes.")

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.isna(). 
        Retorna um dicionário contendo apenas as linhas com valores nulos.
        """
        return self.missing_values.isna(columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.notna(). 
        Retorna um dicionário contendo apenas as linhas sem valores nulos.
        """
        return self.missing_values.notna(columns)

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.fillna(). 
        Modifica e retorna o dicionário de dados com valores preenchidos.
        """
        return self.missing_values.fillna(columns, value)

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.dropna(). 
        Modifica e retorna o dicionário de dados sem as linhas nulas.
        """
        return self.missing_values.dropna(columns)

    def scale(self, columns: Set[str] = None, method: str = 'minMax') -> Dict[str, List[Any]]:
        """
        Aplica escalonamento e retorna o dicionário de dados modificado.

        Args:
            columns (Set[str]): Colunas para aplicar o escalonamento.
            method (str): O método a ser usado: 'minMax' ou 'standard'.

        Returns:
            Dict[str, List[Any]]: O dataset com as colunas escalonadas.
        """
        if method == 'minMax':
            return self.scaler.minMax_scaler(columns)
        elif method == 'standard':
            return self.scaler.standard_scaler(columns)
        else:
            raise ValueError(f"Método de escalonamento '{method}' não suportado.")

    def encode(self, columns: Set[str], method: str = 'label') -> Dict[str, List[Any]]:
        """
        Aplica codificação e retorna o dicionário de dados modificado.

        Args:
            columns (Set[str]): Colunas para aplicar a codificação.
            method (str): O método a ser usado: 'label' ou 'oneHot'.
        
        Returns:
            Dict[str, List[Any]]: O dataset com as colunas codificadas.
        """
        if method == 'label':
            return self.encoder.label_encode(columns)
        elif method == 'oneHot':
            return self.encoder.oneHot_encode(columns)
        else:
            raise ValueError(f"Método de codificação '{method}' não suportado.")