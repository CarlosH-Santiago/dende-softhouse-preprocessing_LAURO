from dende_statistics import Statistics
from typing import Dict, List, Set, Any
import math

class MissingValueProcessor:
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        # faltou a validação se as columns existem no dataset, o que pode levar a erros silenciosos,
        # pois o método pode retornar uma lista com colunas que não existem 
        return list(columns) if columns else list(self.dataset.keys())

    def _get_num_rows(self) -> int:
        if not self.dataset: return 0
        # isso pode ser simplificado, considerando que todas as colunas têm o mesmo número de linhas, 
        # basta pegar o tamanho de qualquer coluna
        return len(next(iter(self.dataset.values())))

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:

        # aqui temos o problema de validação das colunas
        target_cols = self._get_target_columns(columns)
        n_rows = self._get_num_rows()

        # vocês deveriam ter criado um método para criar um dataset vazio com as mesmas colunas, 
        # para evitar repetição de código
        # def create_empty_dataset(self) -> Dict[str, List[Any]]:
        #     return {col: [] for col in self.dataset}
        novo_dataset = {col: [] for col in self.dataset}
        
        for i in range(n_rows):
            if any(self.dataset[col][i] is None for col in target_cols):
                # essa validação deveria ter sido feita antes do loop, para evitar iterações desnecessárias
                for col in self.dataset: 
                    novo_dataset[col].append(self.dataset[col][i])
        return novo_dataset

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        # mesmos problemas do isna, validação das colunas e criação de dataset vazio
        target_cols = self._get_target_columns(columns)
        n_rows = self._get_num_rows()
        novo_dataset = {col: [] for col in self.dataset}
        
        # mesmas questões de validação, deveriam ter sido feitas antes do loop para evitar iterações desnecessárias
        for i in range(n_rows):
            if not any(self.dataset[col][i] is None for col in target_cols):
                for col in self.dataset:
                    novo_dataset[col].append(self.dataset[col][i])

        # bônus: vocês poderiam ter criado um método que recebe um booolean e definir se é isna ou notna, 
        # para evitar repetição de código entre os dois métodos
        return novo_dataset

    def fillna(self, columns: Set[str] = None, value: Any = 0) -> Dict[str, List[Any]]:
        # mesmo problema de validação das colunas, deveria ter sido feita antes do loop 
        # para evitar iterações desnecessárias
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            for i in range(len(self.dataset[col])):
                if self.dataset[col][i] is None:
                    self.dataset[col][i] = value

        return self.dataset

    def dropna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        # boooa, gostei muito disso, apesar da não validação das colunas, 
        # eu gostei do reuso do método notna para criar o dataset limpo, isso é uma boa prática de programação
        dataset_limpo = self.notna(columns)
        for col in self.dataset:
            self.dataset[col] = dataset_limpo[col]
        return self.dataset


class Scaler:

    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        # não deveríamos ter uma instância do Statistics aqui
        # para calcular as estatísticas necessárias para o escalonamento, como média, desvio padrão?

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        # vocês deveriam ter criado uma função utilitária para validação dos dados, para evitar repetição de código
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        # validação das colunas, deveria ter sido feita antes do loop para evitar iterações desnecessárias
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            # boooa, validação se a coluna é numérica mesmo, mas eu acho que deveria lançar uma exceção: 

            # if not all(isinstance(v, (int, float)) for v in self.dataset[col] if v is not None):
            #     raise ValueError(f"Coluna '{col}' contém valores não numéricos.")
            valid_values = [v for v in self.dataset[col] if isinstance(v, (int, float))]
            if not valid_values: continue
                
            # boooa escrita
            v_min, v_max = min(valid_values), max(valid_values)

            
            # aqui poderia ter simplificado:
            # self.dataset[col] = [(v - v_min) / (v_max - v_min) if v_max != v_min else 0.0 for v in self.dataset[col]]
            for i in range(len(self.dataset[col])):
                v = self.dataset[col][i]
                if isinstance(v, (int, float)):
                    if v_max == v_min:
                        self.dataset[col][i] = 0.0
                    else:
                        self.dataset[col][i] = (v - v_min) / (v_max - v_min)
        return self.dataset

    def standard_scaler(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        # validação das colunas, deveria ter sido feita antes do loop para evitar iterações desnecessárias
        target_cols = self._get_target_columns(columns)
        for col in target_cols:
            # boooa, validação se a coluna é numérica mesmo, mas eu acho que deveria lançar uma exceção:
            # if not all(isinstance(v, (int, float)) for v in self.dataset[col] if v is not None):
            #     raise ValueError(f"Coluna '{col}' contém valores não numéricos.")
            valid_values = [v for v in self.dataset[col] if isinstance(v, (int, float))]

            # você poderia lançar uma exceção aqui, caso a coluna não tenha valores numéricos, 
            # para evitar que o método continue executando sem fazer nada, o que pode levar a erros silenciosos
            n = len(valid_values)
            if n == 0: continue
                
            # aqui deveria ter simplificado o cálculo da média e do desvio padrão, usando as funções do módulo
            # statistics, por exemplo:
            # mean = statistics.mean(valid_values)
            # std_dev = statistics.stdev(valid_values) 
            mean = sum(valid_values) / n
            variance = sum((v - mean) ** 2 for v in valid_values) / n
            std_dev = math.sqrt(variance)
            
            # aqui você poderia ter simplificado o loop, usando uma compreensão de lista, por exemplo:
            # self.dataset[col] = [(v - mean) / std_dev if std_dev != 0 else 0.0 for v in self.dataset[col]]
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
        # validação das colunas, deveria ter sido feita em um método a parte para evitar repetição de código
        target_cols = list(columns) if columns else list(self.dataset.keys())
        for col in target_cols:
            # vocês deveria validar se os valores da coluna são do tipo string ou categóricos para evitar processar colunas que não são adequadas
            #  para label encoding, e lançar uma exceção caso contrário
            #if not all(isinstance(v, (str, type(None))) for v in self.dataset[col]):
            #    raise ValueError(f"Coluna '{col}' contém valores não categóricos.")
            # unique_vals = sorted(set(self.dataset[col]))
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            # boooa, bem simples e prático
            self.dataset[col] = [mapping[v] if v is not None else None for v in self.dataset[col]]
        return self.dataset

    def oneHot_encode(self, columns: Set[str]) -> Dict[str, List[Any]]:
        # validação das colunas, deveria ter sido feita em um método a parte para evitar repetição de código
        # deveria ter sido validado também se as colunas existem no dataset
        target_cols = list(columns) if columns else []
        n_rows = len(next(iter(self.dataset.values()))) if self.dataset else 0
        for col in target_cols:
            # deveria verificar se a coluna é do tipo string ou categórica para evitar processar colunas que não são adequadas para one-hot encoding, e lançar uma exceção caso contrário
            # if not all(isinstance(v, (str, type(None))) for v in self.dataset[col]):
            #     raise ValueError(f"Coluna '{col}' contém valores não categóricos.")
            if col not in self.dataset: continue
            # poderia ser:
            # unique_vals = sorted(set(self.dataset[col]))
            unique_vals = list(set(v for v in self.dataset[col] if v is not None))
            unique_vals.sort(key=str)
            for val in unique_vals:
                new_col_name = f"{col}_{val}"
                # self.dataset[new_col_name] = [1 if v == val else 0 for v in self.dataset[col]]
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
