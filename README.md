# 🌴 Dende Preprocessing  

O **Dende Preprocessing** é a biblioteca oficial de tratamento de dados da Dende Softhouse, desenvolvida especificamente para o ecossistema de gestão de eventos da empresa.  

Seu objetivo é transformar dados brutos e ruidosos em *datasets* estruturados e prontos para alimentar modelos de recomendação e soluções de inteligência de negócio.

---

## 🎯 Por que o Preprocessing é vital?

No contexto de eventos, lidamos frequentemente com:

- 📊 **Disparidades numéricas** (ex: preços altos vs. avaliações baixas)
- 📝 **Variáveis categóricas complexas** (ex: "VIP", "Pista", "Show")

Este módulo resolve dois problemas críticos:

### 1️⃣ Viés de Escala  
Evita que valores numericamente maiores (ex: R$ 500) dominem variáveis menores (ex: 4.5 estrelas) apenas por sua magnitude.

### 2️⃣ Conversão Categórica  
Transforma dados textuais em vetores numéricos que algoritmos matemáticos conseguem processar adequadamente.

---

## 🚀 Principais Funcionalidades

| Módulo          | Descrição                                   | Principais Métodos                     |
|-----------------|---------------------------------------------|----------------------------------------|
| **MissingValue** | Tratamento de dados nulos (`None`)          | `fillna`, `dropna`, `isna`             |
| **Scaler**       | Normalização e padronização de escalas     | `minMax_scaler`, `standard_scaler`     |
| **Encoder**      | Conversão de dados categóricos             | `label_encode`, `oneHot_encode`        |

---

## 💻 Exemplo Prático: Pipeline de Eventos

Imagine preparar os dados de um evento para o sistema de recomendação:

```python
from dende_preprocessing import Preprocessing

# 1️⃣ Dados extraídos da base de eventos
raw_data = {
    "preco_ingresso": [150.0, 80.0, None, 300.0],
    "avaliacao": [4.8, None, 3.5, 5.0],
    "categoria": ["Show", "Teatro", "Show", "Workshop"]
}

# 2️⃣ Inicializando o motor de processamento
pipeline = Preprocessing(raw_data)

# 3️⃣ Tratando lacunas: Avaliações vazias tornam-se 0
pipeline.fillna(columns={"avaliacao"}, value=0)

# 4️⃣ Normalização: Coloca preços e avaliações na mesma grandeza (0 a 1)
pipeline.scale(columns={"preco_ingresso", "avaliacao"}, method='minMax')

# 5️⃣ Encoding: Transforma categorias em colunas binárias
dataset_final = pipeline.encode(columns={"categoria"}, method='oneHot')

print(dataset_final)
