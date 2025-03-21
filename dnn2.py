import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados (substitua 'SuaÚltimaColuna' pelo nome real da última coluna)
file_path = './dados_funcionais_patenteadores.xlsx'
df = pd.read_excel(file_path)

# Selecione a coluna alvo e as variáveis de entrada
target_column = 'Relationship between standards / personal values'  # Altere para o nome da última coluna
X = df.drop(columns=[target_column])
y = df[target_column]

# Pré-processamento: normalização de variáveis numéricas e codificação de variáveis categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),  # Normaliza dados numéricos
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)  # Codifica dados categóricos
    ])

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o pipeline que aplica o pré-processamento e o modelo MLP
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')
