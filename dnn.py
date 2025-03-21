# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar o arquivo Excel
file_path = './dados_funcionais_patenteadores.xlsx'  # Substitua com o caminho correto do arquivo
dados = pd.read_excel(file_path)

# Passo 1: Pré-processamento dos dados

# Preencher valores ausentes com a estratégia 'most_frequent' para as colunas categóricas
# Vamos corrigir o código para realizar a imputação apenas nas colunas categóricas
# Preencher valores ausentes com a estratégia 'most_frequent' apenas para as colunas categóricas
imputer = SimpleImputer(strategy='most_frequent')
dados_imputados = dados.copy()

# Colunas categóricas que precisam de imputação
colunas_categoricas = ['Research areas', 'Research subareas', "Bachelor's degree location", 'Master\'s degree location', 
                       'Doctorate\'s degree location', 'Gender', 'Have you ever had a patent awarded?',
                       'Have you ever had any patents licensed?', 'Have you ever had a patent deposited abroad through Patent Cooperation Treaty (PCT)?',
                       'Have any patent request been the result of interaction with the industry?', 'Interaction in patenting process. Active or passive?',
                       'Classification regarding professional orientation', 'Nature of motivation', 'Relationship between standards / personal values', 
                       'Birthplace']  # Incluindo 'Birthplace' para codificação

# Realizando a imputação para as colunas categóricas
for col in colunas_categoricas:
    dados_imputados[col] = imputer.fit_transform(dados[[col]])

# Transformar a coluna 'Birth Interval' para números (já foi feito anteriormente)
dados_imputados['Birth Interval'] = dados_imputados['Birth Interval'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else x)

# Codificar as variáveis categóricas para valores numéricos
label_encoder = LabelEncoder()
for col in colunas_categoricas:
    dados_imputados[col] = label_encoder.fit_transform(dados_imputados[col])

# Passo 2: Definir as variáveis de entrada (X) e a variável alvo (y)
X = dados_imputados.drop(columns=['Identifier', 'Relationship between standards / personal values'])
y = dados_imputados['Relationship between standards / personal values']

# Passo 3: Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Criar o modelo de rede neural (MLP) para prever a última coluna
mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)

# Passo 5: Treinar o modelo
mlp.fit(X_train, y_train)

# Passo 6: Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Passo 7: Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Exibir a acurácia e o relatório de classificação
print(f'Acurácia: {accuracy}')
print(f'Relatório de Classificação:\n{classification_report_result}')

