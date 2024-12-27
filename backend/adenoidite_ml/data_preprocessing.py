import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_adenoiditis_data(input_path, output_path):
    """
    Conjunto de dados de adenoidite pré-processado
    
    Args:
        input_path (str): Caminho para o conjunto de dados em bruto
        output_path (str): Caminho para guardar o conjunto de dados processado
    """
    # Carregar dados
    df = pd.read_csv(input_path)
    
    # Recursos e alvo separados
    X = df.drop(['id_paciente', 'adenoiditis_diagnosis'], axis=1)
    y = df['adenoiditis_diagnosis']
    
    #Variáveis ​​categóricas
    categorical_columns = ['genero']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Imputar valores em falta
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Recursos de escala
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Dicionário de saída
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Guardar dados processados
    for key, value in processed_data.items():
        value.to_csv(f'{output_path}/{key}.csv', index=False)
    
    print("Pré-processamento de dados concluído com sucesso!")
    return processed_data

# Executar o pré-processamento
preprocess_adenoiditis_data(
    'data/raw/adenoidite_dataset_simplificado.csv', 
    'data/processed'
)