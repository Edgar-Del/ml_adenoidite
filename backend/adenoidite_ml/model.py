import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report

def train_knn_model(X_train, y_train, X_test, y_test):
    """
    Treinar o classificador KNN para o diagnóstico de adenoidite
   
    Args:
        X_train (pd.DataFrame): Recursos de Treinamento
        y_train (pd.Series): Etiquetas de treino
        X_test (pd.DataFrame): Recursos de Teste
        y_test (pd.Series): Etiquetas de Testes
    
    Returns:
        KNeighborsClassifier: Modelo KNN treinado
    """
    # Inicializar e treinar o KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predizer/Prever e avaliar
    y_pred = knn.predict(X_test)
    
    print("Relatório de classificação (KNN):")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(knn, 'models/knn_model.joblib')
    
    return knn

def train_kmeans_model(X_train):
    """
   Treinar modelo de cluster K-means
    
    Args:
        X_train (pd.DataFrame): Recursos de Treinos
    
    Returns:
        KMeans: Modelo K-means treinado
    """
    # Inicializar e treinar K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    
    # Avaliação do clustering
    silhouette_avg = silhouette_score(X_train, kmeans.labels_)
    print(f"Pontuação: {silhouette_avg}")
    
    # Guardar modelo
    joblib.dump(kmeans, 'models/kmeans_model.joblib')
    
    return kmeans

def main():
    # Carregar dados pré-processados
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    #
    # Modelos de treino

    knn_model = train_knn_model(X_train, y_train, X_test, y_test)
    kmeans_model = train_kmeans_model(X_train)

if __name__ == "__main__":
    main()