import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score, 
    accuracy_score, 
    classification_report,
    mean_squared_error,
    r2_score
)
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA

class AdenoiditisAnalysis:
    def __init__(self, data_path):
        """Initialize the analysis with data path"""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess the data for analysis"""
        # Separate features and target
        self.X = self.df.drop(['id_paciente', 'adenoiditis_diagnosis'], axis=1)
        self.y = self.df['adenoiditis_diagnosis']
        
        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

    def kmeans_analysis(self, n_clusters=3):
        """Perform K-means clustering analysis"""
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_labels = kmeans.fit_predict(self.X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_scaled, self.kmeans_labels)
        
        # Visualize results using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.kmeans_labels, cmap='viridis')
        plt.title('K-means Clustering Results (PCA)')
        plt.colorbar(scatter)
        plt.savefig('results/kmeans_clusters.png')
        plt.close()
        
        return silhouette_avg

    def knn_analysis(self, n_neighbors=5):
        """Perform K-Nearest Neighbors analysis"""
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.X_train, self.y_train)
        
        y_pred = knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('KNN Confusion Matrix')
        plt.savefig('results/knn_confusion_matrix.png')
        plt.close()
        
        return accuracy, classification_report(self.y_test, y_pred)

    def hierarchical_clustering(self):
        """Perform hierarchical clustering analysis"""
        # Create dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram = shc.dendrogram(shc.linkage(self.X_scaled, method='ward'))
        plt.title('Dendrogram')
        plt.savefig('results/dendrogram.png')
        plt.close()
        
        # Perform hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=3)
        hc_labels = hc.fit_predict(self.X_scaled)
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, cmap='viridis')
        plt.title('Hierarchical Clustering Results (PCA)')
        plt.colorbar(scatter)
        plt.savefig('results/hierarchical_clusters.png')
        plt.close()
        
        return silhouette_score(self.X_scaled, hc_labels)

    def linear_regression_analysis(self):
        """Perform linear regression analysis"""
        # Use a continuous target variable for regression
        continuous_target = self.X_scaled['snoring_frequency']  # Example continuous variable
        X_reg = self.X_scaled.drop('snoring_frequency', axis=1)
        
        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, continuous_target, test_size=0.2, random_state=42
        )
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_train_reg, y_train_reg)
        y_pred_reg = lr.predict(X_test_reg)
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_reg, y_pred_reg)
        plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.savefig('results/linear_regression.png')
        plt.close()
        
        return r2_score(y_test_reg, y_pred_reg)

    def decision_tree_analysis(self):
        """Perform decision tree analysis"""
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        y_pred = dt.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Decision Tree Feature Importance')
        plt.savefig('results/decision_tree_importance.png')
        plt.close()
        
        return accuracy, classification_report(self.y_test, y_pred)

    def random_forest_analysis(self):
        """Perform random forest analysis"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        y_pred = rf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Random Forest Feature Importance')
        plt.savefig('results/random_forest_importance.png')
        plt.close()
        
        return accuracy, classification_report(self.y_test, y_pred)

def main():
    # Initialize analysis
    analysis = AdenoiditisAnalysis('data/raw/adenoidite_dataset_simplificado.csv')
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Perform analyses and print results
    print("Running comprehensive adenoiditis analysis...")
    
    # 1. K-means Analysis
    print("\n1. K-means Clustering")
    silhouette_avg = analysis.kmeans_analysis()
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    # 2. KNN Analysis
    print("\n2. K-Nearest Neighbors")
    knn_accuracy, knn_report = analysis.knn_analysis()
    print(f"Accuracy: {knn_accuracy:.3f}")
    print("Classification Report:")
    print(knn_report)
    
    # 3. Hierarchical Clustering
    print("\n3. Hierarchical Clustering")
    hc_silhouette = analysis.hierarchical_clustering()
    print(f"Silhouette Score: {hc_silhouette:.3f}")
    
    # 4. Linear Regression
    print("\n4. Linear Regression")
    r2_score = analysis.linear_regression_analysis()
    print(f"R² Score: {r2_score:.3f}")
    
    # 5. Decision Tree
    print("\n5. Decision Tree")
    dt_accuracy, dt_report = analysis.decision_tree_analysis()
    print(f"Accuracy: {dt_accuracy:.3f}")
    print("Classification Report:")
    print(dt_report)
    
    # 6. Random Forest
    print("\n6. Random Forest")
    rf_accuracy, rf_report = analysis.random_forest_analysis()
    print(f"Accuracy: {rf_accuracy:.3f}")
    print("Classification Report:")
    print(rf_report)
    
    print("\nAnalysis complete. Results and visualizations saved in 'results' directory.")

if __name__ == "__main__":
    main()
