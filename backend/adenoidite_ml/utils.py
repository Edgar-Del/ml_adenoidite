import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import joblib
import os

class MLUtils:
    @staticmethod
    def load_model(model_path: str):
        """
        Load a trained machine learning model
        
        Args:
            model_path (str): Path to the saved model file
        
        Returns:
            Loaded model object
        """
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            logging.error(f"Model not found at {model_path}")
            raise

    @staticmethod
    def save_model(model, model_path: str):
        """
        Save a trained machine learning model
        
        Args:
            model: Trained model object
            model_path (str): Path to save the model
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            logging.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    @staticmethod
    def validate_input_data(data: pd.DataFrame) -> bool:
        """
        Validate input data for machine learning models
        
        Args:
            data (pd.DataFrame): Input dataframe to validate
        
        Returns:
            bool: Whether the data is valid
        """
        # Check for required columns
        required_columns = [
            'age_months', 'gender', 'nasal_obstruction_severity',
            'mouth_breathing_frequency', 'snoring_intensity',
            'sleep_disruption_score', 'adenoid_size_mm'
        ]
        
        # Check column existence
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.warning(f"Missing columns: {missing_columns}")
            return False
        
        # Check for missing values
        if data.isnull().any().any():
            logging.warning("Input data contains missing values")
            return False
        
        # Additional validation checks
        try:
            # Validate age range (0-60 months)
            if not ((data['age_months'] >= 0) & (data['age_months'] <= 60)).all():
                logging.warning("Invalid age range")
                return False
            
            # Validate severity scores (0-10)
            severity_columns = [
                'nasal_obstruction_severity', 
                'mouth_breathing_frequency', 
                'snoring_intensity', 
                'sleep_disruption_score'
            ]
            for col in severity_columns:
                if not ((data[col] >= 0) & (data[col] <= 10)).all():
                    logging.warning(f"Invalid values in {col}")
                    return False
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False
        
        return True

    @staticmethod
    def generate_diagnosis_report(
        knn_predictions: np.ndarray, 
        kmeans_clusters: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnosis report
        
        Args:
            knn_predictions (np.ndarray): KNN classification predictions
            kmeans_clusters (np.ndarray): KMeans clustering results
        
        Returns:
            Dict containing diagnosis insights
        """
        report = {
            'total_samples': len(knn_predictions),
            'knn_diagnosis': {
                'positive_cases': int(np.sum(knn_predictions)),
                'negative_cases': int(len(knn_predictions) - np.sum(knn_predictions)),
                'positive_percentage': float(np.mean(knn_predictions) * 100)
            },
            'clustering': {
                'cluster_distribution': dict(zip(
                    *np.unique(kmeans_clusters, return_counts=True)
                ))
            },
            'risk_assessment': {
                'high_risk_cluster': int(np.argmax(
                    np.bincount(kmeans_clusters[knn_predictions == 1])
                ))
            }
        }
        return report

    @staticmethod
    def calculate_feature_importance(model, feature_names: list) -> Dict[str, float]:
        """
        Calculate feature importance for interpretability
        
        Args:
            model: Trained machine learning model
            feature_names (list): Names of features
        
        Returns:
            Dict of feature importances
        """
        try:
            # For KNN, we'll use a proxy method
            # This is a simplified approximation
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # For KNN, we might need a different approach
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            return dict(zip(feature_names, importances))
        except Exception as e:
            logging.error(f"Feature importance calculation error: {e}")
            return {}

def setup_logging():
    """
    Configure logging for the ML pipeline
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_pipeline.log'),
            logging.StreamHandler()
        ]
    )

# Export the utility class and setup function
__all__ = ['MLUtils', 'setup_logging']