# Project Structure
"""
adenoiditis_ml/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   └── core/
│       ├── __init__.py
│       └── config.py
│
├── ml/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       └── evaluation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_ml.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── saved/
│
├── requirements.txt
├── README.md
└── Dockerfile
"""

# File: requirements.txt
"""
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
python-dotenv==1.0.0
pydantic==2.5.2
joblib==1.3.2
"""

# File: app/core/config.py
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Adenoiditis ML API"
    MODEL_PATH: str = "models/saved/random_forest_model.joblib"
    
    class Config:
        case_sensitive = True

settings = Settings()
"""

# File: app/api/models.py
"""
from pydantic import BaseModel
from typing import List, Optional

class PredictionInput(BaseModel):
    age: int
    gender: str
    snoring_frequency: float
    sleep_difficulty: float
    breathing_difficulty: float
    mouth_breathing: float
    sore_throat_frequency: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 7,
                "gender": "M",
                "snoring_frequency": 8.5,
                "sleep_difficulty": 7.0,
                "breathing_difficulty": 6.5,
                "mouth_breathing": 8.0,
                "sore_throat_frequency": 5.5
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str
"""

# File: app/api/endpoints.py
"""
from fastapi import APIRouter, HTTPException
from app.api.models import PredictionInput, PredictionOutput
from ml.models.predict import predict_adenoiditis
from typing import List

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        prediction, probability = predict_adenoiditis(input_data.dict())
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        
        return PredictionOutput(
            prediction=prediction,
            probability=float(probability),
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
"""

# File: app/main.py
"""
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(router, prefix=settings.API_V1_STR)
"""

# File: ml/data/preprocessing.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Create copy of dataframe
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['gender']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Split features and target
        X = df_processed.drop(['adenoiditis_diagnosis'], axis=1)
        y = df_processed['adenoiditis_diagnosis']
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X_scaled, y
    
    def transform(self, data: Dict) -> pd.DataFrame:
        # Convert single prediction input to DataFrame
        df = pd.DataFrame([data])
        
        # Transform categorical variables
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col].astype(str))
        
        # Scale features
        return pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )
"""

# File: ml/models/train.py
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data.preprocessing import DataPreprocessor
from ml.utils.evaluation import evaluate_model

def train_model(data_path: str, model_path: str) -> None:
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model and preprocessor
    joblib.dump(model, f"{model_path}/model.joblib")
    joblib.dump(preprocessor, f"{model_path}/preprocessor.joblib")
"""

# File: ml/models/predict.py
"""
import joblib
import pandas as pd
from typing import Tuple, Dict

def load_model(model_path: str):
    model = joblib.load(f"{model_path}/model.joblib")
    preprocessor = joblib.load(f"{model_path}/preprocessor.joblib")
    return model, preprocessor

def predict_adenoiditis(input_data: Dict) -> Tuple[int, float]:
    # Load model and preprocessor
    model, preprocessor = load_model("models/saved")
    
    # Preprocess input
    X = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability
"""

# File: ml/utils/evaluation.py
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    
    # Feature importance plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        features = X_test.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title("Feature Importances")
        plt.bar(range(X_test.shape[1]), importances[indices])
        plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('models/saved/feature_importance.png')
        plt.close()
"""

# File: Dockerfile
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# File: README.md
"""
# Adenoiditis ML Project

Machine learning project for adenoiditis diagnosis prediction with REST API.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train model:
```bash
python -m ml.models.train
```

4. Run API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/api/v1/predict`: Make adenoiditis prediction
- GET `/api/v1/health`: Health check

## Docker

Build and run with Docker:

```bash
docker build -t adenoiditis-ml .
docker run -p 8000:8000 adenoiditis-ml
```

## Project Structure

- `app/`: FastAPI application
- `ml/`: Machine learning code
- `tests/`: Unit tests
- `data/`: Data files
- `models/`: Saved models
"""
