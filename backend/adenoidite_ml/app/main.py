from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import joblib
from io import StringIO

app = FastAPI(title="API de Diagnóstico de adenoidite")

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando os modelos pré-treinados
knn_model = joblib.load('models/knn_model.joblib')
kmeans_model = joblib.load('models/kmeans_model.joblib')

@app.post("/diagnostico")
async def diagnose_adenoiditis(file: UploadFile = File(...)):
    """
    Diagnosticar adenoidite com base no ficheiro de dados do paciente
    """
    try:
        # Lendo o conteúdo do ficheiro
        contents = await file.read()
        s = str(contents, 'utf-8')
        data = pd.read_csv(StringIO(s))
        
        # Pré-processar dados (semelhante ao pré-processamento de treino)
        # Nota: num cenário real, utilizaria exatamente as mesmas etapas de pré-processamento do treino
       
        # Previsão usando KNN
        knn_prediction = knn_model.predict(data)
        
        # Cluster usando o KMeans
        kmeans_clusters = kmeans_model.predict(data)
        
        return {
            "knn_diagnostico": knn_prediction.tolist(),
            "kmeans_clusters": kmeans_clusters.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)