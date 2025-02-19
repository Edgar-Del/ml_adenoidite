# Diagnóstico de Adenoidite com Machine Learning

## Visão Geral do Projecto
Este projecto desenvolve uma solução de ML para diagnosticar adenoidite em crianças de 0 a 5 anos, adaptada para o Hospital Pediátrico do Lubango.

## Componentes Principais
- API de Machine Learning baseada em Python utilizando FastAPI.
- Modelos de Machine Learning:
  - Classificação com K-Nearest Neighbors (KNN).
  - Agrupamento com K-Means Clustering.
- Frontend desenvolvido em Next.js para interação com o usuário.

## Instruções de Configuração

### Configuração do Backend
1. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Treinar o modelo:
```bash
python -m ml.models.train
```

4. Rodar a API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/api/v1/diagnosticar`: ENVIAR DADOS PARA TESTAR
- GET `/api/v1/health`: RECEBER O FEEDBACK "MÉDICO" DO SISTEMA

## Docker


```bash
docker build -t adenoiditis-ml .
docker run -p 8000:8000 adenoiditis-ml
```

## Esturuta do projecto

- `app/`: App FastAPI
- `ml/`: Código do Agente ML
- `tests/`: Testes Unitários
- `data/`: Ficheiros de dados
- `models/`: Modelos Salvos
```

### Configuração do Frontend
1. Instale as dependências:
```bash
npm install
npm run dev
```

## Preparação dos Dados
- Certifique-se de que os dados dos pacientes estejam no formato CSV (Por enquanto é esta a forma de funcionamento).
- Inclua características relevantes para o diagnóstico de adenoidite.
- As características pré-processadas incluem medições médicas, sintomas, entre outros.

## Treinamento do Modelo
- Use o endpoint `/train` para treinar novamente os modelos com novos dados.

## Processo de Diagnóstico
1. Faça o upload do arquivo CSV com dados do paciente.
2. A API realiza o pré-processamento dos dados.
3. O modelo KNN fornece a classificação do diagnóstico.
4. O modelo K-Means identifica possíveis clusters.
