import pandas as pd
import numpy as np

def generate_adenoiditis_dataset(n_samples=1000):
    """
    Gerar conjunto de dados sintético para diagnóstico de adenoidite
   
    Args:
        n_samples (int): Número de registos de amostra a gerar
    
    Returns:
        pd.DataFrame: Conjunto de dados médicos sintéticos
    """
    np.random.seed(42)  # Garantindo a reprodutibilidade
    
    data = {
        # INFORMAÇÕES DO PACIENTE
        # Dados Demográficos
        'id_paciente': range(1, n_samples + 1),
        'idade_mes': np.random.randint(0, 60, n_samples),
        'genero': np.random.choice(['Masculino', 'Feminino'], n_samples),
        
        # Sintomas (escala de 0 a 10)
        'severidade_obstrucao_nasal': np.random.uniform(0, 10, n_samples),
        'frequencia_respiracao_bucal': np.random.uniform(0, 10, n_samples),
        'intensidade_ressoar': np.random.uniform(0, 10, n_samples),
        'pontuacao_interrupcao_sono': np.random.uniform(0, 10, n_samples),
        
        # Parte do Exame Físico
        'tamanho_adenoidite_mm': np.random.uniform(5, 25, n_samples),
        'percentagem_obstrucao_v_areas': np.random.uniform(0, 100, n_samples),
        'tamanho_amigdala': np.random.randint(1, 4, n_samples),
        
        # Histório Médico
        'contagem_respiracoes_ano_passado': np.random.randint(0, 6, n_samples),
        'historico_alergia': np.random.choice([0, 1], n_samples),
        'problemas_adenoide_h_familiar': np.random.choice([0, 1], n_samples),
        
        # Indicadores de Diagnóstico
        'lateral_xray_adenoid_nasopharynx_ratio': np.random.uniform(0, 1, n_samples),
        'inflammatory_markers_level': np.random.uniform(0, 100, n_samples)
    }
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Adicionar rótulo/etiqueta de diagnóstico (lógica simplificada)
    df['adenoiditis_diagnosis'] = (
        (df['severidade_obstrucao_nasal'] > 7) & 
        (df['tamanho_adenoidite_mm'] > 15) & 
        (df['contagem_respiracoes_ano_passado'] > 3)
    ).astype(int)
    
    return df

# Gerar e guardar conjunto de dados
dataset = generate_adenoiditis_dataset()
dataset.to_csv('data/raw/adenoidite_dataset_simplificado.csv', index=False)
print(dataset.head())
print("\nEstatísticas do Dataset:")
print(dataset['adenoiditis_diagnosis'].value_counts(normalize=True))
