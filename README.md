# Bank Account Fraud (BAF) - Equipo 5
Carlos Alberto Mentado Reyes - A01276065
Raymundo Iván Díaz Alejandre - A01735644
José Eduardo Puentes Martínez - A01733177
Fernanda Díaz Gutiérrez - A01639572

## Objetivo
Este proyecto resuelve el reto de detección de fraude bancario (Bank Account Fraud, NeurIPS 2022) aplicando la metodología CRISP-DM
El objetivo principal es maximizar el recall al 5% de FPR en un dataset altamente desbalanceado (aprox 0.8-1% fraudes), evaluando además la fairness (Predictive Equality) entre los grupos de edad

## Dataset
- **Fuente:** Bank Account Fraud (BAF), NeurIPS 2022  
- **Características:** 1 millón de registros, 32 columnas, son datos sintéticos pero realistas con sesgos controlados 
- **Variable objetivo:** fraud_bool (1 = fraude, 0 = no fraude) 
- **Split elegido:**  
  - Train: meses 0–5  
  - Val:  mes 6  
  - Test: mes 7  

## Pipeline
1. **Preprocesamiento sin leaks:**
   - Convertimos negativos codificados a NaN
   - Imputación: mediana en numéricas y moda en categóricas
   - Scaling y One-Hot Encoding
2. **Modelo:** LightGBM con peso de clase para balancear el desbalance
3. **Selección de umbral:**  
   - Umbral operativo fijado en validación para obtener FPR = 0.05 (exacto)
4. **Evaluación en test:**  
   - AUC, AP (PR-AUC), Recall@5%FPR, FPR 
   - Fairness: comparación de FPR en edad <50 vs >=50

## Resultados obtenidos
- **Validación (mes 6):**  
  - AUC = 0.890 
  - AP (PR-AUC) = 0.176 
  - Recall@5%FPR = 0.524 
  - Umbral operativo = 0.7683

- **Test (mes 7):**  
  - AUC = 0.890  
  - AP = 0.211 
  - FPR@test_thr = 0.0419
  - Recall@test_thr = 0.535
  - Recall@exact 5%FPR (curve-based) = 0.562

- **Fairness (Predictive Equality, edad):**  
  - FPR <50 = 0.0332
  - FPR >=50 = 0.1011
  - Ratio FPR = 0.33 (sesgo hacia grupo mayor, esperado en BAF)
  - El modelo genera más falsos positivos en mayores de 50, lo cual refleja el sesgo esperado en BAF

- Sin overfitting: métricas consistentes entre validación y test

## Para ejecutar

- Clonar el repositorio
- Se necesita Python 3.9+ y las librerías en requirements.txt: 
- Dependencias: pip install -r requirements.txt
- Ejecución: python baf_model.py --csv ./Datasets/Base.csv --out_dir ./outputs
- Salidas: 
    - metrics_baf_base.json / metrics_baf_base.csv → métricas 
    - val_roc_base.png → curva ROC en validación (marcando punto operativo)
    - test_fpr_by_age.png → barras de FPR por grupo de edad
- Correr usando conda (sugerencia):
    - conda activate baf
    - python implementacion/baf_model.py --csv Datasets/Base.csv --out_dir outputs

