# SIMAP-IA

**Sistema Inteligente de Mantenimiento Predictivo con Inteligencia Artificial**

SIMAP-IA es una aplicación web pública de inferencia para estimar el riesgo de falla de una máquina bajo condiciones operativas ingresadas por el usuario.

Esta versión corresponde a la **versión operativa de lanzamiento público**. La aplicación no entrena modelos en producción; únicamente carga artefactos previamente entrenados y exportados desde el entorno local privado del proyecto.

---

## Créditos del proyecto

**Autor:** Antonio Nicolás Toro González  
**Maestría:** Maestría en Inteligencia Artificial para la Transformación Digital  
**Landing Page:** https://skepsis-apps.github.io/landing_page/

---

## Resumen ejecutivo

SIMAP-IA permite simular condiciones operativas de una máquina de fresado o mecanizado rotativo y estimar la probabilidad de falla mediante un modelo de machine learning entrenado con el dataset AI4I 2020.

La aplicación pública integra:

- Simulador interactivo de riesgo.
- Pruebas preconfiguradas.
- Prueba personalizada.
- Clasificación de riesgo.
- Reportes PDF.
- Explorador del dataset.
- Visualizaciones descriptivas.
- Imagen conceptual del equipo.
- Registro automático privado en Google Sheets para el creador.
- Arquitectura separada entre entrenamiento privado e inferencia pública.

---

## Estado del proyecto

**Estado actual:** versión pública operativa de lanzamiento.

La aplicación está preparada para ser desplegada en:

```text
GitHub + Streamlit Community Cloud
```

El repositorio público contiene solo los elementos necesarios para inferencia, visualización, reporte y trazabilidad de uso. No contiene scripts de entrenamiento ni credenciales privadas.

---

## Alcance funcional

SIMAP-IA estima la probabilidad de que ocurra una falla de máquina bajo las condiciones operativas ingresadas.

El sistema entrega:

- Probabilidad estimada de falla.
- Nivel de riesgo.
- Recomendación operativa.
- Identificador único de prueba.
- Registro trazable de la simulación.
- Reporte PDF descargable.

---

## Límites del prototipo

SIMAP-IA **no debe interpretarse** como:

- Predicción exacta de fecha de falla.
- Cálculo de vida útil restante.
- Sistema industrial certificado.
- Sustituto de sensores reales.
- Sustituto de inspección técnica.
- Diagnóstico definitivo para operación industrial real.
- Sistema SCADA, CMMS o EAM productivo.

El prototipo es una herramienta académica y demostrativa de mantenimiento predictivo basado en inteligencia artificial.

---

## Dataset utilizado

**Dataset:** Predictive Maintenance Dataset - AI4I 2020  
**Autor del dataset:** Stephan Matzka  
**Fuente:** Kaggle  
**URL:** https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020  

Referencia académica sugerida:

```text
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"
2020 Third International Conference on Artificial Intelligence for Industries (AI4I),
2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.
```

El dataset AI4I 2020 es sintético y fue modelado a partir de un proceso de fresado o mecanizado. No corresponde a una máquina real específica identificable ni a datos históricos industriales de una empresa concreta.

---

## Características del dataset

El archivo usado por la aplicación pública es:

```text
data/ai4i2020.csv
```

Resumen del dataset:

| Indicador | Valor |
|---|---:|
| Registros | 10,000 |
| Columnas | 14 |
| Fallas | 339 |
| No fallas | 9,661 |
| Tasa de falla | 3.39% |
| Valores faltantes | 0 |
| Filas duplicadas | 0 |

---

## Equipo representado por el prototipo

La aplicación incluye una imagen conceptual de una máquina de fresado o mecanizado rotativo industrial.

La imagen representa de forma aproximada un proceso donde intervienen:

- Tipo de producto.
- Temperatura ambiente.
- Temperatura del proceso.
- Velocidad rotacional.
- Torque.
- Desgaste de herramienta.
- Condición de falla.

La imagen es referencial y no representa una máquina real específica del dataset.

---

## Variables de entrada del modelo

| Variable | Tipo | Descripción |
|---|---|---|
| `Type` | Categórica | Tipo o calidad del producto: L, M o H. |
| `Air temperature [K]` | Numérica | Temperatura ambiente del aire en Kelvin. |
| `Process temperature [K]` | Numérica | Temperatura del proceso en Kelvin. |
| `Rotational speed [rpm]` | Numérica | Velocidad rotacional en revoluciones por minuto. |
| `Torque [Nm]` | Numérica | Torque o carga mecánica aplicada. |
| `Tool wear [min]` | Numérica | Desgaste acumulado de la herramienta en minutos. |

---

## Variable objetivo

| Variable | Descripción |
|---|---|
| `Machine failure` | Indica si ocurrió una falla de máquina: 0 = no falla, 1 = falla. |

---

## Variables excluidas del modelo principal

Las siguientes columnas se excluyen como predictoras:

```text
TWF, HDF, PWF, OSF, RNF
```

Estas variables representan modos de falla específicos:

| Columna | Significado |
|---|---|
| `TWF` | Tool Wear Failure |
| `HDF` | Heat Dissipation Failure |
| `PWF` | Power Failure |
| `OSF` | Overstrain Failure |
| `RNF` | Random Failure |

Se muestran únicamente en el explorador del dataset. No se usan como entradas del modelo principal para evitar fuga de información, ya que son variables demasiado cercanas a la variable objetivo.

También se excluyen:

```text
UDI
Product ID
```

por ser identificadores sin valor predictivo operativo generalizable para el simulador público.

---

## Ingeniería de variables

El pipeline genera internamente variables derivadas para capturar relaciones físicas y operativas:

| Variable derivada | Fórmula | Sentido técnico |
|---|---|---|
| `Thermal delta [K]` | `Process temperature [K] - Air temperature [K]` | Diferencia térmica entre proceso y ambiente. |
| `Approx mechanical power` | `Torque [Nm] * Rotational speed [rpm]` | Aproximación de carga mecánica operativa. |
| `Torque wear load` | `Torque [Nm] * Tool wear [min]` | Carga acumulada por desgaste bajo torque. |
| `Torque speed ratio` | `Torque [Nm] / Rotational speed [rpm]` | Relación entre esfuerzo mecánico y velocidad. |
| `Tool wear normalized` | `Tool wear [min] / máximo aprendido` | Normalización del desgaste respecto al máximo aprendido en entrenamiento. |

---

## Rangos de entrada del simulador

Los rangos numéricos se limitan al dominio observado en el dataset AI4I 2020:

| Variable | Mínimo | Máximo |
|---|---:|---:|
| `Air temperature [K]` | 295.3 | 304.5 |
| `Process temperature [K]` | 305.7 | 313.8 |
| `Rotational speed [rpm]` | 1168 | 2886 |
| `Torque [Nm]` | 3.8 | 76.6 |
| `Tool wear [min]` | 0 | 253 |

Esta restricción reduce el riesgo de inferencias fuera del dominio de entrenamiento.

---

## Modelo desplegado

La app pública utiliza el artefacto:

```text
artifacts/simap_pipeline.joblib
```

Este archivo contiene el pipeline de inferencia necesario:

- Ingeniería de variables.
- Preprocesamiento.
- Codificación de variables categóricas.
- Escalamiento de variables numéricas.
- Modelo entrenado.

Artefactos complementarios:

```text
artifacts/model_metrics.json
artifacts/threshold_config.json
artifacts/feature_schema.json
```

---

## Métricas locales del modelo

Modelo seleccionado durante el entrenamiento privado:

```text
gradient_boosting
```

Métricas registradas:

| Métrica | Valor |
|---|---:|
| PR-AUC | 0.9028 |
| ROC-AUC | 0.9661 |
| Precision | 1.0000 |
| Recall | 0.8088 |
| F1 | 0.8943 |
| F2 | 0.8410 |

Matriz de confusión:

| Componente | Valor |
|---|---:|
| TN | 1932 |
| FP | 0 |
| FN | 13 |
| TP | 55 |

Estas métricas corresponden a la validación local del prototipo y no deben interpretarse como garantía de desempeño industrial en condiciones reales no observadas.

---

## Arquitectura del proyecto

El proyecto se separa en dos carpetas:

```text
SIMAP-IA/
├── simap-ia-local-private/
└── simap-ia-web-public/
```

---

## Carpeta privada/local

```text
simap-ia-local-private/
```

Contiene elementos de entrenamiento y construcción del modelo:

- Descarga del dataset desde Kaggle.
- Validación del dataset.
- Entrenamiento de modelos.
- Selección del modelo.
- Exportación de artefactos.
- Pruebas técnicas.
- Scripts privados.

Esta carpeta **no se sube a GitHub**.

---

## Carpeta pública/web

```text
simap-ia-web-public/
```

Contiene la aplicación lista para despliegue:

- App de Streamlit.
- Código mínimo de inferencia.
- Artefactos entrenados.
- Dataset de referencia.
- Imagen conceptual.
- Documentación.
- Archivo de dependencias.
- Ejemplo de secretos sin credenciales reales.

Esta es la carpeta que se sube a GitHub y se conecta con Streamlit Community Cloud.

---

## Estructura pública esperada

```text
simap-ia-web-public/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.example.toml
├── artifacts/
│   ├── simap_pipeline.joblib
│   ├── model_metrics.json
│   ├── threshold_config.json
│   └── feature_schema.json
├── assets/
│   └── IMAGEN_1.png
├── data/
│   └── ai4i2020.csv
├── docs/
│   ├── configuracion_google_sheets_simap.md
│   └── model_card.md
└── src/
    ├── __init__.py
    ├── features.py
    ├── prediction.py
    ├── risk_rules.py
    ├── ui_helpers.py
    └── validators.py
```

---

## Funcionalidades de la aplicación pública

### 1. Simulador de riesgo de falla

El simulador permite ingresar condiciones operativas y obtener:

- Probabilidad estimada de falla.
- Nivel de riesgo.
- Recomendación operativa.
- Identificador único de prueba.
- Registro automático de la simulación.

### 2. Pruebas configuradas

La app incluye pruebas de referencia:

```text
Prueba 1 - Operación estable
Prueba 2 - Carga intermedia
Prueba 3 - Riesgo crítico
Prueba personalizada
```

### 3. Reporte PDF individual

Cada simulación puede generar un reporte PDF con:

- Identificador de prueba.
- Fecha y hora UTC.
- Variables ingresadas.
- Probabilidad estimada.
- Nivel de riesgo.
- Recomendación.
- Créditos del dataset.

### 4. Reporte PDF de pruebas configuradas

La app permite generar un reporte consolidado de las pruebas predefinidas.

### 5. Historial temporal de sesión

Durante la sesión, la app mantiene un historial temporal de pruebas ejecutadas.

### 6. Registro automático en Google Sheets

SIMAP-IA registra automáticamente pruebas e interacciones para el creador de la app.

Este registro no requiere acción del usuario final y funciona como bitácora privada de trazabilidad.

Pestañas esperadas:

```text
registro_pruebas
registro_interacciones
```

### 7. Explorador del dataset

La app incluye un explorador del dataset AI4I 2020 con:

- Resumen general.
- Diccionario de variables.
- Datos filtrables.
- Descarga de CSV filtrado.
- Visualizaciones.
- Modos de falla.

### 8. Visualizaciones

La sección de gráficos incluye:

- Distribución por tipo de producto.
- Distribución de la variable objetivo.
- Conteo por modo de falla.
- Promedios operativos por clase de falla.
- Relación torque vs velocidad rotacional.

El gráfico torque vs velocidad utiliza visualización robusta para mostrar puntos de falla y no falla.

### 9. Imagen conceptual

La app incluye un botón para mostrar u ocultar la imagen conceptual del equipo representado.

### 10. Créditos del dataset

La app muestra los créditos del dataset y la referencia académica correspondiente.

---

## Flujo interno de inferencia

El flujo funcional de SIMAP-IA es:

```text
Entrada del usuario
→ Validación de datos
→ Construcción del registro
→ Ingeniería de variables
→ Preprocesamiento
→ Inferencia con pipeline entrenado
→ Clasificación de riesgo
→ Visualización del resultado
→ Registro automático
→ Reporte PDF opcional
```

---

## Clasificación de riesgo

La probabilidad estimada se transforma en un nivel de riesgo mediante reglas definidas en:

```text
src/risk_rules.py
```

Los niveles usados por la aplicación son:

```text
Bajo
Medio
Alto
Crítico
```

Cada nivel se acompaña de una recomendación operativa.

---

## Ejemplo funcional

Caso de riesgo crítico:

```text
Tipo de producto: L
Temperatura del aire [K]: 302.0
Temperatura del proceso [K]: 313.0
Velocidad rotacional [rpm]: 1200
Torque [Nm]: 65
Desgaste de herramienta [min]: 230
```

Resultado esperado aproximado:

```text
Probabilidad estimada de falla: 98.82%
Nivel de riesgo: Crítico
Recomendación: Realizar inspección inmediata antes de continuar operación.
```

---

## Instalación local

Entrar a la carpeta pública:

```powershell
cd "D:\DISCO C\Antonio Toro\Proyectos_IA\Proyecto_SIMAP-IA\SIMAP-IA\simap-ia-web-public"
```

Crear entorno virtual:

```powershell
python -m venv .venv
```

Activar entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Actualizar pip:

```powershell
python -m pip install --upgrade pip
```

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

Validar sintaxis:

```powershell
python -m py_compile app.py
```

Ejecutar aplicación:

```powershell
python -m streamlit run app.py
```

---

## Dependencias principales

Las dependencias se declaran en:

```text
requirements.txt
```

La aplicación utiliza principalmente:

- Streamlit.
- pandas.
- scikit-learn.
- joblib.
- Altair.
- ReportLab.
- gspread.
- google-auth.

---

## Configuración de Google Sheets

La configuración local se realiza mediante:

```text
.streamlit/secrets.toml
```

Este archivo **no se sube a GitHub**.

Ejemplo seguro incluido en el repositorio:

```text
.streamlit/secrets.example.toml
```

Estructura esperada:

```toml
SIMAP_PERSISTENCE_BACKEND = "google_sheets"
GOOGLE_SHEETS_ENABLED = true
GOOGLE_SHEET_ID = "ID_DE_LA_HOJA"
GOOGLE_PREDICTIONS_SHEET = "registro_pruebas"
GOOGLE_EVENTS_SHEET = "registro_interacciones"
APP_VERSION = "SIMAP-IA-v1.0"

[GOOGLE_SERVICE_ACCOUNT]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "cuenta-servicio@proyecto.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

La hoja de Google Sheets debe estar compartida con el correo `client_email` de la cuenta de servicio como **Editor**.

---

## Seguridad de secretos

Reglas obligatorias:

- No subir `.streamlit/secrets.toml`.
- No subir archivos JSON de Google Cloud.
- No subir llaves privadas.
- No subir `.env`.
- No subir `.venv`.
- No subir carpetas de registros locales.
- No subir scripts de configuración de secretos.
- Revocar cualquier clave que haya sido expuesta accidentalmente.

---

## Despliegue en Streamlit Community Cloud

Flujo de despliegue:

```text
GitHub → Streamlit Community Cloud → App pública
```

Pasos:

1. Subir solo la carpeta pública `simap-ia-web-public` al repositorio.
2. Verificar que `.streamlit/secrets.toml` no esté en GitHub.
3. Verificar que no existan JSON ni llaves privadas en el repositorio.
4. Crear la app en Streamlit Community Cloud.
5. Seleccionar el repositorio.
6. Seleccionar la rama `main`.
7. Usar como archivo principal:

```text
app.py
```

8. Pegar los secretos reales en la sección **Secrets** de Streamlit Cloud.
9. Desplegar.
10. Probar simulador, reportes, explorador, gráficos y Google Sheets.

---

## Archivos permitidos en GitHub

```text
.gitignore
README.md
requirements.txt
app.py
.streamlit/secrets.example.toml
artifacts/simap_pipeline.joblib
artifacts/model_metrics.json
artifacts/threshold_config.json
artifacts/feature_schema.json
assets/IMAGEN_1.png
data/ai4i2020.csv
docs/configuracion_google_sheets_simap.md
docs/model_card.md
src/__init__.py
src/features.py
src/prediction.py
src/risk_rules.py
src/ui_helpers.py
src/validators.py
```

---

## Archivos prohibidos en GitHub

```text
simap-ia-local-private/
training/
data_ingestion/
tests/
notebooks/
.venv/
venv/
env/
registros/
logs/
.streamlit/secrets.toml
.streamlit/secrets_danado_backup.toml
*.json de credenciales
*.pem
*.key
*.p12
.env
.env.*
setup_google_sheets_secrets.py
test_google_sheets_direct.py
reset_secrets_simap.py
patch_grafico_torque_velocidad.py
```

---

## Validaciones antes del despliegue

Validar dataset:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/ai4i2020.csv'); print(df.shape); print(df['Machine failure'].sum()); print(df.isna().sum().sum())"
```

Resultado esperado:

```text
(10000, 14)
339
0
```

Validar sintaxis:

```powershell
python -m py_compile app.py
```

Validar importación:

```powershell
python -c "import app; print('app.py importado correctamente')"
```

Validar dependencias Google Sheets:

```powershell
python -c "import gspread; from google.oauth2.service_account import Credentials; print('Google Sheets deps OK')"
```

Ejecutar app:

```powershell
python -m streamlit run app.py
```

---

## Checklist antes de hacer push

Ejecutar:

```powershell
git status
```

No deben aparecer:

```text
.streamlit/secrets.toml
.streamlit/secrets_danado_backup.toml
*.json de credenciales
.venv/
registros/
setup_google_sheets_secrets.py
test_google_sheets_direct.py
reset_secrets_simap.py
```

Ejecutar revisión adicional:

```powershell
git diff --cached --name-only | Select-String -Pattern "secrets_danado|\.streamlit/secrets.toml|service_account|credential|\.env|\.pem|\.key|\.p12|registros|setup_google|test_google|reset_secrets"
```

Si no devuelve nada, el commit público es seguro.

---

## Comandos Git usados para publicación

Desde la carpeta pública:

```powershell
git init
git add .
git status
git commit -m "Lanzamiento publico SIMAP-IA"
git branch -M main
git remote set-url origin https://github.com/antoniot73/app_simap-ia.git
git push -u origin main
```

También puede usarse remoto SSH si la llave está configurada:

```powershell
git remote set-url origin git@github.com:antoniot73/app_simap-ia.git
git push -u origin main
```

---

## Repositorio público

Repositorio previsto:

```text
https://github.com/antoniot73/app_simap-ia
```

---

## Mantenimiento

Para actualizar la app después de cambios:

```powershell
git status
git add .
git commit -m "Actualizacion SIMAP-IA"
git push
```

Para ver el remoto activo:

```powershell
git remote -v
```

Para comprobar la rama:

```powershell
git branch
```

---

## Buenas prácticas de mantenimiento

- Mantener entrenamiento y despliegue separados.
- No entrenar modelos dentro de Streamlit Cloud.
- Versionar artefactos exportados.
- Actualizar `model_card.md` si cambia el modelo.
- Actualizar este README si cambia la funcionalidad pública.
- Revisar `.gitignore` antes de cada publicación.
- Validar Google Sheets después de cada despliegue.
- Probar PDF y explorador del dataset antes de publicar.

---

## Contacto

**Autor:** Antonio Nicolás Toro González  
**Landing Page:** https://skepsis-apps.github.io/landing_page/

---

## Licencia y uso

Este proyecto se presenta como prototipo académico-operativo para fines de aprendizaje, demostración y comunicación técnica en mantenimiento predictivo con inteligencia artificial.

Para uso industrial real se requiere:

- Recolección de datos propios.
- Validación con sensores reales.
- Calibración por equipo.
- Pruebas de robustez.
- Evaluación de sesgo operativo.
- Supervisión experta.
- Integración con procedimientos de mantenimiento aprobados.
