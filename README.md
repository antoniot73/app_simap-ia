# SIMAP-IA Web Public

Aplicación web pública de inferencia para clasificación de riesgo de falla.

## Importante

Este repositorio no contiene scripts de entrenamiento. Solo contiene:
- app.py
- src/
- artifacts/
- requirements.txt
- docs/

## Ejecución local

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

Sube solo esta carpeta a GitHub y conéctala a Streamlit Community Cloud.

Archivo principal:
```text
app.py
```
