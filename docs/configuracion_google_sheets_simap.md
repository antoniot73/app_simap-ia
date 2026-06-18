# Configuración de Google Sheets para SIMAP-IA

## Objetivo

Registrar automáticamente las pruebas e interacciones de la aplicación para el creador de SIMAP-IA.

La app registra:

- `registro_pruebas`: cada predicción calculada, incluyendo presets y pruebas particulares.
- `registro_interacciones`: eventos de uso como inicio de sesión, descarga de PDF, descarga de CSV de respaldo, apertura de imagen y generación del reporte de presets.

## Flujo

Usuario calcula riesgo en SIMAP-IA → se genera `test_id` y `session_id` → la app muestra la predicción → la app guarda una fila en Google Sheets.

## Pasos

1. Crear un Google Sheet.
2. Crear o reutilizar una cuenta de servicio de Google Cloud.
3. Compartir el Google Sheet con el `client_email` de la cuenta de servicio como editor.
4. Copiar `.streamlit/secrets.example.toml` a `.streamlit/secrets.toml`.
5. Pegar el ID del Google Sheet y las credenciales reales.
6. Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

7. Ejecutar:

```powershell
streamlit run app.py
```

## Pruebas rápidas

```powershell
python -m py_compile app.py
python -c "import gspread; import google.oauth2.service_account; print('Google Sheets deps OK')"
python -c "import app; print('app.py importado correctamente')"
streamlit run app.py
```

## Seguridad

No subir `.streamlit/secrets.toml` a GitHub.
El archivo incluido es solo `secrets.example.toml` y no contiene credenciales reales.
