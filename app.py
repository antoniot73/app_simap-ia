"""
Aplicación web de inferencia para SIMAP-IA.

La app no entrena modelos. Solo carga artefactos entrenados localmente.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4
import json
import logging
import os

import pandas as pd
import streamlit as st

from src.prediction import SIMAPPredictor
from src.risk_rules import SIMAPRiskClassifier
from src.ui_helpers import format_probability, risk_badge
from src.validators import SIMAPInputValidator


LOGGER = logging.getLogger("simap_ia_app")


ARTIFACTS_DIR = Path("artifacts")
PIPELINE_PATH = ARTIFACTS_DIR / "simap_pipeline.joblib"
THRESHOLDS_PATH = ARTIFACTS_DIR / "threshold_config.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

ASSETS_DIR = Path("assets")
CONCEPTUAL_IMAGE_PATH = ASSETS_DIR / "IMAGEN_1.png"

DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "ai4i2020.csv"
DATASET_SOURCE_URL = (
    "https://www.kaggle.com/datasets/"
    "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
)
DATASET_CITATION = (
    'S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance '
    'Applications," 2020 Third International Conference on Artificial Intelligence '
    'for Industries (AI4I), 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.'
)

MODEL_INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_DATASET_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_MODE_LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

APP_NAME = "SIMAP-IA"
APP_VERSION = "1.0.0"
PERSISTENCE_DIR = Path("registros")
LOCAL_PREDICTIONS_LOG_PATH = PERSISTENCE_DIR / "registro_pruebas.csv"
LOCAL_EVENTS_LOG_PATH = PERSISTENCE_DIR / "registro_interacciones.csv"

PREDICTION_LOG_COLUMNS = [
    "timestamp_utc",
    "event_type",
    "test_id",
    "session_id",
    "case_name",
    "case_kind",
    "is_preset",
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "failure_probability",
    "failure_probability_pct",
    "risk_level",
    "recommendation",
    "dataset_name",
    "dataset_url",
    "app_name",
    "app_version",
]

EVENT_LOG_COLUMNS = [
    "timestamp_utc",
    "event_type",
    "session_id",
    "test_id",
    "case_name",
    "app_name",
    "app_version",
    "metadata_json",
]

DATASET_REFERENCE_RANGES = {
    "Air temperature [K]": {
        "min": 295.3,
        "max": 304.5,
        "default": 298.0,
        "step": 0.1,
        "description": "Rango observado en AI4I 2020 para temperatura ambiente.",
    },
    "Process temperature [K]": {
        "min": 305.7,
        "max": 313.8,
        "default": 308.0,
        "step": 0.1,
        "description": "Rango observado en AI4I 2020 para temperatura de proceso.",
    },
    "Rotational speed [rpm]": {
        "min": 1168.0,
        "max": 2886.0,
        "default": 1500.0,
        "step": 1.0,
        "description": "Rango observado en AI4I 2020 para velocidad rotacional.",
    },
    "Torque [Nm]": {
        "min": 3.8,
        "max": 76.6,
        "default": 40.0,
        "step": 0.1,
        "description": "Rango observado en AI4I 2020 para torque.",
    },
    "Tool wear [min]": {
        "min": 0.0,
        "max": 253.0,
        "default": 100.0,
        "step": 1.0,
        "description": "Rango observado en AI4I 2020 para desgaste de herramienta.",
    },
}


CUSTOM_PRESET_NAME = "Prueba personalizada"

DEFAULT_CUSTOM_RECORD = {
    "Type": "L",
    "Air temperature [K]": 298.0,
    "Process temperature [K]": 308.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100.0,
}

PRESET_TESTS = {
    CUSTOM_PRESET_NAME: DEFAULT_CUSTOM_RECORD,
    "Prueba 1 - Operación estable": {
        "Type": "M",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 35.0,
        "Tool wear [min]": 20.0,
    },
    "Prueba 2 - Carga intermedia": {
        "Type": "L",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 309.0,
        "Rotational speed [rpm]": 1350.0,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 190.0,
    },
    "Prueba 3 - Riesgo crítico": {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 313.0,
        "Rotational speed [rpm]": 1200.0,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 230.0,
    },
}

CONFIGURED_PRESET_NAMES = [
    "Prueba 1 - Operación estable",
    "Prueba 2 - Carga intermedia",
    "Prueba 3 - Riesgo crítico",
]

INPUT_WIDGET_KEYS = {
    "Type": "input_machine_type",
    "Air temperature [K]": "input_air_temperature",
    "Process temperature [K]": "input_process_temperature",
    "Rotational speed [rpm]": "input_rotational_speed",
    "Torque [Nm]": "input_torque",
    "Tool wear [min]": "input_tool_wear",
}


@st.cache_resource
def load_predictor() -> SIMAPPredictor:
    """Carga el predictor entrenado localmente."""
    return SIMAPPredictor(PIPELINE_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    """Carga archivos JSON de configuración o métricas."""
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Carga el dataset AI4I 2020 desde un archivo CSV local.

    La app pública no descarga datos desde Kaggle en tiempo de ejecución.
    El archivo esperado es:
        simap-ia-web-public/data/ai4i2020.csv

    Args:
        path: Ruta local del dataset.

    Returns:
        pd.DataFrame: Dataset cargado o DataFrame vacío si no existe.
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()




def generate_session_identifier(prefix: str = "SIMAP-SESSION") -> str:
    """
    Genera un identificador anónimo de sesión para trazabilidad de uso.

    Args:
        prefix: Prefijo del identificador.

    Returns:
        str: Identificador anónimo de sesión.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_token = uuid4().hex[:10].upper()
    return f"{prefix}-{timestamp}-{random_token}"


def get_current_session_id() -> str:
    """
    Obtiene el identificador anónimo de la sesión actual.

    Returns:
        str: Identificador de sesión.
    """
    return str(st.session_state.get("session_id", ""))


def coerce_bool(value: object, default: bool = False) -> bool:
    """
    Convierte valores comunes de configuración a booleano.

    Args:
        value: Valor leído desde secretos o variables de entorno.
        default: Valor por defecto si no se puede interpretar.

    Returns:
        bool: Valor interpretado.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "si", "sí", "enabled", "google_sheets"}:
        return True

    if normalized in {"0", "false", "no", "n", "disabled", "local", ""}:
        return False

    return default


def get_config_value(key: str, default: object = "") -> object:
    """
    Lee una configuración desde st.secrets o variables de entorno.

    Args:
        key: Nombre de la configuración.
        default: Valor por defecto.

    Returns:
        object: Valor encontrado o default.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


def get_nested_secret(section: str, default: object = None) -> object:
    """
    Lee una sección completa desde st.secrets.

    Args:
        section: Nombre de la sección.
        default: Valor por defecto.

    Returns:
        object: Sección de secretos o default.
    """
    try:
        if section in st.secrets:
            return st.secrets[section]
    except Exception:
        pass

    return default


def normalize_service_account_info(service_account_data: object) -> dict:
    """
    Normaliza credenciales de cuenta de servicio para google-auth.

    Args:
        service_account_data: Diccionario, AttrDict de Streamlit o JSON string.

    Returns:
        dict: Credenciales normalizadas.

    Raises:
        ValueError: Si el formato no es válido.
    """
    if service_account_data is None:
        raise ValueError("No se encontró la sección GOOGLE_SERVICE_ACCOUNT.")

    if isinstance(service_account_data, str):
        try:
            info = json.loads(service_account_data)
        except json.JSONDecodeError as error:
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON no contiene JSON válido.") from error
    else:
        info = dict(service_account_data)

    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = info["private_key"].replace("\\n", "\n")

    required_keys = {"client_email", "private_key", "token_uri"}
    missing_keys = sorted(required_keys - set(info.keys()))

    if missing_keys:
        raise ValueError(
            "Credenciales de Google incompletas. Faltan: "
            + ", ".join(missing_keys)
        )

    return info


def get_persistence_config() -> dict:
    """
    Construye la configuración de persistencia local y Google Sheets.

    Returns:
        dict: Configuración normalizada.
    """
    backend = str(get_config_value("SIMAP_PERSISTENCE_BACKEND", "local")).strip().lower()
    google_enabled = coerce_bool(get_config_value("GOOGLE_SHEETS_ENABLED", False))
    google_enabled = google_enabled or backend == "google_sheets"

    return {
        "backend": backend,
        "google_sheets_enabled": google_enabled,
        "sheet_id": str(
            get_config_value(
                "GOOGLE_SHEET_ID",
                get_config_value("SIMAP_GOOGLE_SHEET_ID", ""),
            )
        ).strip(),
        "predictions_sheet": str(
            get_config_value("GOOGLE_PREDICTIONS_SHEET", "registro_pruebas")
        ).strip(),
        "events_sheet": str(
            get_config_value("GOOGLE_EVENTS_SHEET", "registro_interacciones")
        ).strip(),
        "credentials_path": str(
            get_config_value(
                "GOOGLE_CREDENTIALS_PATH",
                get_config_value("SIMAP_GOOGLE_CREDENTIALS_PATH", ""),
            )
        ).strip(),
        "app_version": str(get_config_value("APP_VERSION", APP_VERSION)).strip(),
    }


def is_google_sheets_ready(config: dict) -> bool:
    """
    Evalúa si Google Sheets está habilitado y tiene configuración mínima.

    Args:
        config: Configuración de persistencia.

    Returns:
        bool: True si puede intentar conexión con Google Sheets.
    """
    if not bool(config.get("google_sheets_enabled", False)):
        return False

    if not str(config.get("sheet_id", "")).strip():
        return False

    return True


@st.cache_resource
def load_google_sheets_client() -> object:
    """
    Carga cliente de Google Sheets mediante cuenta de servicio.

    Returns:
        object: Cliente gspread autenticado.

    Raises:
        RuntimeError: Si falta alguna dependencia o credencial.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as error:
        raise RuntimeError(
            "Faltan dependencias para Google Sheets. "
            "Instala: python -m pip install gspread google-auth"
        ) from error

    config = get_persistence_config()
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    credentials_path = str(config.get("credentials_path", "")).strip()

    if credentials_path:
        credential_file = Path(credentials_path)
        if not credential_file.exists():
            raise RuntimeError(f"No existe el archivo de credenciales: {credential_file}")

        credentials = Credentials.from_service_account_file(
            str(credential_file),
            scopes=scopes,
        )
    else:
        service_account_data = get_nested_secret("GOOGLE_SERVICE_ACCOUNT")
        if service_account_data is None:
            service_account_data = get_config_value("GOOGLE_SERVICE_ACCOUNT_JSON", "")

        service_account_info = normalize_service_account_info(service_account_data)
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=scopes,
        )

    return gspread.authorize(credentials)


def get_or_create_worksheet(
    client: object,
    sheet_id: str,
    worksheet_name: str,
    headers: list[str],
) -> object:
    """
    Obtiene o crea una pestaña en Google Sheets y asegura encabezados.

    Args:
        client: Cliente gspread autenticado.
        sheet_id: ID del Google Sheet.
        worksheet_name: Nombre de la pestaña.
        headers: Encabezados esperados.

    Returns:
        object: Worksheet de gspread.
    """
    spreadsheet = client.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except Exception:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=1000,
            cols=max(len(headers), 1),
        )

    current_headers = worksheet.row_values(1)

    if not current_headers:
        worksheet.append_row(headers)
    elif current_headers[: len(headers)] != headers:
        worksheet.insert_row(headers, index=1)

    return worksheet


def append_row_to_local_csv(
    row: dict,
    headers: list[str],
    output_path: Path,
) -> bool:
    """
    Agrega una fila a un CSV local de respaldo.

    Args:
        row: Registro plano.
        headers: Encabezados esperados.
        output_path: Ruta de salida.

    Returns:
        bool: True si el guardado local fue correcto.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_df = pd.DataFrame([{header: row.get(header, "") for header in headers}])
        file_exists = output_path.exists()
        row_df.to_csv(
            output_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )
        return True
    except Exception as error:
        LOGGER.warning("No fue posible guardar registro local: %s", error)
        return False


def append_row_to_google_sheets(
    row: dict,
    headers: list[str],
    worksheet_name: str,
) -> tuple[bool, str]:
    """
    Agrega una fila a una pestaña de Google Sheets.

    Args:
        row: Registro plano.
        headers: Encabezados esperados.
        worksheet_name: Pestaña de destino.

    Returns:
        tuple[bool, str]: Estado y mensaje.
    """
    config = get_persistence_config()

    if not is_google_sheets_ready(config):
        return False, "Google Sheets no está configurado."

    try:
        client = load_google_sheets_client()
        worksheet = get_or_create_worksheet(
            client=client,
            sheet_id=str(config["sheet_id"]),
            worksheet_name=worksheet_name,
            headers=headers,
        )
        ordered_row = [row.get(header, "") for header in headers]
        worksheet.append_row(ordered_row, value_input_option="USER_ENTERED")
        return True, f"Registro guardado en Google Sheets: {worksheet_name}."
    except Exception as error:
        LOGGER.warning("No fue posible guardar en Google Sheets: %s", error)
        return False, f"No fue posible guardar en Google Sheets: {error}"


def append_row_persistent(
    row: dict,
    headers: list[str],
    local_path: Path,
    worksheet_name: str,
) -> dict:
    """
    Guarda una fila en CSV local y, si está configurado, en Google Sheets.

    Args:
        row: Registro plano.
        headers: Encabezados esperados.
        local_path: Ruta de respaldo local.
        worksheet_name: Pestaña de Google Sheets.

    Returns:
        dict: Estado de persistencia.
    """
    local_ok = append_row_to_local_csv(row, headers, local_path)
    google_ok, google_message = append_row_to_google_sheets(
        row=row,
        headers=headers,
        worksheet_name=worksheet_name,
    )

    status = {
        "local": local_ok,
        "google_sheets": google_ok,
        "message": google_message,
    }
    st.session_state["last_persistence_status"] = status
    return status


def build_prediction_log_row(result: dict, event_type: str = "prediction_calculated") -> dict:
    """
    Construye una fila plana de predicción para Google Sheets.

    Args:
        result: Resultado de predicción.
        event_type: Tipo de evento.

    Returns:
        dict: Fila plana.
    """
    record = result["record"]
    config = get_persistence_config()

    return {
        "timestamp_utc": result.get("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "event_type": event_type,
        "test_id": result.get("test_id", ""),
        "session_id": result.get("session_id", get_current_session_id()),
        "case_name": result.get("case_name", ""),
        "case_kind": result.get("case_kind", ""),
        "is_preset": str(result.get("case_kind", "") == "preset_configurado"),
        "type": record.get("Type", ""),
        "air_temperature_k": record.get("Air temperature [K]", ""),
        "process_temperature_k": record.get("Process temperature [K]", ""),
        "rotational_speed_rpm": record.get("Rotational speed [rpm]", ""),
        "torque_nm": record.get("Torque [Nm]", ""),
        "tool_wear_min": record.get("Tool wear [min]", ""),
        "failure_probability": round(float(result.get("probability", 0.0)), 8),
        "failure_probability_pct": round(float(result.get("probability_percent", 0.0)), 4),
        "risk_level": result.get("risk_level", ""),
        "recommendation": result.get("recommendation", ""),
        "dataset_name": result.get("dataset", "Predictive Maintenance Dataset (AI4I 2020)"),
        "dataset_url": result.get("dataset_url", DATASET_SOURCE_URL),
        "app_name": APP_NAME,
        "app_version": str(config.get("app_version", APP_VERSION)),
    }


def build_interaction_log_row(
    event_type: str,
    metadata: dict | None = None,
    test_id: str = "",
    case_name: str = "",
) -> dict:
    """
    Construye una fila de evento de interacción.

    Args:
        event_type: Tipo de evento.
        metadata: Metadatos no sensibles.
        test_id: Identificador de prueba relacionado.
        case_name: Nombre de caso relacionado.

    Returns:
        dict: Fila plana de evento.
    """
    config = get_persistence_config()
    safe_metadata = metadata or {}
    metadata_json = json.dumps(safe_metadata, ensure_ascii=False, default=str)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event_type": event_type,
        "session_id": get_current_session_id(),
        "test_id": test_id,
        "case_name": case_name,
        "app_name": APP_NAME,
        "app_version": str(config.get("app_version", APP_VERSION)),
        "metadata_json": metadata_json[:1500],
    }


def register_prediction_result(
    result: dict,
    event_type: str = "prediction_calculated",
) -> dict:
    """
    Registra automáticamente una predicción para el creador de la app.

    Args:
        result: Resultado de predicción.
        event_type: Tipo de evento de predicción.

    Returns:
        dict: Estado de persistencia.
    """
    config = get_persistence_config()
    row = build_prediction_log_row(result, event_type=event_type)
    status = append_row_persistent(
        row=row,
        headers=PREDICTION_LOG_COLUMNS,
        local_path=LOCAL_PREDICTIONS_LOG_PATH,
        worksheet_name=str(config["predictions_sheet"]),
    )

    register_interaction_event(
        event_type=event_type,
        metadata={
            "case_kind": result.get("case_kind", ""),
            "risk_level": result.get("risk_level", ""),
            "google_sheets_prediction_saved": status.get("google_sheets", False),
        },
        test_id=result.get("test_id", ""),
        case_name=result.get("case_name", ""),
    )

    return status


def register_interaction_event(
    event_type: str,
    metadata: dict | None = None,
    test_id: str = "",
    case_name: str = "",
) -> dict:
    """
    Registra una interacción de usuario en la bitácora del creador.

    Args:
        event_type: Tipo de evento.
        metadata: Metadatos no sensibles.
        test_id: Prueba relacionada.
        case_name: Caso relacionado.

    Returns:
        dict: Estado de persistencia.
    """
    config = get_persistence_config()
    row = build_interaction_log_row(
        event_type=event_type,
        metadata=metadata,
        test_id=test_id,
        case_name=case_name,
    )

    return append_row_persistent(
        row=row,
        headers=EVENT_LOG_COLUMNS,
        local_path=LOCAL_EVENTS_LOG_PATH,
        worksheet_name=str(config["events_sheet"]),
    )


def register_session_started_event() -> None:
    """
    Registra una sola vez el inicio anónimo de sesión.
    """
    if bool(st.session_state.get("session_started_registered", False)):
        return

    register_interaction_event(
        event_type="session_started",
        metadata={"component": "app", "dataset_available": DATASET_PATH.exists()},
    )
    st.session_state["session_started_registered"] = True


def render_persistence_status() -> None:
    """
    Muestra estado de la persistencia para el creador de la aplicación.
    """
    config = get_persistence_config()
    status = st.session_state.get("last_persistence_status", {})

    with st.expander("Registro automático del creador", expanded=False):
        st.markdown(
            """
            SIMAP-IA registra automáticamente las pruebas e interacciones usando
            identificadores anónimos. Este registro es para el creador de la
            aplicación y no requiere que el usuario descargue archivos.
            """
        )
        st.code(get_current_session_id(), language="text")

        if is_google_sheets_ready(config):
            if bool(status.get("google_sheets", False)):
                st.success("Google Sheets configurado y último registro enviado correctamente.")
            else:
                st.warning(
                    "Google Sheets está configurado, pero el último envío no fue confirmado. "
                    "Revisa permisos de la cuenta de servicio y el acceso al Google Sheet."
                )
        else:
            st.info(
                "Google Sheets no está configurado en este entorno. "
                "La app conserva respaldo local temporal en la carpeta registros/."
            )

        st.write(f"Pestaña de pruebas: `{config['predictions_sheet']}`")
        st.write(f"Pestaña de interacciones: `{config['events_sheet']}`")
        st.caption(str(status.get("message", "Sin registros ejecutados todavía.")))




def render_scope_notice() -> None:
    """Muestra aviso de alcance del prototipo."""
    st.info(
        "Este prototipo estima el riesgo de falla bajo condiciones operativas. "
        "No predice la fecha exacta de una falla, no calcula vida útil restante "
        "y no sustituye un sistema industrial real."
    )


def initialize_session_state() -> None:
    """
    Inicializa variables de estado usadas por la interfaz.

    Esta función evita depender de variables globales mutables y controla:
    - visibilidad de la imagen conceptual,
    - selección de presets,
    - valores de entrada del simulador,
    - resultados y reportes generados durante la sesión.
    """
    if "show_conceptual_equipment_image" not in st.session_state:
        st.session_state["show_conceptual_equipment_image"] = False

    if "selected_preset_name" not in st.session_state:
        st.session_state["selected_preset_name"] = CUSTOM_PRESET_NAME

    for column_name, widget_key in INPUT_WIDGET_KEYS.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = DEFAULT_CUSTOM_RECORD[column_name]

    if "last_prediction_result" not in st.session_state:
        st.session_state["last_prediction_result"] = None

    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    if "preset_report_results" not in st.session_state:
        st.session_state["preset_report_results"] = []

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = generate_session_identifier()

    if "session_started_registered" not in st.session_state:
        st.session_state["session_started_registered"] = False

    if "last_persistence_status" not in st.session_state:
        st.session_state["last_persistence_status"] = {
            "local": False,
            "google_sheets": False,
            "message": "Registro no ejecutado todavía.",
        }


def apply_selected_preset_to_inputs() -> None:
    """
    Copia los valores del preset seleccionado a los campos del simulador.
    """
    selected_name = st.session_state.get("selected_preset_name", CUSTOM_PRESET_NAME)
    selected_record = PRESET_TESTS.get(selected_name, DEFAULT_CUSTOM_RECORD)

    for column_name, widget_key in INPUT_WIDGET_KEYS.items():
        st.session_state[widget_key] = selected_record[column_name]


def build_record_from_session_state() -> dict:
    """
    Construye un registro de inferencia a partir de los widgets del simulador.

    Returns:
        dict: Registro compatible con el pipeline de predicción.
    """
    return {
        "Type": st.session_state[INPUT_WIDGET_KEYS["Type"]],
        "Air temperature [K]": float(st.session_state[INPUT_WIDGET_KEYS["Air temperature [K]"]]),
        "Process temperature [K]": float(st.session_state[INPUT_WIDGET_KEYS["Process temperature [K]"]]),
        "Rotational speed [rpm]": float(st.session_state[INPUT_WIDGET_KEYS["Rotational speed [rpm]"]]),
        "Torque [Nm]": float(st.session_state[INPUT_WIDGET_KEYS["Torque [Nm]"]]),
        "Tool wear [min]": float(st.session_state[INPUT_WIDGET_KEYS["Tool wear [min]"]]),
    }


def record_matches_preset(record: dict, preset_record: dict, tolerance: float = 1e-9) -> bool:
    """
    Verifica si un registro coincide con los valores de un preset.

    Args:
        record: Registro ingresado por el usuario.
        preset_record: Registro definido en PRESET_TESTS.
        tolerance: Tolerancia para comparar valores numéricos.

    Returns:
        bool: True si el registro coincide con el preset.
    """
    if record["Type"] != preset_record["Type"]:
        return False

    for column_name in NUMERIC_DATASET_COLUMNS:
        if abs(float(record[column_name]) - float(preset_record[column_name])) > tolerance:
            return False

    return True


def identify_case_metadata(record: dict) -> tuple[str, str]:
    """
    Identifica si el registro corresponde a un preset o a una prueba particular.

    Args:
        record: Registro ingresado por el usuario.

    Returns:
        tuple[str, str]: Nombre del caso y tipo de caso.
    """
    selected_name = st.session_state.get("selected_preset_name", CUSTOM_PRESET_NAME)

    if selected_name in CONFIGURED_PRESET_NAMES:
        selected_preset = PRESET_TESTS[selected_name]
        if record_matches_preset(record, selected_preset):
            return selected_name, "preset_configurado"

    for preset_name in CONFIGURED_PRESET_NAMES:
        if record_matches_preset(record, PRESET_TESTS[preset_name]):
            return preset_name, "preset_configurado"

    return "Prueba particular", "particular"



def toggle_conceptual_equipment_image() -> None:
    """
    Alterna la visibilidad de la imagen conceptual del equipo y registra el evento.

    El registro externo es opcional. Si Google Sheets no está configurado, la
    interacción continúa normalmente.
    """
    current_state = bool(st.session_state.get("show_conceptual_equipment_image", False))
    st.session_state["show_conceptual_equipment_image"] = not current_state

    event_type = "image_closed" if current_state else "image_opened"
    register_interaction_event(
        event_type=event_type,
        metadata={"component": "conceptual_equipment_image"},
    )


def render_conceptual_equipment_image_button() -> None:
    """
    Muestra un botón de un solo clic para desplegar u ocultar la imagen conceptual.

    La imagen debe guardarse en:
        simap-ia-web-public/assets/IMAGEN_1.png
    """
    st.markdown("### Equipo representado por el prototipo")
    st.caption(
        "Imagen referencial del proceso de fresado/mecanizado usado como "
        "representación conceptual del dataset AI4I 2020."
    )

    is_visible = bool(st.session_state.get("show_conceptual_equipment_image", False))
    button_label = (
        "Ocultar imagen conceptual del equipo"
        if is_visible
        else "Mostrar imagen conceptual del equipo"
    )

    st.button(
        button_label,
        key="toggle_conceptual_equipment_image_button",
        use_container_width=True,
        on_click=toggle_conceptual_equipment_image,
    )

    if not bool(st.session_state.get("show_conceptual_equipment_image", False)):
        return

    if not CONCEPTUAL_IMAGE_PATH.exists():
        st.warning(
            "No se encontró la imagen conceptual. Verifica que exista el archivo: "
            f"{CONCEPTUAL_IMAGE_PATH}"
        )
        return

    st.image(
        str(CONCEPTUAL_IMAGE_PATH),
        caption=(
            "Representación conceptual del equipo. "
            "No corresponde a una máquina real específica del dataset."
        ),
        use_container_width=True,
    )


def render_problem_explanation() -> None:
    """
    Muestra explicación funcional del problema que resuelve SIMAP-IA.
    """
    with st.expander("¿Qué hace este simulador?", expanded=False):
        st.markdown(
            """
            **SIMAP-IA estima la probabilidad de falla de una máquina bajo condiciones operativas.**

            El sistema recibe variables de operación, las transforma mediante ingeniería de
            características y usa un modelo entrenado localmente para estimar el riesgo de que
            ocurra una falla.

            El resultado no debe interpretarse como una fecha exacta de falla ni como vida útil
            restante. Es una estimación de riesgo bajo las condiciones ingresadas.
            """
        )


def render_dataset_variable_dictionary() -> None:
    """
    Muestra diccionario de variables originales del dataset AI4I 2020.
    """
    variable_rows = [
        {
            "Variable en el dataset": "Type",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Tipo o calidad del producto: L, M o H.",
        },
        {
            "Variable en el dataset": "Air temperature [K]",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Temperatura ambiente del aire medida en Kelvin.",
        },
        {
            "Variable en el dataset": "Process temperature [K]",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Temperatura del proceso medida en Kelvin.",
        },
        {
            "Variable en el dataset": "Rotational speed [rpm]",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Velocidad de rotación de la máquina en revoluciones por minuto.",
        },
        {
            "Variable en el dataset": "Torque [Nm]",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Torque o esfuerzo mecánico aplicado, medido en Newton-metro.",
        },
        {
            "Variable en el dataset": "Tool wear [min]",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Desgaste acumulado de la herramienta en minutos.",
        },
        {
            "Variable en el dataset": "Machine failure",
            "Uso en SIMAP-IA": "Variable objetivo",
            "Descripción": "Indica si hubo falla de máquina: 0 = no falla, 1 = falla.",
        },
        {
            "Variable en el dataset": "TWF, HDF, PWF, OSF, RNF",
            "Uso en SIMAP-IA": "Excluidas del modelo principal",
            "Descripción": "Indicadores de modos de falla. No se usan como entradas para evitar fuga de información.",
        },
    ]

    with st.expander("¿De dónde salen las variables?", expanded=False):
        st.markdown(
            """
            Las variables provienen del archivo **ai4i2020.csv** del dataset AI4I 2020.
            El simulador no inventa las variables: usa las columnas operativas originales
            del dataset y estima la variable objetivo **Machine failure**.
            """
        )
        st.table(pd.DataFrame(variable_rows))


def render_feature_interaction_explanation() -> None:
    """
    Explica cómo se relacionan las variables para estimar riesgo de falla.
    """
    interaction_rows = [
        {
            "Relación": "Estrés térmico",
            "Variables involucradas": "Process temperature [K] y Air temperature [K]",
            "Interpretación": "Una diferencia térmica elevada puede indicar condiciones térmicas más exigentes.",
        },
        {
            "Relación": "Carga mecánica",
            "Variables involucradas": "Torque [Nm] y Rotational speed [rpm]",
            "Interpretación": "Torque alto con velocidad baja puede representar mayor esfuerzo operativo.",
        },
        {
            "Relación": "Desgaste bajo carga",
            "Variables involucradas": "Torque [Nm] y Tool wear [min]",
            "Interpretación": "El desgaste es más relevante cuando ocurre junto con mayor esfuerzo mecánico.",
        },
        {
            "Relación": "Condición acumulada",
            "Variables involucradas": "Tool wear [min]",
            "Interpretación": "Mayor desgaste puede aumentar la vulnerabilidad de la máquina.",
        },
        {
            "Relación": "Contexto de producto",
            "Variables involucradas": "Type",
            "Interpretación": "El tipo L, M o H permite al modelo aprender diferencias operativas entre categorías.",
        },
    ]

    with st.expander("¿Cómo se interrelacionan las variables para detectar fallas?", expanded=False):
        st.markdown(
            """
            El modelo no toma decisiones usando una sola variable. Primero crea variables
            derivadas y luego aprende patrones combinados.

            Por ejemplo, un torque alto no siempre implica falla. Pero torque alto junto con
            desgaste elevado, diferencia térmica alta o velocidad anómala puede aumentar el riesgo.
            """
        )
        st.table(pd.DataFrame(interaction_rows))


def render_derived_features_explanation() -> None:
    """
    Muestra las variables derivadas creadas internamente por el pipeline.
    """
    derived_rows = [
        {
            "Variable derivada": "Thermal delta [K]",
            "Fórmula": "Process temperature [K] - Air temperature [K]",
            "Sentido técnico": "Mide diferencia térmica entre proceso y ambiente.",
        },
        {
            "Variable derivada": "Approx mechanical power",
            "Fórmula": "Torque [Nm] * Rotational speed [rpm]",
            "Sentido técnico": "Aproxima carga mecánica operativa.",
        },
        {
            "Variable derivada": "Torque wear load",
            "Fórmula": "Torque [Nm] * Tool wear [min]",
            "Sentido técnico": "Representa desgaste acumulado bajo esfuerzo mecánico.",
        },
        {
            "Variable derivada": "Torque speed ratio",
            "Fórmula": "Torque [Nm] / Rotational speed [rpm]",
            "Sentido técnico": "Relaciona esfuerzo mecánico con velocidad de operación.",
        },
        {
            "Variable derivada": "Tool wear normalized",
            "Fórmula": "Tool wear [min] / máximo desgaste observado",
            "Sentido técnico": "Escala el desgaste respecto al máximo aprendido en entrenamiento.",
        },
    ]

    with st.expander("Variables derivadas creadas por el modelo", expanded=False):
        st.markdown(
            """
            Además de las variables originales, SIMAP-IA genera variables derivadas
            dentro del pipeline. Estas variables ayudan al modelo a capturar relaciones
            físicas y operativas que no aparecen explícitamente en el dataset.
            """
        )
        st.table(pd.DataFrame(derived_rows))


def render_model_flow_explanation() -> None:
    """
    Muestra el flujo completo desde datos hasta predicción.
    """
    with st.expander("Flujo interno del modelo", expanded=False):
        st.markdown(
            """
            El flujo interno de SIMAP-IA es:

            **1. Entrada de variables operativas**  
            El usuario ingresa tipo de producto, temperaturas, velocidad, torque y desgaste.

            **2. Validación de entradas**  
            La app revisa rangos, tipos de datos y valores permitidos.

            **3. Ingeniería de variables**  
            El pipeline crea variables derivadas como diferencia térmica, carga mecánica
            aproximada y carga por desgaste.

            **4. Preprocesamiento**  
            La variable categórica `Type` se codifica y las variables numéricas se escalan.

            **5. Modelo predictivo**  
            El modelo entrenado estima la probabilidad de `Machine failure = 1`.

            **6. Clasificación de riesgo**  
            La probabilidad se traduce a bajo, medio, alto o crítico usando umbrales definidos.
            """
        )


def render_explanatory_panels() -> None:
    """
    Renderiza todos los paneles explicativos principales de la app.
    """
    render_problem_explanation()
    render_dataset_variable_dictionary()
    render_feature_interaction_explanation()
    render_derived_features_explanation()
    render_model_flow_explanation()


def build_reference_range_table() -> pd.DataFrame:
    """
    Construye una tabla con rangos empíricos observados en el dataset AI4I 2020.

    Returns:
        pd.DataFrame: Tabla de variables, mínimos, máximos y unidades.
    """
    rows = []

    for variable_name, config in DATASET_REFERENCE_RANGES.items():
        unit = variable_name.split("[")[-1].replace("]", "") if "[" in variable_name else ""

        rows.append(
            {
                "Variable": variable_name,
                "Mínimo observado": config["min"],
                "Máximo observado": config["max"],
                "Valor inicial": config["default"],
                "Unidad": unit,
            }
        )

    return pd.DataFrame(rows)


def render_input_reference_ranges() -> None:
    """
    Muestra los rangos de referencia usados por el simulador.
    """
    with st.expander("Rangos de entrada usados por el simulador", expanded=False):
        st.markdown(
            """
            Los campos numéricos se limitan a los rangos observados en el dataset
            **AI4I 2020**. Esto evita que el modelo reciba valores fuera del dominio
            donde fue entrenado.

            En un sistema industrial real, estos límites deberían ajustarse con datos
            históricos propios de la planta, especificaciones técnicas de sensores y reglas
            de operación del equipo.
            """
        )
        st.table(build_reference_range_table())


def get_numeric_input_config(variable_name: str) -> dict:
    """
    Obtiene la configuración de entrada numérica para una variable.

    Args:
        variable_name: Nombre de la variable del dataset.

    Returns:
        dict: Configuración de rango, valor inicial, paso y ayuda.

    Raises:
        KeyError: Si la variable no existe en DATASET_REFERENCE_RANGES.
    """
    if variable_name not in DATASET_REFERENCE_RANGES:
        raise KeyError(f"No existe configuración de rango para: {variable_name}")

    return DATASET_REFERENCE_RANGES[variable_name]




def render_dataset_credit() -> None:
    """
    Muestra crédito formal del dataset AI4I 2020.
    """
    with st.expander("Créditos del dataset", expanded=False):
        st.markdown(
            f"""
            **Dataset utilizado:** Predictive Maintenance Dataset (AI4I 2020)

            **Autor:** Stephan Matzka

            **Fuente:** Kaggle - Predictive Maintenance Dataset (AI4I 2020)

            **Enlace:** [{DATASET_SOURCE_URL}]({DATASET_SOURCE_URL})

            **Referencia académica:**  
            {DATASET_CITATION}

            **Nota de alcance:**  
            El dataset AI4I 2020 es sintético y fue modelado a partir de un
            proceso de fresado/mecanizado. No corresponde a una máquina real
            específica identificable. SIMAP-IA debe interpretarse como prototipo
            académico y demostrativo.
            """
        )


def build_dataset_dictionary_table() -> pd.DataFrame:
    """
    Construye el diccionario de columnas del dataset AI4I 2020.

    Returns:
        pd.DataFrame: Tabla descriptiva de columnas.
    """
    rows = [
        {
            "Columna": "UDI",
            "Tipo": "Identificador",
            "Uso en SIMAP-IA": "Solo exploración",
            "Descripción": "Identificador único del registro.",
        },
        {
            "Columna": "Product ID",
            "Tipo": "Identificador categórico",
            "Uso en SIMAP-IA": "No usado como predictor",
            "Descripción": "Identificador del producto con variante L, M o H y número serial.",
        },
        {
            "Columna": "Type",
            "Tipo": "Categórica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Tipo o calidad del producto: L, M o H.",
        },
        {
            "Columna": "Air temperature [K]",
            "Tipo": "Numérica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Temperatura ambiente del aire medida en Kelvin.",
        },
        {
            "Columna": "Process temperature [K]",
            "Tipo": "Numérica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Temperatura del proceso medida en Kelvin.",
        },
        {
            "Columna": "Rotational speed [rpm]",
            "Tipo": "Numérica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Velocidad rotacional de la máquina en revoluciones por minuto.",
        },
        {
            "Columna": "Torque [Nm]",
            "Tipo": "Numérica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Torque o carga mecánica aplicada durante el proceso.",
        },
        {
            "Columna": "Tool wear [min]",
            "Tipo": "Numérica",
            "Uso en SIMAP-IA": "Entrada del modelo",
            "Descripción": "Desgaste acumulado de la herramienta en minutos.",
        },
        {
            "Columna": "Machine failure",
            "Tipo": "Binaria",
            "Uso en SIMAP-IA": "Variable objetivo",
            "Descripción": "Indica si hubo falla de máquina: 0 = no falla, 1 = falla.",
        },
    ]

    for mode_column, mode_label in FAILURE_MODE_LABELS.items():
        rows.append(
            {
                "Columna": mode_column,
                "Tipo": "Binaria",
                "Uso en SIMAP-IA": "Solo exploración",
                "Descripción": (
                    f"{mode_label}. No se usa como entrada del modelo principal "
                    "para evitar fuga de información."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_dataset_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Calcula indicadores generales del dataset.

    Args:
        df: Dataset AI4I 2020.

    Returns:
        dict[str, float | int]: Indicadores calculados.
    """
    total_rows = int(len(df))
    failure_count = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df.columns else 0
    non_failure_count = int(total_rows - failure_count)
    failure_rate = float(failure_count / total_rows) if total_rows > 0 else 0.0

    return {
        "rows": total_rows,
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "failure_count": failure_count,
        "non_failure_count": non_failure_count,
        "failure_rate": failure_rate,
    }


def render_dataset_summary(df: pd.DataFrame) -> None:
    """
    Muestra métricas generales del dataset.

    Args:
        df: Dataset AI4I 2020.
    """
    summary = build_dataset_summary(df)
    col1, col2, col3 = st.columns(3)

    col1.metric("Registros", f"{summary['rows']:,}")
    col2.metric("Columnas", f"{summary['columns']:,}")
    col3.metric("Tasa de falla", f"{summary['failure_rate']:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Fallas", f"{summary['failure_count']:,}")
    col5.metric("No fallas", f"{summary['non_failure_count']:,}")
    col6.metric("Valores faltantes", f"{summary['missing_values']:,}")

    st.caption(f"Filas duplicadas detectadas: {summary['duplicated_rows']:,}")


def apply_dataset_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros interactivos al dataset.

    Args:
        df: Dataset AI4I 2020.

    Returns:
        pd.DataFrame: Dataset filtrado.
    """
    filtered_df = df.copy()

    col1, col2, col3 = st.columns(3)

    type_options = ["Todos"] + sorted(df["Type"].dropna().unique().tolist())
    selected_type = col1.selectbox(
        "Tipo de producto",
        type_options,
        key="dataset_filter_type",
    )

    failure_option = col2.selectbox(
        "Machine failure",
        ["Todos", "No falla (0)", "Falla (1)"],
        key="dataset_filter_target",
    )

    mode_option = col3.selectbox(
        "Modo de falla",
        ["Todos"] + FAILURE_MODE_COLUMNS,
        key="dataset_filter_failure_mode",
    )

    if selected_type != "Todos":
        filtered_df = filtered_df[filtered_df["Type"] == selected_type]

    if failure_option == "No falla (0)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 0]
    elif failure_option == "Falla (1)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 1]

    if mode_option != "Todos":
        filtered_df = filtered_df[filtered_df[mode_option] == 1]

    with st.expander("Filtros numéricos", expanded=False):
        for variable_name in NUMERIC_DATASET_COLUMNS:
            min_value = float(df[variable_name].min())
            max_value = float(df[variable_name].max())
            step_value = 1.0 if "rpm" in variable_name or "min" in variable_name else 0.1

            selected_range = st.slider(
                f"Rango de {variable_name}",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value),
                step=step_value,
                key=f"dataset_filter_{variable_name}",
            )

            filtered_df = filtered_df[
                (filtered_df[variable_name] >= selected_range[0])
                & (filtered_df[variable_name] <= selected_range[1])
            ]

    return filtered_df


def render_dataset_table(df: pd.DataFrame) -> None:
    """
    Muestra tabla interactiva del dataset con filtros y descarga.

    Args:
        df: Dataset AI4I 2020.
    """
    filtered_df = apply_dataset_filters(df)

    st.write(f"Registros filtrados: **{len(filtered_df):,}** de **{len(df):,}**")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV filtrado",
        data=csv_data,
        file_name="ai4i2020_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_dataset_visualizations(df: pd.DataFrame) -> None:
    """
    Muestra visualizaciones del dataset AI4I 2020.

    Esta versión corrige el gráfico de torque vs velocidad rotacional
    creando una tabla limpia para visualización. Se evita depender de
    nombres de columnas con espacios, corchetes y unidades dentro del
    motor gráfico de Streamlit.

    Args:
        df: Dataset AI4I 2020.
    """
    st.markdown("#### Distribución por tipo de producto")
    type_counts = df["Type"].value_counts().sort_index()
    st.bar_chart(type_counts)

    st.markdown("#### Distribución de la variable objetivo")
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    target_counts.index = ["No falla (0)", "Falla (1)"]
    st.bar_chart(target_counts)

    st.markdown("#### Conteo por modo de falla")
    mode_counts = df[FAILURE_MODE_COLUMNS].sum().sort_values(ascending=False)
    st.bar_chart(mode_counts)

    st.markdown("#### Promedios operativos por clase de falla")
    grouped_summary = (
        df.groupby(TARGET_COLUMN)[NUMERIC_DATASET_COLUMNS]
        .mean()
        .round(3)
        .rename(index={0: "No falla", 1: "Falla"})
    )
    st.dataframe(grouped_summary, use_container_width=True)

    st.markdown("#### Relación torque vs velocidad rotacional")

    required_columns = [
        "Torque [Nm]",
        "Rotational speed [rpm]",
        TARGET_COLUMN,
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        st.warning(
            "No se puede generar el gráfico porque faltan columnas: "
            f"{missing_columns}"
        )
        return

    chart_df = (
        df[required_columns]
        .dropna()
        .rename(
            columns={
                "Torque [Nm]": "torque_nm",
                "Rotational speed [rpm]": "rotational_speed_rpm",
                TARGET_COLUMN: "machine_failure",
            }
        )
    )

    if chart_df.empty:
        st.warning("No hay datos disponibles para generar el gráfico.")
        return

    sample_size = min(1500, len(chart_df))
    chart_df = chart_df.sample(n=sample_size, random_state=42)
    chart_df["estado_falla"] = chart_df["machine_failure"].map(
        {
            0: "No falla",
            1: "Falla",
        }
    )

    try:
        import altair as alt

        scatter_chart = (
            alt.Chart(chart_df)
            .mark_circle(size=45, opacity=0.65)
            .encode(
                x=alt.X(
                    "torque_nm:Q",
                    title="Torque [Nm]",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "rotational_speed_rpm:Q",
                    title="Velocidad rotacional [rpm]",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "estado_falla:N",
                    title="Machine failure",
                ),
                tooltip=[
                    alt.Tooltip("torque_nm:Q", title="Torque [Nm]", format=".2f"),
                    alt.Tooltip(
                        "rotational_speed_rpm:Q",
                        title="Velocidad rotacional [rpm]",
                        format=".2f",
                    ),
                    alt.Tooltip("estado_falla:N", title="Estado"),
                ],
            )
            .interactive()
        )

        st.altair_chart(scatter_chart, use_container_width=True)
    except Exception:
        st.caption(
            "Visualización alternativa sin Altair. "
            "El gráfico usa nombres internos limpios para asegurar renderizado."
        )
        st.scatter_chart(
            chart_df,
            x="torque_nm",
            y="rotational_speed_rpm",
            use_container_width=True,
        )



def render_failure_mode_explorer(df: pd.DataFrame) -> None:
    """
    Muestra información de modos de falla del dataset.

    Args:
        df: Dataset AI4I 2020.
    """
    st.warning(
        "Los modos TWF, HDF, PWF, OSF y RNF se muestran solo para exploración. "
        "No se usan como entradas del modelo principal porque revelarían información "
        "demasiado cercana a la variable objetivo."
    )

    rows = []
    for column, label in FAILURE_MODE_LABELS.items():
        count = int(df[column].sum())
        rows.append(
            {
                "Modo": column,
                "Nombre": label,
                "Registros con falla": count,
                "Porcentaje del dataset": f"{count / len(df):.2%}",
            }
        )

    st.table(pd.DataFrame(rows))


def render_dataset_explorer() -> None:
    """
    Renderiza el explorador interactivo del dataset AI4I 2020.
    """
    with st.expander("Explorar dataset AI4I 2020", expanded=False):
        df = load_dataset()

        if df.empty:
            st.warning(
                "No se encontró el dataset local. Copia el archivo ai4i2020.csv en "
                "la carpeta simap-ia-web-public/data/ para activar el explorador."
            )
            return

        st.markdown(
            """
            Esta sección permite revisar el dataset usado para entrenar y validar
            el prototipo. La exploración se realiza con el archivo local
            `data/ai4i2020.csv`, sin llamadas externas y sin costos.
            """
        )

        summary_tab, dictionary_tab, data_tab, charts_tab, failure_tab = st.tabs(
            [
                "Resumen",
                "Diccionario",
                "Datos filtrables",
                "Gráficos",
                "Modos de falla",
            ]
        )

        with summary_tab:
            render_dataset_summary(df)

        with dictionary_tab:
            st.dataframe(build_dataset_dictionary_table(), use_container_width=True)

        with data_tab:
            render_dataset_table(df)

        with charts_tab:
            render_dataset_visualizations(df)

        with failure_tab:
            render_failure_mode_explorer(df)





def generate_test_identifier(prefix: str = "SIMAP") -> str:
    """
    Genera un identificador aleatorio para registrar pruebas.

    Args:
        prefix: Prefijo del identificador.

    Returns:
        str: Identificador único corto compatible con registro en Google Sheets.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_token = uuid4().hex[:8].upper()
    return f"{prefix}-{timestamp}-{random_token}"


def normalize_risk_level(level: str) -> str:
    """
    Normaliza el nivel de riesgo a una etiqueta legible en español.

    Args:
        level: Nivel devuelto por el clasificador.

    Returns:
        str: Nivel normalizado.
    """
    normalized = str(level).strip().lower()
    mapping = {
        "low": "Bajo",
        "bajo": "Bajo",
        "medium": "Medio",
        "medio": "Medio",
        "high": "Alto",
        "alto": "Alto",
        "critical": "Crítico",
        "critico": "Crítico",
        "crítico": "Crítico",
    }

    return mapping.get(normalized, str(level))


def create_prediction_result(
    case_name: str,
    case_kind: str,
    record: dict,
    probability: float,
    risk,
) -> dict:
    """
    Construye un resultado de predicción trazable.

    Args:
        case_name: Nombre del caso evaluado.
        case_kind: Tipo de caso: preset_configurado o particular.
        record: Variables de entrada.
        probability: Probabilidad estimada de falla.
        risk: Objeto de riesgo devuelto por SIMAPRiskClassifier.

    Returns:
        dict: Resultado listo para mostrar, exportar y registrar.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    risk_level_raw = str(risk.level)
    risk_level = normalize_risk_level(risk_level_raw)

    return {
        "test_id": generate_test_identifier(),
        "session_id": get_current_session_id(),
        "timestamp_utc": timestamp_utc,
        "case_name": case_name,
        "case_kind": case_kind,
        "record": dict(record),
        "probability": float(probability),
        "probability_percent": float(probability) * 100,
        "risk_level_raw": risk_level_raw,
        "risk_level": risk_level,
        "recommendation": str(risk.recommendation),
        "dataset": "Predictive Maintenance Dataset (AI4I 2020)",
        "dataset_url": DATASET_SOURCE_URL,
        "app_name": APP_NAME,
        "app_version": str(get_persistence_config().get("app_version", APP_VERSION)),
    }


def predict_single_record(
    record: dict,
    predictor: SIMAPPredictor,
    validator: SIMAPInputValidator,
    risk_classifier: SIMAPRiskClassifier,
    case_name: str,
    case_kind: str,
) -> tuple[dict | None, list[str]]:
    """
    Ejecuta validación e inferencia para un solo registro.

    Args:
        record: Variables de entrada.
        predictor: Predictor del pipeline entrenado.
        validator: Validador de entradas.
        risk_classifier: Clasificador de niveles de riesgo.
        case_name: Nombre del caso.
        case_kind: Tipo del caso.

    Returns:
        tuple[dict | None, list[str]]: Resultado o None, y lista de errores.
    """
    errors = validator.validate_record(record)

    if errors:
        return None, errors

    input_df = validator.to_dataframe(record)
    probability = predictor.predict_failure_probability(input_df)
    risk = risk_classifier.classify(probability)
    result = create_prediction_result(
        case_name=case_name,
        case_kind=case_kind,
        record=record,
        probability=probability,
        risk=risk,
    )

    return result, []


def build_results_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convierte resultados en una tabla plana compatible con Google Sheets.

    Args:
        results: Resultados de predicción.

    Returns:
        pd.DataFrame: Tabla plana de resultados.
    """
    rows = []

    for result in results:
        record = result["record"]
        rows.append(
            {
                "test_id": result["test_id"],
                "session_id": result.get("session_id", get_current_session_id()),
                "timestamp_utc": result["timestamp_utc"],
                "case_name": result["case_name"],
                "case_kind": result["case_kind"],
                "type": record["Type"],
                "air_temperature_k": record["Air temperature [K]"],
                "process_temperature_k": record["Process temperature [K]"],
                "rotational_speed_rpm": record["Rotational speed [rpm]"],
                "torque_nm": record["Torque [Nm]"],
                "tool_wear_min": record["Tool wear [min]"],
                "probability": round(result["probability"], 8),
                "probability_percent": round(result["probability_percent"], 4),
                "risk_level": result["risk_level"],
                "recommendation": result["recommendation"],
                "dataset": result["dataset"],
                "app_version": result.get("app_version", APP_VERSION),
            }
        )

    return pd.DataFrame(rows)


def build_results_csv(results: list[dict]) -> bytes:
    """
    Genera CSV descargable para Google Sheets.

    Args:
        results: Resultados de predicción.

    Returns:
        bytes: CSV codificado en UTF-8 con BOM para compatibilidad.
    """
    results_df = build_results_dataframe(results)
    return results_df.to_csv(index=False).encode("utf-8-sig")



def escape_pdf_text(value: object) -> str:
    """
    Escapa texto para escritura segura dentro de un PDF básico.

    Args:
        value: Valor a convertir a texto PDF.

    Returns:
        str: Texto escapado para un literal PDF.
    """
    clean_text = str(value)
    clean_text = clean_text.replace("\\", "\\\\")
    clean_text = clean_text.replace("(", "\\(")
    clean_text = clean_text.replace(")", "\\)")
    return clean_text


def wrap_report_line(text: str, max_width: int = 92) -> list[str]:
    """
    Divide una línea textual en fragmentos de ancho aproximado.

    Args:
        text: Línea original.
        max_width: Longitud máxima aproximada por línea.

    Returns:
        list[str]: Líneas envueltas.
    """
    words = str(text).split()
    if not words:
        return [""]

    wrapped_lines = []
    current_line = ""

    for word in words:
        candidate = f"{current_line} {word}".strip()

        if len(candidate) <= max_width:
            current_line = candidate
        else:
            wrapped_lines.append(current_line)
            current_line = word

    if current_line:
        wrapped_lines.append(current_line)

    return wrapped_lines


def build_basic_pdf_report(results: list[dict], report_title: str) -> bytes:
    """
    Genera un PDF básico sin dependencias externas.

    Esta función actúa como respaldo cuando ReportLab no está disponible en el
    entorno de ejecución de Streamlit. El reporte mantiene el identificador de
    prueba, variables operativas, resultado predictivo, recomendación y crédito
    del dataset.

    Args:
        results: Resultados de predicción.
        report_title: Título del reporte.

    Returns:
        bytes: Contenido PDF válido.
    """
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = [
        report_title,
        f"Generado UTC: {generated_at}",
        "SIMAP-IA - Sistema Inteligente de Mantenimiento Predictivo con IA",
        "",
        "Dataset: Predictive Maintenance Dataset (AI4I 2020)",
        "Autor: Stephan Matzka",
        f"Fuente: {DATASET_SOURCE_URL}",
        f"Referencia: {DATASET_CITATION}",
        "",
        "Nota: reporte generado por respaldo PDF sin dependencias externas.",
        "",
    ]

    for index, result in enumerate(results, start=1):
        record = result["record"]
        lines.extend(
            [
                f"{index}. {result['case_name']} ({result['case_kind']})",
                f"Identificador: {result['test_id']}",
                f"Fecha UTC: {result['timestamp_utc']}",
                f"Probabilidad estimada de falla: {result['probability_percent']:.2f}%",
                f"Nivel de riesgo: {result['risk_level']}",
                f"Recomendacion: {result['recommendation']}",
                "Variables de entrada:",
                f"  Tipo de producto: {record['Type']}",
                f"  Temperatura del aire [K]: {record['Air temperature [K]']:.2f}",
                f"  Temperatura del proceso [K]: {record['Process temperature [K]']:.2f}",
                f"  Velocidad rotacional [rpm]: {record['Rotational speed [rpm]']:.2f}",
                f"  Torque [Nm]: {record['Torque [Nm]']:.2f}",
                f"  Desgaste de herramienta [min]: {record['Tool wear [min]']:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "Registro sugerido para Google Sheets:",
            "Usar test_id como identificador unico para cruzar pruebas, reportes y registros.",
        ]
    )

    printable_lines: list[str] = []
    for line in lines:
        printable_lines.extend(wrap_report_line(line))

    lines_per_page = 45
    pages = [
        printable_lines[start : start + lines_per_page]
        for start in range(0, len(printable_lines), lines_per_page)
    ]

    if not pages:
        pages = [["Reporte SIMAP-IA sin contenido."]]

    objects: list[bytes] = []
    page_object_numbers: list[int] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    # Reserved objects:
    # 1 Catalog, 2 Pages, 3 Font
    objects.extend([b"", b"", b""])
    font_object_number = 3

    for page_lines in pages:
        content_commands = ["BT", "/F1 10 Tf", "50 740 Td", "14 TL"]

        for line_index, line in enumerate(page_lines):
            escaped_line = escape_pdf_text(line)
            encoded_line = escaped_line.encode("latin-1", errors="replace").decode("latin-1")
            if line_index == 0:
                content_commands.append(f"({encoded_line}) Tj")
            else:
                content_commands.append(f"T* ({encoded_line}) Tj")

        content_commands.append("ET")
        content_stream = "\n".join(content_commands).encode("latin-1", errors="replace")

        content_object = (
            b"<< /Length "
            + str(len(content_stream)).encode("ascii")
            + b" >>\nstream\n"
            + content_stream
            + b"\nendstream"
        )
        content_object_number = add_object(content_object)

        page_object_number = len(objects) + 1
        page_content = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
            f"/Contents {content_object_number} 0 R >>"
        ).encode("ascii")
        add_object(page_content)
        page_object_numbers.append(page_object_number)

    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{page_number} 0 R" for page_number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("ascii")
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    pdf_parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets = [0]

    for object_number, object_content in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in pdf_parts))
        pdf_parts.append(f"{object_number} 0 obj\n".encode("ascii"))
        pdf_parts.append(object_content)
        pdf_parts.append(b"\nendobj\n")

    xref_offset = sum(len(part) for part in pdf_parts)
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf_parts.append(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf_parts.append(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
    )

    return b"".join(pdf_parts)




def build_pdf_report(results: list[dict], report_title: str) -> bytes:
    """
    Genera un reporte PDF en memoria usando ReportLab.

    Args:
        results: Resultados de predicción.
        report_title: Título del reporte.

    Returns:
        bytes: Contenido PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except Exception:
        return build_basic_pdf_report(results, report_title)

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SIMAPTitle",
            parent=styles["Title"],
            fontSize=17,
            leading=21,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SIMAPBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
        )
    )

    elements = []
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    elements.append(Paragraph(report_title, styles["SIMAPTitle"]))
    elements.append(Paragraph(f"<b>Generado UTC:</b> {generated_at}", styles["SIMAPBody"]))
    elements.append(
        Paragraph(
            "Sistema Inteligente de Mantenimiento Predictivo con Inteligencia Artificial. "
            "El reporte resume pruebas simuladas y genera identificadores para registro "
            "externo en Google Sheets.",
            styles["SIMAPBody"],
        )
    )
    elements.append(Spacer(1, 0.25 * cm))

    credit_text = (
        "<b>Dataset:</b> Predictive Maintenance Dataset (AI4I 2020). "
        "<b>Autor:</b> Stephan Matzka. "
        f"<b>Fuente:</b> {DATASET_SOURCE_URL}. "
        f"<b>Referencia:</b> {DATASET_CITATION}"
    )
    elements.append(Paragraph(credit_text, styles["SIMAPBody"]))
    elements.append(Spacer(1, 0.35 * cm))

    for index, result in enumerate(results, start=1):
        record = result["record"]
        elements.append(
            Paragraph(
                f"<b>{index}. {result['case_name']}</b> "
                f"({result['case_kind']})",
                styles["Heading3"],
            )
        )

        summary_data = [
            ["Identificador", result["test_id"]],
            ["Fecha UTC", result["timestamp_utc"]],
            ["Probabilidad estimada de falla", f"{result['probability_percent']:.2f}%"],
            ["Nivel de riesgo", result["risk_level"]],
            ["Recomendación", result["recommendation"]],
        ]

        input_data = [
            ["Variable", "Valor"],
            ["Tipo de producto", record["Type"]],
            ["Temperatura del aire [K]", f"{record['Air temperature [K]']:.2f}"],
            ["Temperatura del proceso [K]", f"{record['Process temperature [K]']:.2f}"],
            ["Velocidad rotacional [rpm]", f"{record['Rotational speed [rpm]']:.2f}"],
            ["Torque [Nm]", f"{record['Torque [Nm]']:.2f}"],
            ["Desgaste de herramienta [min]", f"{record['Tool wear [min]']:.2f}"],
        ]

        summary_table = Table(summary_data, colWidths=[5.2 * cm, 11.0 * cm])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )

        input_table = Table(input_data, colWidths=[7.0 * cm, 4.5 * cm])
        input_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )

        elements.append(summary_table)
        elements.append(Spacer(1, 0.18 * cm))
        elements.append(input_table)
        elements.append(Spacer(1, 0.35 * cm))

    google_df = build_results_dataframe(results)
    elements.append(Paragraph("<b>Registro recomendado para Google Sheets</b>", styles["Heading3"]))
    elements.append(
        Paragraph(
            "Cada fila contiene un identificador aleatorio único test_id. "
            "También se adjunta CSV descargable desde la aplicación.",
            styles["SIMAPBody"],
        )
    )
    elements.append(Spacer(1, 0.15 * cm))

    google_preview_columns = [
        "test_id",
        "timestamp_utc",
        "case_name",
        "probability_percent",
        "risk_level",
    ]
    preview_data = [google_preview_columns]
    for _, row in google_df[google_preview_columns].iterrows():
        preview_data.append([str(row[column]) for column in google_preview_columns])

    preview_table = Table(
        preview_data,
        colWidths=[4.2 * cm, 4.3 * cm, 4.4 * cm, 2.2 * cm, 2.0 * cm],
        repeatRows=1,
    )
    preview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.append(preview_table)

    document.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()

    return pdf_data


def evaluate_configured_presets(
    predictor: SIMAPPredictor,
    validator: SIMAPInputValidator,
    risk_classifier: SIMAPRiskClassifier,
) -> list[dict]:
    """
    Ejecuta las tres pruebas configuradas del prototipo.

    Args:
        predictor: Predictor entrenado.
        validator: Validador de entradas.
        risk_classifier: Clasificador de niveles de riesgo.

    Returns:
        list[dict]: Resultados de las pruebas configuradas.
    """
    results = []

    for preset_name in CONFIGURED_PRESET_NAMES:
        record = dict(PRESET_TESTS[preset_name])
        result, errors = predict_single_record(
            record=record,
            predictor=predictor,
            validator=validator,
            risk_classifier=risk_classifier,
            case_name=preset_name,
            case_kind="preset_configurado",
        )

        if errors:
            raise ValueError(f"Errores en {preset_name}: {errors}")

        if result is not None:
            results.append(result)

    return results


def render_prediction_result_exports(result: dict) -> None:
    """
    Muestra descargas del resultado individual y fila para Google Sheets.

    Args:
        result: Resultado de predicción.
    """
    st.markdown("#### Registro de la prueba")
    st.caption(
        "Este identificador se registra automáticamente para el creador de la app "
        "y permite cruzar la prueba con reportes posteriores."
    )
    st.code(result["test_id"], language="text")

    last_status = st.session_state.get("last_persistence_status", {})
    if bool(last_status.get("google_sheets", False)):
        st.caption("Registro automático en Google Sheets confirmado.")
    elif is_google_sheets_ready(get_persistence_config()):
        st.caption("Predicción generada; el registro automático en Google Sheets no fue confirmado.")
    else:
        st.caption("Google Sheets no configurado en este entorno; se mantiene respaldo local temporal.")

    result_df = build_results_dataframe([result])
    st.dataframe(result_df, use_container_width=True)

    col1, col2 = st.columns(2)
    pdf_data = build_pdf_report(
        [result],
        report_title="Reporte individual de prueba SIMAP-IA",
    )
    csv_data = build_results_csv([result])

    col1.download_button(
        label="Descargar reporte PDF de esta prueba",
        data=pdf_data,
        file_name=f"simap_reporte_{result['test_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        on_click=register_interaction_event,
        args=(
            "individual_pdf_requested",
            {"source": "single_prediction"},
            result["test_id"],
            result["case_name"],
        ),
    )

    col2.download_button(
        label="Descargar CSV de respaldo de esta prueba",
        data=csv_data,
        file_name=f"simap_respaldo_{result['test_id']}.csv",
        mime="text/csv",
        use_container_width=True,
        on_click=register_interaction_event,
        args=(
            "single_csv_requested",
            {"source": "single_prediction_backup"},
            result["test_id"],
            result["case_name"],
        ),
    )


def render_session_history_exports() -> None:
    """
    Permite descargar el historial de pruebas ejecutadas durante la sesión.
    """
    history = st.session_state.get("prediction_history", [])

    if not history:
        return

    with st.expander("Historial de pruebas de esta sesión", expanded=False):
        st.dataframe(build_results_dataframe(history), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.download_button(
            label="Descargar PDF del historial",
            data=build_pdf_report(history, "Reporte de historial de pruebas SIMAP-IA"),
            file_name="simap_historial_pruebas.pdf",
            mime="application/pdf",
            use_container_width=True,
            on_click=register_interaction_event,
            args=(
                "history_pdf_requested",
                {"records": len(history)},
                "",
                "Historial de sesión",
            ),
        )
        col2.download_button(
            label="Descargar CSV de respaldo del historial",
            data=build_results_csv(history),
            file_name="simap_historial_respaldo.csv",
            mime="text/csv",
            use_container_width=True,
            on_click=register_interaction_event,
            args=(
                "history_csv_requested",
                {"records": len(history)},
                "",
                "Historial de sesión",
            ),
        )


def render_configured_preset_reports(
    predictor: SIMAPPredictor,
    validator: SIMAPInputValidator,
    risk_classifier: SIMAPRiskClassifier,
) -> None:
    """
    Renderiza generación y descarga de reportes para las tres pruebas configuradas.

    Args:
        predictor: Predictor entrenado.
        validator: Validador de entradas.
        risk_classifier: Clasificador de niveles de riesgo.
    """
    with st.expander("Reportes de pruebas configuradas", expanded=False):
        st.markdown(
            """
            Genera un reporte con las tres pruebas base del simulador. Cada ejecución
            crea identificadores aleatorios nuevos para que puedan registrarse en
            Google Sheets sin duplicar llaves.
            """
        )

        if st.button(
            "Generar reporte de las 3 pruebas configuradas",
            use_container_width=True,
            key="generate_configured_preset_report_button",
        ):
            try:
                preset_results_generated = evaluate_configured_presets(
                    predictor=predictor,
                    validator=validator,
                    risk_classifier=risk_classifier,
                )
                st.session_state["preset_report_results"] = preset_results_generated

                for preset_result in preset_results_generated:
                    register_prediction_result(
                        preset_result,
                        event_type="configured_preset_report_prediction",
                    )

                register_interaction_event(
                    event_type="configured_preset_report_generated",
                    metadata={"records": len(preset_results_generated)},
                    case_name="Reporte de 3 pruebas configuradas",
                )
            except Exception as error:
                st.error(f"No fue posible generar el reporte de presets: {error}")
                st.session_state["preset_report_results"] = []

        preset_results = st.session_state.get("preset_report_results", [])

        if not preset_results:
            return

        st.dataframe(build_results_dataframe(preset_results), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.download_button(
            label="Descargar PDF de las 3 pruebas",
            data=build_pdf_report(
                preset_results,
                "Reporte de pruebas configuradas SIMAP-IA",
            ),
            file_name="simap_reporte_3_pruebas_configuradas.pdf",
            mime="application/pdf",
            use_container_width=True,
            on_click=register_interaction_event,
            args=(
                "configured_tests_pdf_requested",
                {"records": len(preset_results)},
                "",
                "Reporte de 3 pruebas configuradas",
            ),
        )
        col2.download_button(
            label="Descargar CSV de respaldo de las 3 pruebas",
            data=build_results_csv(preset_results),
            file_name="simap_3_pruebas_respaldo.csv",
            mime="text/csv",
            use_container_width=True,
            on_click=register_interaction_event,
            args=(
                "configured_tests_csv_requested",
                {"records": len(preset_results)},
                "",
                "Reporte de 3 pruebas configuradas",
            ),
        )




def render_metrics(metrics: dict) -> None:
    """Muestra métricas locales del modelo."""
    with st.expander("Métricas locales del modelo desplegado", expanded=False):
        if metrics:
            st.json(metrics)
        else:
            st.warning("No se encontró model_metrics.json.")


def main() -> None:
    """Ejecuta la aplicación Streamlit."""
    st.set_page_config(
        page_title="SIMAP-IA",
        page_icon="🛠️",
        layout="centered",
    )

    initialize_session_state()
    register_session_started_event()

    st.title("SIMAP-IA")
    st.subheader("Sistema Inteligente de Mantenimiento Predictivo con Inteligencia Artificial")

    thresholds = load_json(THRESHOLDS_PATH)
    metrics = load_json(METRICS_PATH)

    validator = SIMAPInputValidator()
    risk_classifier = SIMAPRiskClassifier(
        thresholds
        if thresholds
        else {
            "low": [0.0, 0.2],
            "medium": [0.2, 0.5],
            "high": [0.5, 0.75],
            "critical": [0.75, 1.0],
        }
    )

    st.divider()
    st.header("Simulador de riesgo de falla")

    st.selectbox(
        "Preselector de pruebas",
        list(PRESET_TESTS.keys()),
        key="selected_preset_name",
        help=(
            "Selecciona una prueba configurada o usa la prueba personalizada. "
            "Al cambiar la selección, los campos del simulador se actualizan automáticamente."
        ),
        on_change=apply_selected_preset_to_inputs,
    )

    with st.form("risk_form"):
        st.selectbox(
            "Tipo de producto",
            ["L", "M", "H"],
            key=INPUT_WIDGET_KEYS["Type"],
            help="Categorías originales del dataset AI4I 2020.",
        )

        air_temp_config = get_numeric_input_config("Air temperature [K]")
        process_temp_config = get_numeric_input_config("Process temperature [K]")
        speed_config = get_numeric_input_config("Rotational speed [rpm]")
        torque_config = get_numeric_input_config("Torque [Nm]")
        tool_wear_config = get_numeric_input_config("Tool wear [min]")

        st.number_input(
            "Temperatura del aire [K]",
            min_value=float(air_temp_config["min"]),
            max_value=float(air_temp_config["max"]),
            step=float(air_temp_config["step"]),
            key=INPUT_WIDGET_KEYS["Air temperature [K]"],
            help=str(air_temp_config["description"]),
        )

        st.number_input(
            "Temperatura del proceso [K]",
            min_value=float(process_temp_config["min"]),
            max_value=float(process_temp_config["max"]),
            step=float(process_temp_config["step"]),
            key=INPUT_WIDGET_KEYS["Process temperature [K]"],
            help=str(process_temp_config["description"]),
        )

        st.number_input(
            "Velocidad rotacional [rpm]",
            min_value=float(speed_config["min"]),
            max_value=float(speed_config["max"]),
            step=float(speed_config["step"]),
            key=INPUT_WIDGET_KEYS["Rotational speed [rpm]"],
            help=str(speed_config["description"]),
        )

        st.number_input(
            "Torque [Nm]",
            min_value=float(torque_config["min"]),
            max_value=float(torque_config["max"]),
            step=float(torque_config["step"]),
            key=INPUT_WIDGET_KEYS["Torque [Nm]"],
            help=str(torque_config["description"]),
        )

        st.number_input(
            "Desgaste de herramienta [min]",
            min_value=float(tool_wear_config["min"]),
            max_value=float(tool_wear_config["max"]),
            step=float(tool_wear_config["step"]),
            key=INPUT_WIDGET_KEYS["Tool wear [min]"],
            help=str(tool_wear_config["description"]),
        )

        submitted = st.form_submit_button("Calcular riesgo")

    if submitted:
        record = build_record_from_session_state()
        case_name, case_kind = identify_case_metadata(record)

        try:
            predictor = load_predictor()
            result, errors = predict_single_record(
                record=record,
                predictor=predictor,
                validator=validator,
                risk_classifier=risk_classifier,
                case_name=case_name,
                case_kind=case_kind,
            )

            if errors:
                for error in errors:
                    st.error(error)
                return

            if result is None:
                st.error("No fue posible generar un resultado válido.")
                return

            st.session_state["last_prediction_result"] = result
            st.session_state["prediction_history"].append(result)
            register_prediction_result(result, event_type="prediction_calculated")

            st.success("Predicción generada correctamente.")
            st.metric(
                "Probabilidad estimada de falla",
                f"{result['probability_percent']:.2f}%",
            )
            st.metric("Nivel de riesgo", risk_badge(result["risk_level_raw"]))
            st.write("**Recomendación:**", result["recommendation"])
            render_prediction_result_exports(result)

        except Exception as error:
            st.error(f"No fue posible generar la predicción: {error}")

    try:
        predictor_for_reports = load_predictor()
        render_configured_preset_reports(
            predictor=predictor_for_reports,
            validator=validator,
            risk_classifier=risk_classifier,
        )
    except Exception as error:
        st.warning(f"No fue posible preparar reportes de presets: {error}")

    render_session_history_exports()

    st.divider()
    st.header("Información técnica y contexto del prototipo")
    render_scope_notice()
    render_persistence_status()
    render_conceptual_equipment_image_button()
    render_dataset_credit()
    render_dataset_explorer()
    render_input_reference_ranges()
    render_explanatory_panels()
    render_metrics(metrics)

    st.divider()
    st.caption(
        "SIMAP-IA usa artefactos entrenados localmente. "
        "El repositorio público no debe contener scripts de entrenamiento."
    )


if __name__ == "__main__":
    main()
