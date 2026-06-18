"""
Motor de inferencia de SIMAP-IA.

Carga artefactos entrenados localmente y genera predicciones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class SIMAPPredictor:
    """Servicio de inferencia para el pipeline entrenado."""

    def __init__(self, pipeline_path: Path) -> None:
        """
        Inicializa el predictor.

        Raises:
            FileNotFoundError: Si el artefacto no existe.
        """
        if not pipeline_path.exists():
            raise FileNotFoundError(f"No existe el artefacto del modelo: {pipeline_path}")

        self._pipeline_path = pipeline_path
        self._pipeline: Any = joblib.load(pipeline_path)

    def predict_failure_probability(self, features: pd.DataFrame) -> float:
        """
        Calcula la probabilidad de falla.

        Returns:
            float: Probabilidad de falla entre 0 y 1.
        """
        if not hasattr(self._pipeline, "predict_proba"):
            raise RuntimeError("El pipeline cargado no soporta predict_proba.")

        probabilities = self._pipeline.predict_proba(features)
        return float(probabilities[0, 1])
