"""
Ingeniería de variables para SIMAP-IA.

Este módulo crea variables derivadas con sentido físico-operativo a partir
de las variables originales del dataset AI4I 2020.

Debe existir tanto en:
- simap-ia-local-private/src/features.py
- simap-ia-web-public/src/features.py

Esto permite que los artefactos entrenados localmente puedan cargarse
correctamente en la aplicación web.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class SIMAPFeatureNames:
    """
    Define variables usadas, derivadas, excluidas y objetivo del proyecto.

    Attributes:
        input_features: Variables operativas usadas como entrada.
        derived_features: Variables creadas por ingeniería de características.
        excluded_features: Variables que no deben entrar al predictor principal.
        target: Variable objetivo del modelo.
    """

    input_features: tuple[str, ...] = (
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    )

    derived_features: tuple[str, ...] = (
        "Thermal delta [K]",
        "Approx mechanical power",
        "Torque wear load",
        "Torque speed ratio",
        "Tool wear normalized",
    )

    excluded_features: tuple[str, ...] = (
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    )

    target: str = "Machine failure"


class SIMAPFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador compatible con scikit-learn para crear variables derivadas.
    """

    _EPSILON: ClassVar[float] = 1e-9

    def __init__(self) -> None:
        """Inicializa el transformador de variables."""
        self.max_tool_wear_: float | None = None
        self.feature_names = SIMAPFeatureNames()

    @property
    def input_features(self) -> tuple[str, ...]:
        """Devuelve las variables originales requeridas."""
        return self.feature_names.input_features

    @property
    def derived_features(self) -> tuple[str, ...]:
        """Devuelve las variables derivadas generadas."""
        return self.feature_names.derived_features

    @property
    def output_features(self) -> tuple[str, ...]:
        """Devuelve todas las variables generadas por el transformador."""
        return self.input_features + self.derived_features

    @staticmethod
    def get_numeric_features() -> tuple[str, ...]:
        """Devuelve las variables numéricas originales."""
        return (
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        )

    def _validate_dataframe(self, data: object) -> pd.DataFrame:
        """
        Valida que la entrada sea un DataFrame con columnas requeridas.

        Raises:
            TypeError: Si la entrada no es un pandas.DataFrame.
            ValueError: Si faltan columnas requeridas.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("SIMAPFeatureEngineer requiere un pandas.DataFrame.")

        df = data.copy()
        missing_columns: list[str] = []

        for column in self.input_features:
            if column not in df.columns:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

        return df

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas operativas numéricas y valida valores faltantes.

        Raises:
            ValueError: Si existen valores no numéricos o faltantes.
        """
        numeric_columns = list(self.get_numeric_features())

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        missing_values = int(df[numeric_columns].isna().sum().sum())

        if missing_values > 0:
            raise ValueError(
                "Existen valores no numéricos o faltantes en columnas operativas."
            )

        return df

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "SIMAPFeatureEngineer":
        """
        Ajusta el transformador calculando el máximo desgaste observado.
        """
        df = self._validate_dataframe(X)
        df = self._convert_numeric_columns(df)

        max_tool_wear = float(df["Tool wear [min]"].max())

        if not np.isfinite(max_tool_wear) or max_tool_wear <= 0:
            max_tool_wear = 1.0

        self.max_tool_wear_ = max_tool_wear

        logging.info(
            "SIMAPFeatureEngineer ajustado correctamente. max_tool_wear_=%.4f",
            self.max_tool_wear_,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Genera variables derivadas para entrenamiento o inferencia.
        """
        if self.max_tool_wear_ is None:
            raise RuntimeError("Debe ejecutar fit() antes de transform().")

        df = self._validate_dataframe(X)
        df = self._convert_numeric_columns(df)

        df["Thermal delta [K]"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

        df["Approx mechanical power"] = (
            df["Torque [Nm]"] * df["Rotational speed [rpm]"]
        )

        df["Torque wear load"] = (
            df["Torque [Nm]"] * df["Tool wear [min]"]
        )

        df["Torque speed ratio"] = (
            df["Torque [Nm]"]
            / (df["Rotational speed [rpm]"].abs() + self._EPSILON)
        )

        df["Tool wear normalized"] = (
            df["Tool wear [min]"] / max(self.max_tool_wear_, self._EPSILON)
        )

        return df[list(self.output_features)]

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> np.ndarray:
        """
        Devuelve nombres de variables de salida compatibles con scikit-learn.
        """
        return np.array(self.output_features, dtype=object)


def build_feature_preview(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """
    Genera una vista previa de las variables derivadas.
    """
    if rows <= 0:
        raise ValueError("rows debe ser mayor que 0.")

    transformer = SIMAPFeatureEngineer()
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    return transformed_df.head(rows)
