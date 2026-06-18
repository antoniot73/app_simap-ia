"""
Validadores de entrada para la aplicación web SIMAP-IA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NumericRange:
    """Rango numérico permitido."""

    minimum: float
    maximum: float


class SIMAPInputValidator:
    """
    Valida registros ingresados por el usuario antes de inferencia.
    """

    allowed_types: tuple[str, ...] = ("L", "M", "H")

    numeric_ranges: dict[str, NumericRange] = {
        "Air temperature [K]": NumericRange(250.0, 350.0),
        "Process temperature [K]": NumericRange(250.0, 400.0),
        "Rotational speed [rpm]": NumericRange(500.0, 4000.0),
        "Torque [Nm]": NumericRange(0.0, 120.0),
        "Tool wear [min]": NumericRange(0.0, 300.0),
    }

    required_features: tuple[str, ...] = (
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    )

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Indica si un valor puede convertirse a float."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """
        Valida un registro de entrada.

        Returns:
            list[str]: Lista de errores. Lista vacía si todo es válido.
        """
        errors: list[str] = []

        for feature in self.required_features:
            if feature not in record:
                errors.append(f"Falta la variable obligatoria: {feature}")

        if record.get("Type") not in self.allowed_types:
            errors.append(f"Type debe ser uno de {self.allowed_types}.")

        for column, valid_range in self.numeric_ranges.items():
            value = record.get(column)

            if not self._is_number(value):
                errors.append(f"{column} debe ser numérico.")
                continue

            numeric_value = float(value)
            if numeric_value < valid_range.minimum or numeric_value > valid_range.maximum:
                errors.append(
                    f"{column} debe estar entre {valid_range.minimum} y {valid_range.maximum}."
                )

        return errors

    def to_dataframe(self, record: dict[str, Any]) -> pd.DataFrame:
        """
        Convierte un registro validado a DataFrame con orden estable.
        """
        ordered_record = {
            "Type": str(record["Type"]),
            "Air temperature [K]": float(record["Air temperature [K]"]),
            "Process temperature [K]": float(record["Process temperature [K]"]),
            "Rotational speed [rpm]": float(record["Rotational speed [rpm]"]),
            "Torque [Nm]": float(record["Torque [Nm]"]),
            "Tool wear [min]": float(record["Tool wear [min]"]),
        }

        return pd.DataFrame([ordered_record])
