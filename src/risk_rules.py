"""
Reglas de clasificación de riesgo para SIMAP-IA.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    """Resultado de clasificación de riesgo."""

    level: str
    recommendation: str


class SIMAPRiskClassifier:
    """Clasifica una probabilidad en niveles de riesgo."""

    def __init__(self, thresholds: dict[str, list[float]]) -> None:
        """
        Inicializa el clasificador de riesgo.
        """
        self._thresholds = thresholds

    @staticmethod
    def default_recommendations() -> dict[str, str]:
        """Devuelve recomendaciones por nivel."""
        return {
            "Bajo": "Continuar operación normal y mantener monitoreo rutinario.",
            "Medio": "Incrementar monitoreo preventivo y revisar tendencia operativa.",
            "Alto": "Programar inspección preventiva del equipo.",
            "Crítico": "Realizar inspección inmediata antes de continuar operación.",
        }

    def classify(self, probability: float) -> RiskResult:
        """
        Clasifica una probabilidad en nivel de riesgo.
        """
        if probability < 0.0 or probability > 1.0:
            raise ValueError("La probabilidad debe estar entre 0 y 1.")

        if probability < self._thresholds["medium"][0]:
            level = "Bajo"
        elif probability < self._thresholds["high"][0]:
            level = "Medio"
        elif probability < self._thresholds["critical"][0]:
            level = "Alto"
        else:
            level = "Crítico"

        return RiskResult(level=level, recommendation=self.default_recommendations()[level])
