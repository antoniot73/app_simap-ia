"""
Utilidades de interfaz para SIMAP-IA.
"""

from __future__ import annotations


def format_probability(probability: float) -> str:
    """Formatea una probabilidad como porcentaje."""
    return f"{probability * 100:.2f}%"


def risk_badge(level: str) -> str:
    """Devuelve una etiqueta visual de riesgo."""
    badges = {
        "Bajo": "🟢 Bajo",
        "Medio": "🟡 Medio",
        "Alto": "🟠 Alto",
        "Crítico": "🔴 Crítico",
    }
    return badges.get(level, level)
