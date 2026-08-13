"""Detección de drift de datos: compara la distribución de features recientes
contra la referencia registrada en el entrenamiento."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models.features import FEATURES, features_vector


def calcular_drift(
    features_recientes: list[dict[str, float]],
    referencia: dict[str, Any],
    umbral_z: float = 3.0,
) -> dict[str, Any]:
    if not features_recientes or not referencia or not referencia.get("media"):
        return {"nivel": "sin_referencia", "features_desviadas": [], "z_scores": {}}

    media = np.array(referencia["media"])
    desv = np.array(referencia["std"])
    X = np.vstack([features_vector(f) for f in features_recientes])
    z = np.nan_to_num((X.mean(axis=0) - media) / desv, nan=0.0)

    desviadas = [FEATURES[i] for i in np.where(np.abs(z) > umbral_z)[0]]
    top = sorted(range(len(z)), key=lambda i: abs(z[i]), reverse=True)[:5]
    z_scores = {FEATURES[i]: round(float(z[i]), 3) for i in top}

    if len(desviadas) >= 3:
        nivel = "critico"
    elif desviadas:
        nivel = "atencion"
    else:
        nivel = "ok"

    return {"nivel": nivel, "features_desviadas": desviadas, "z_scores": z_scores}
