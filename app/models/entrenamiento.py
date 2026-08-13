"""Pipeline de entrenamiento: dos clasificadores binarios (hipo / hiper) sobre
GradientBoosting, con imputación, validación en holdout y métricas."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.models.features import FEATURES, features_vector
from app.models.labels import CLASES, CLASE_HIPO, CLASE_HIPER, CLASE_NORMAL

_CLASES_ML = [CLASE_HIPO, CLASE_HIPER]


def _metricas_binaria(y_real: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    if len(set(y_real)) < 2 or len(set(y_pred)) < 2:
        return {
            "accuracy": float(accuracy_score(y_real, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc_roc": None,
        }
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision": float(precision_score(y_real, y_pred, zero_division=0)),
        "recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "f1": float(f1_score(y_real, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_real, proba)) if len(set(y_real)) > 1 else None,
    }


def entrenar_modelos(
    X: list[dict[str, float]],
    y: list[str],
    seed: int = 42,
    n_estimators: int = 120,
) -> dict[str, Any] | None:
    """Entrena los modelos hipo/hiper. Retorna artefacto serializable o None si no hay datos."""
    if not X or len(X) < 10:
        return None

    X_vec = np.vstack([features_vector(f) for f in X])
    y_arr = np.asarray(y)

    conteos = {c: int((y_arr == c).sum()) for c in CLASES}
    if conteos[CLASE_NORMAL] < 3 or (conteos[CLASE_HIPO] + conteos[CLASE_HIPER]) < 3:
        return None

    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X_vec)

    idx = np.arange(len(y_arr))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, stratify=y_arr, random_state=seed)

    modelos: dict[str, Any] = {}
    metricas_por_clase: dict[str, dict] = {}

    for clase in _CLASES_ML:
        yb = (y_arr == clase).astype(int)
        if int(yb.sum()) < 2:
            modelos[clase] = None
            metricas_por_clase[clase] = {}
            continue
        modelo = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            random_state=seed,
        )
        modelo.fit(Xt[train_idx], yb[train_idx])
        proba_test = modelo.predict_proba(Xt[test_idx])[:, 1]
        pred_bin = (proba_test >= 0.5).astype(int)
        target_bin = (y_arr[test_idx] == clase).astype(int)
        metricas_por_clase[clase] = _metricas_binaria(target_bin, pred_bin, proba_test)
        modelos[clase] = modelo

    # Precisión global de 3 clases (argmax de probabilidades normalizadas)
    m_hipo = modelos[CLASE_HIPO]
    m_hiper = modelos[CLASE_HIPER]
    p_hipo = m_hipo.predict_proba(Xt[test_idx])[:, 1] if m_hipo else np.zeros(len(test_idx))
    p_hiper = m_hiper.predict_proba(Xt[test_idx])[:, 1] if m_hiper else np.zeros(len(test_idx))
    p_normal = np.clip(1 - p_hipo - p_hiper, 0.0, 1.0)
    argmax = np.argmax(np.vstack([p_normal, p_hipo, p_hiper]), axis=0)
    pred_3clases = np.asarray(CLASES)[argmax]
    accuracy_global = float(accuracy_score(y_arr[test_idx], pred_3clases))

    referencia = {
        "media": [float(v) for v in Xt.mean(axis=0)],
        "std": [float(v) for v in (Xt.std(axis=0) + 1e-9)],
        "n": int(len(Xt)),
    }

    version = f"diabetes-hipo-hiper-v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

    return {
        "version": version,
        "fecha_entrenamiento": datetime.now(timezone.utc),
        "tipo": "hipo-hiper",
        "feature_order": FEATURES,
        "imputer": imputer,
        "modelo_hipo": modelos[CLASE_HIPO],
        "modelo_hiper": modelos[CLASE_HIPER],
        "referencia": referencia,
        "metricas": {
            "hipo": metricas_por_clase[CLASE_HIPO],
            "hiper": metricas_por_clase[CLASE_HIPER],
            "accuracy_global": accuracy_global,
        },
        "conteos": conteos,
        "total_muestras": int(len(y_arr)),
    }


def serializar_artefacto(artefacto: dict[str, Any]) -> bytes:
    return pickle.dumps(artefacto, protocol=pickle.HIGHEST_PROTOCOL)


def deserializar_artefacto(data: bytes) -> dict[str, Any] | None:
    """Deserializar artefacto de sklearn.

    NOTA: Pickle se usa solo para modelos internos de confianza (sklearn).
    Los datos vienen de MongoDB interno (controlado por BioGuard).
    No se usa pickle con datos untrusted de usuarios.
    """
    if not data:
        return None
    try:
        return pickle.loads(data)  # nosec: B301 - datos internos confiables
    except Exception:
        return None
