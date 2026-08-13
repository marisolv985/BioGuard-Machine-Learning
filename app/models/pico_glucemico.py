"""Motor de picos glucémicos (fórmulas F1-F3 + matriz de clasificación de riesgo).

F1 - IMC:         peso_kg / estatura_m^2
F2 - z:           w0 + (w1 * Pulso) + (w2 * Sudor) + (w3 * Temp) + (w4 * IMC)
F3 - P(Pico):     1 / (1 + e^-z)

Matriz de clasificación (misma lógica para el motor local del móvil):
  - Hipoglucemia Nocturna: Pulso > 110 BPM, Temp < 35 °C, Sudor > 80 µS → Crítico Alto.
  - Hiperglucemia Severa:  Pulso 95-110 BPM, Temp > 37.2 °C, Sudor < 20 µS → Moderado Alto.
  - Estado Óptimo:          Pulso 60-80 BPM, Temp 36-36.7 °C, Sudor 15-35 µS → Bajo (Estable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---- Casos clínicos / acciones automatizadas -------------------------------
CASO_HIPO = "Hipoglucemia Nocturna"
CASO_HIPER = "Hiperglucemia Severa"
CASO_OPTIMO = "Estado Optimo"
CASO_VIGILANCIA = "Vigilancia"

ACCION_HIPO = (
    "Detonar alerta sonora/haptica en Reloj y Celular; si no hay respuesta en 60s, "
    "enviar SMS con ubicacion GPS a red familiar."
)
ACCION_HIPER = (
    "Enviar notificacion dirigida al celular con recomendaciones de hidratacion "
    "y caminata ligera."
)
ACCION_OPTIMO = (
    "Mantener streaming BLE continuo cada 10 segundos y realizar almacenamiento "
    "silencioso en SQLite local."
)
ACCION_VIGILANCIA = "Mantener monitoreo continuo; evaluar siguiente lectura en 10 segundos."

# ---- Pesos por defecto (se persisten en Mongo: collection "modelos") --------
PESOS_DEFAULT = {
    "w0": -8.0,
    "w1": 0.05,   # Pulso (bpm)
    "w2": 0.02,   # Sudoración GSR (µS)
    "w3": -0.04,  # Temperatura (°C)
    "w4": 0.15,   # IMC
}


@dataclass(frozen=True)
class PesosPico:
    w0: float = -8.0
    w1: float = 0.05
    w2: float = 0.02
    w3: float = -0.04
    w4: float = 0.15

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PesosPico":
        if not d:
            return cls()
        return cls(
            w0=float(d.get("w0", PESOS_DEFAULT["w0"])),
            w1=float(d.get("w1", PESOS_DEFAULT["w1"])),
            w2=float(d.get("w2", PESOS_DEFAULT["w2"])),
            w3=float(d.get("w3", PESOS_DEFAULT["w3"])),
            w4=float(d.get("w4", PESOS_DEFAULT["w4"])),
        )

    @classmethod
    def from_settings(cls, s: Any) -> "PesosPico":
        return cls(
            w0=float(getattr(s, "pesos_w0", PESOS_DEFAULT["w0"])),
            w1=float(getattr(s, "pesos_w1", PESOS_DEFAULT["w1"])),
            w2=float(getattr(s, "pesos_w2", PESOS_DEFAULT["w2"])),
            w3=float(getattr(s, "pesos_w3", PESOS_DEFAULT["w3"])),
            w4=float(getattr(s, "pesos_w4", PESOS_DEFAULT["w4"])),
        )

    def to_dict(self) -> dict[str, float]:
        return {"w0": self.w0, "w1": self.w1, "w2": self.w2, "w3": self.w3, "w4": self.w4}


def calcular_imc(peso_kg: float, estatura_m: float) -> float:
    """F1: IMC = peso / estatura^2."""
    if estatura_m is None or estatura_m <= 0:
        raise ValueError("la estatura en metros debe ser mayor a 0")
    if peso_kg is None or peso_kg <= 0:
        raise ValueError("el peso en kg debe ser mayor a 0")
    return peso_kg / (estatura_m**2)


def calcular_z(
    pulso_bpm: float,
    sudor_us: float,
    temperatura_c: float,
    imc: float,
    pesos: PesosPico | None = None,
) -> float:
    """F2: z = w0 + w1*Pulso + w2*Sudor + w3*Temp + w4*IMC."""
    p = pesos or PesosPico()
    return p.w0 + (p.w1 * pulso_bpm) + (p.w2 * sudor_us) + (p.w3 * temperatura_c) + (p.w4 * imc)


def calcular_p_pico(z: float) -> float:
    """F3: P(Pico) = 1 / (1 + e^-z)."""
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 1.0


def clasificar(
    pulso_bpm: float,
    temperatura_c: float,
    sudor_us: float,
) -> tuple[str, str, str]:
    """Matriz de clasificación: (caso_clinico, nivel_riesgo, accion_automatizada)."""
    if pulso_bpm > 110.0 and temperatura_c < 35.0 and sudor_us > 80.0:
        return CASO_HIPO, "Critico Alto", ACCION_HIPO
    if 95.0 <= pulso_bpm <= 110.0 and temperatura_c > 37.2 and sudor_us < 20.0:
        return CASO_HIPER, "Moderado Alto", ACCION_HIPER
    if 60.0 <= pulso_bpm <= 80.0 and 36.0 <= temperatura_c <= 36.7 and 15.0 <= sudor_us <= 35.0:
        return CASO_OPTIMO, "Bajo (Estable)", ACCION_OPTIMO
    return CASO_VIGILANCIA, "Por evaluar", ACCION_VIGILANCIA


def evaluar(
    peso_kg: float,
    estatura_m: float,
    pulso_bpm: float,
    sudor_us: float,
    temperatura_c: float,
    pesos: PesosPico | None = None,
) -> dict[str, Any]:
    """Ejecuta F1 -> F2 -> F3 -> matriz y devuelve el bloque de salida."""
    imc = round(calcular_imc(peso_kg, estatura_m), 2)
    z = round(calcular_z(pulso_bpm, sudor_us, temperatura_c, imc, pesos), 4)
    p_pico = round(calcular_p_pico(z), 4)
    caso_clinico, nivel_riesgo, accion_automatizada = clasificar(pulso_bpm, temperatura_c, sudor_us)
    return {
        "imc": imc,
        "z": z,
        "p_pico": p_pico,
        "caso_clinico": caso_clinico,
        "nivel_riesgo": nivel_riesgo,
        "accion_automatizada": accion_automatizada,
    }
