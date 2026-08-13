"""Pruebas del motor de picos glucémicos (F1/F2/F3 + matriz de riesgo)."""

import math

import pytest

from app.models.pico_glucemico import (
    ACCION_HIPER,
    ACCION_HIPO,
    ACCION_OPTIMO,
    CASO_HIPER,
    CASO_HIPO,
    CASO_OPTIMO,
    CASO_VIGILANCIA,
    PesosPico,
    calcular_imc,
    calcular_p_pico,
    calcular_z,
    clasificar,
    evaluar,
)


def test_calcular_imc():
    # F1: 80 kg / (1.75 m)^2 = 26.122...
    assert calcular_imc(80.0, 1.75) == pytest.approx(26.12, abs=0.01)


def test_calcular_imc_estatura_invalida():
    with pytest.raises(ValueError):
        calcular_imc(80.0, 0.0)


def test_calcular_z_y_sigmoide():
    pesos = PesosPico(w0=-10.0, w1=0.05, w2=0.03, w3=-0.6, w4=0.2)
    imc = calcular_imc(80.0, 1.75)
    z = calcular_z(pulso_bpm=120.0, sudor_us=90.0, temperatura_c=34.0, imc=imc, pesos=pesos)
    p = calcular_p_pico(z)
    # z = -10 + 6 + 2.7 - 20.4 + 5.22 = -16.48 -> P muy baja
    assert z == pytest.approx(-16.48, abs=0.01)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-z)))


def test_calcular_z_hipo_activa_probabilidad_alta():
    # Pulso 130, sudor 95, temp 34.4, IMC 22: z alto -> P alto
    pesos = PesosPico()
    imc = calcular_imc(60.0, 1.65)
    z = calcular_z(130.0, 95.0, 34.4, imc, pesos)
    assert calcular_p_pico(z) > 0.9


def test_matriz_hipoglucemia_nocturna():
    caso, nivel, accion = clasificar(115.0, 34.5, 85.0)
    assert caso == CASO_HIPO
    assert nivel == "Critico Alto"
    assert accion == ACCION_HIPO


def test_matriz_hiperglucemia_severa():
    caso, nivel, accion = clasificar(100.0, 37.5, 12.0)
    assert caso == CASO_HIPER
    assert nivel == "Moderado Alto"
    assert accion == ACCION_HIPER


def test_matriz_estado_optimo():
    caso, nivel, accion = clasificar(72.0, 36.4, 25.0)
    assert caso == CASO_OPTIMO
    assert nivel == "Bajo (Estable)"
    assert accion == ACCION_OPTIMO


def test_matriz_limites_inclusivos():
    assert clasificar(110.0, 35.0, 80.0)[0] != CASO_HIPO  # 110 no es > 110
    assert clasificar(95.0, 37.3, 19.0)[0] == CASO_HIPER  # 95 inclusive
    assert clasificar(60.0, 36.0, 15.0)[0] == CASO_OPTIMO  # límites inferiores inclusive


def test_matriz_fuera_de_patron_vigilancia():
    caso, nivel, accion = clasificar(90.0, 36.5, 45.0)
    assert caso == CASO_VIGILANCIA


def test_evaluar_bloque_salida():
    r = evaluar(
        peso_kg=80.0,
        estatura_m=1.75,
        pulso_bpm=72.0,
        sudor_us=25.0,
        temperatura_c=36.4,
        pesos=PesosPico(),
    )
    assert set(r) == {
        "imc",
        "z",
        "p_pico",
        "caso_clinico",
        "nivel_riesgo",
        "accion_automatizada",
    }
    assert r["caso_clinico"] == CASO_OPTIMO
    assert r["imc"] == pytest.approx(26.12, abs=0.01)
    assert 0.0 <= r["p_pico"] <= 1.0


def test_pesos_from_settings_override():
    class FakeSettings:
        pesos_w0 = -1.0
        pesos_w1 = 0.1
        pesos_w2 = 0.0
        pesos_w3 = 0.0
        pesos_w4 = 0.0

    p = PesosPico.from_settings(FakeSettings())
    assert p.w0 == -1.0
    assert p.w1 == 0.1
