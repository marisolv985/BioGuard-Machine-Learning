# BioGuard - Servicio de Predicción ML (Python / FastAPI)

Microservicio "cerebro predictivo" de BioGuard. Recibe telemetría de pacientes (smartwatch) y devuelve la
**probabilidad de crisis metabólica (0 a 1)**. Regla de negocio: `probabilidad >= 0.85` = **estado crítico**.

> Estado actual: predictor **baseline v0** — no es un valor fijo; calcula la probabilidad a partir de las
> señales vitales recibidas (severidad por desviación de rangos, explicable señal por señal). Cuando haya
> datos históricos se sustituye por un modelo ML entrenado **sin cambiar el contrato de la API**.

## Predictor baseline v0 (cómo funciona)

1. Cada señal vital se compara contra un rango saludable y un rango extremo → **severidad** `0..1`
   (`0` = saludable, `1` = peligro extremo). Ej.: FC en `60-100` → severidad 0; FC `170` → severidad 1.
2. Se combina **peor señal** (60% por defecto) + **media ponderada** de todas las señales.
3. El resultado se pasa por una función logística a una probabilidad `0..1`.

| Señales sanas (FC 72, T 36.6, SpO2 98…) | Señales críticas (FC 170, T 41, SpO2 70…) |
|---|---|
| `probabilidad = 0.0474` → `esCritico=false` | `probabilidad = 0.9324` → `esCritico=true` |

La respuesta incluye `contribuciones` (desglose de severidad por señal) y `explicacion` para trazabilidad
clínica.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/v1/salud` | Health check + versión del modelo activo |
| `POST` | `/api/v1/predicciones` | Recibe telemetría y devuelve la predicción |
| `GET` | `/docs` | Swagger UI (OpenAPI interactivo) |

## Estructura

```
app/
  main.py              # Aplicación FastAPI (montaje, CORS, routers)
  core/config.py       # Configuración vía variables de entorno (BIOGUARD_*)
  schemas/
    common.py          # Convertidor a camelCase (interop .NET)
    telemetria.py      # Esquema de entrada + validaciones
    prediccion.py      # Esquema de salida (probabilidad, esCritico, nivelRiesgo)
  api/
    deps.py            # Dependencias (singleton del predictor)
    routes/
      health.py        # GET /salud
      predict.py       # POST /predicciones
  services/
    base.py            # PredictorBase (interfaz) + niveles de riesgo
    baseline.py        # PredictorBaseline v0 (calcula con las señales vitales)
    predictor.py       # PredictorMock + factoría según BIOGUARD_PREDICTOR_ACTIVO
tests/                 # Pruebas con pytest + TestClient
```

## Requisitos

- Python 3.14 (probado) / 3.11+
- `venv/` ya existente con las dependencias instaladas.

## Uso local

```powershell
# Instalar dependencias (solo la primera vez)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Ejecutar pruebas
.\venv\Scripts\python.exe -m pytest

# Levantar servidor
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Swagger en <http://127.0.0.1:8000/docs>

### Ejemplo de petición

```json
POST /api/v1/predicciones

{
  "pacienteId": "P-001",
  "frecuenciaCardiaca": 95,
  "temperatura": 37.2,
  "saturacionOxigeno": 96.0,
  "frecuenciaRespiratoria": 18,
  "presionSistolica": 120,
  "presionDiastolica": 80,
  "glucosa": 105.0,
  "dispositivo": "smartwatch-v1"
}
```

```json
201 Created

{
  "pacienteId": "P-001",
  "probabilidad": 0.0474,
  "esCritico": false,
  "nivelRiesgo": "BAJO",
  "umbralCritico": 0.85,
  "timestamp": "2026-08-04T04:31:28.314312Z",
  "modeloId": "baseline-v0",
  "version": "0.1.0",
  "mensaje": "Estado estable: probabilidad por debajo del umbral crítico",
  "contribuciones": [
    { "senal": "Frecuencia cardíaca", "valor": 72.0, "severidad": 0.0 },
    { "senal": "Temperatura", "valor": 36.6, "severidad": 0.0 },
    { "senal": "Saturación de oxígeno", "valor": 98.0, "severidad": 0.0 },
    { "senal": "Glucosa", "valor": 95.0, "severidad": 0.0 }
  ],
  "explicacion": "Baseline v0: severidad por desviación de rangos vitales (peor señal + media ponderada) convertida a probabilidad logística."
}
```

La API **acepta y devuelve camelCase** (formato por defecto de .NET/System.Text.Json), pero también acepta
`snake_case` si tu cliente lo prefiere. Campos fuera de rango, desconocidos o presión sistólica <= diastólica
devuelven `422 Unprocessable Entity` automáticamente.

## Integración con el backend .NET

Desde `Controllers/MLController.cs` (o `SensoresController.cs`) usa un `HttpClient` registrado:

```csharp
public class BioguardMlClient
{
    private readonly HttpClient _http;
    public BioguardMlClient(HttpClient http) => _http = http;

    public async Task<PrediccionMl?> PredecirAsync(TelemetriaDto t, CancellationToken ct = default)
    {
        var payload = new
        {
            pacienteId = t.PacienteId,
            frecuenciaCardiaca = t.FrecuenciaCardiaca,
            temperatura = t.Temperatura,
            saturacionOxigeno = t.SaturacionOxigeno,
            frecuenciaRespiratoria = t.FrecuenciaRespiratoria,
            presionSistolica = t.PresionSistolica,
            presionDiastolica = t.PresionDiastolica,
            glucosa = t.Glucosa,
            dispositivo = t.Dispositivo
        };
        var resp = await _http.PostAsJsonAsync("api/v1/predicciones", payload, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<PrediccionMl>(ct);
    }
}

public record PrediccionMl(
    string PacienteId,
    double Probabilidad,
    bool EsCritico,
    string NivelRiesgo,
    double UmbralCritico,
    DateTime Timestamp,
    string ModeloId,
    string Version,
    string? Mensaje);
```

```csharp
builder.Services.AddHttpClient<BioguardMlClient>(c => c.BaseAddress = new Uri("http://localhost:8000/"));
```

## Configuración (`BIOGUARD_*`)

| Variable | Defecto | Descripción |
|---|---|---|
| `BIOGUARD_UMBRAL_CRITICO` | `0.85` | Umbral de negocio para estado crítico |
| `BIOGUARD_PREDICTOR_ACTIVO` | `baseline` | `baseline` (reacciona a señales) o `mock` (valor fijo) |
| `BIOGUARD_MODELO_ACTIVO` | `baseline-v0` | Id del modelo activo (para trazabilidad) |
| `BIOGUARD_BASELINE_PESO_PEOR_SENAL` | `0.6` | Peso de la peor señal vs la media ponderada |
| `BIOGUARD_PROBABILIDAD_MOCK` | `0.90` | Probabilidad fija del mock (solo modo `mock`) |
| `BIOGUARD_CORS_ORIGINS` | `["*"]` | Orígenes permitidos |
| `BIOGUARD_DEBUG` | `true` | Modo debug |
| `BIOGUARD_LOG_LEVEL` | `INFO` | Nivel de logging |

Copia `.env.example` a `.env` para personalizarla.

## Docker

```powershell
docker build -t bioguard-ml .
docker run -p 8000:8000 bioguard-ml
```

## Reemplazar el baseline por el modelo real

1. Crea `PredictorModeloML(PredictorBase)` en `app/services/` que cargue tu modelo entrenado (p. ej. joblib/onnx)
   y calcule `probabilidad` a partir de `TelemetriaEntrada`.
2. Actualiza `crear_predictor()` en `app/services/predictor.py` para devolverlo según `settings.modelo_activo`.
3. Los endpoints y contratos no cambian: el backend .NET sigue consumiendo el mismo JSON.
