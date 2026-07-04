from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import time
from contextlib import asynccontextmanager

# Diccionario global para mantener el modelo cargado en memoria
ml_models = {}

# Patrón moderno en FastAPI para manejar la carga al inicio del servidor
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando servidor y cargando modelo en memoria...")
    # aggregation_strategy="simple" agrupa sub-tokens en palabras completas
    ml_models["ner_model"] = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    print("Modelo cargado exitosamente.")
    
    yield # Aquí es donde la aplicación corre y recibe requests
    
    # Limpieza de memoria al apagar el servidor
    print("Apagando servidor, liberando modelo...")
    ml_models.clear()

# Inicializamos FastAPI
app = FastAPI(lifespan=lifespan, title="NER API (Reconocimiento de Entidades)")

# --- SCHEMAS DE PYDANTIC (Validación de Inputs y Outputs) ---
class PredictRequest(BaseModel):
    # Validamos que el input sea un string de entre 2 y 2000 caracteres
    text: str = Field(..., min_length=2, max_length=2000, description="Texto en inglés para analizar")

class EntityModel(BaseModel):
    entity_group: str
    word: str
    confidence: float

class PredictResponse(BaseModel):
    prediction: list[EntityModel]
    inference_time_ms: float
    metadata: dict

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    """Verifica el estado de la API y si el modelo está cargado en memoria."""
    if "ner_model" in ml_models:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Recibe texto, ejecuta el modelo de ML y retorna la predicción estructurada."""
    if "ner_model" not in ml_models:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    
    start_time = time.time()
    
    # 1. Inferencia
    raw_results = ml_models["ner_model"](request.text)
    
    # 2. Calcular tiempo
    inference_time_ms = (time.time() - start_time) * 1000
    
    # 3. Formatear predicciones (convertimos float32 de numpy a float nativo de Python para JSON)
    formatted_predictions = []
    for item in raw_results:
        formatted_predictions.append(EntityModel(
            entity_group=item.get("entity_group", "UNKNOWN"),
            word=item.get("word", ""),
            confidence=float(item.get("score", 0.0)) # Cast vital para evitar errores de parseo
        ))
        
    # 4. Respuesta estructurada
    return PredictResponse(
        prediction=formatted_predictions,
        inference_time_ms=round(inference_time_ms, 2),
        metadata={
            "model_version": "dslim/bert-base-NER",
            "language": "en",
            "entities_found": len(formatted_predictions)
        }
    )