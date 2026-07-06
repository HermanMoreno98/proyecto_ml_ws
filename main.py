import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, Pipeline, pipeline

logger = logging.getLogger("ner_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

NER_MODEL_NAME = "dslim/bert-base-NER"
HF_CACHE_DIR = "/root/.cache/huggingface"
SUPPORTED_ENTITY_GROUPS = ["PER", "LOC", "ORG", "MISC"]

ml_models: dict[str, Any] = {
    "ner_model": None,
    "startup_error": None,
}


def load_ner_pipeline() -> Pipeline:
    """Crea y retorna el pipeline NER usando CPU y caché local."""
    tokenizer = AutoTokenizer.from_pretrained(
        NER_MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
    )
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servidor y cargando modelo en memoria...")
    try:
        ml_models["ner_model"] = load_ner_pipeline()
        ml_models["startup_error"] = None
        logger.info("Modelo cargado exitosamente.")
    except Exception as exc:
        ml_models["ner_model"] = None
        ml_models["startup_error"] = str(exc)
        logger.exception("Error al cargar el modelo NER en el arranque.")

    yield

    logger.info("Apagando servidor, liberando modelo...")
    ml_models.clear()


app = FastAPI(lifespan=lifespan, title="NER API (Reconocimiento de Entidades)")

# --- SCHEMAS DE PYDANTIC (Validación de Inputs y Outputs) ---
class PredictRequest(BaseModel):
    # Validamos que el input sea un string de entre 2 y 2000 caracteres
    text: str = Field(..., min_length=2, max_length=2000, description="Texto en inglés para analizar")

class EntityModel(BaseModel):
    entity_group: str
    word: str
    confidence: float
    start: int | None = None
    end: int | None = None

class PredictResponse(BaseModel):
    prediction: list[EntityModel]
    inference_time_ms: float
    metadata: dict[str, Any]

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    model_loaded = ml_models.get("ner_model") is not None
    response: dict[str, Any] = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }
    if ml_models.get("startup_error"):
        response["error"] = ml_models["startup_error"]
    return response

@app.get("/")
def root_info():
    return {
        "service": "NER API",
        "description": "Reconoce entidades nombradas en texto en inglés.",
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict",
        "supported_entities": "/supported-entities",
    }

@app.get("/supported-entities")
def supported_entities():
    """Retorna las etiquetas de entidad soportadas por el modelo."""
    return {"supported_entity_groups": SUPPORTED_ENTITY_GROUPS}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Recibe texto, ejecuta el modelo de ML y retorna la predicción estructurada."""
    if ml_models.get("ner_model") is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Revise /health.")

    payload = request.text.strip()
    if not payload:
        raise HTTPException(status_code=422, detail="El texto de entrada no puede estar vacío.")

    start_time = time.time()
    try:
        raw_results = await run_in_threadpool(ml_models["ner_model"], payload)
    except Exception as exc:
        logger.exception("Error durante la inferencia NER.")
        raise HTTPException(status_code=500, detail="Fallo interno durante la inferencia.") from exc

    inference_time_ms = (time.time() - start_time) * 1000

    formatted_predictions = [
        EntityModel(
            entity_group=item.get("entity_group", "UNKNOWN"),
            word=item.get("word", ""),
            confidence=float(item.get("score", 0.0)),
            start=item.get("start"),
            end=item.get("end"),
        )
        for item in raw_results
    ]

    entity_counts: dict[str, int] = {}
    total_confidence = 0.0
    for item in raw_results:
        group = item.get("entity_group", "UNKNOWN")
        entity_counts[group] = entity_counts.get(group, 0) + 1
        total_confidence += float(item.get("score", 0.0))

    average_confidence = round(total_confidence / len(raw_results), 4) if raw_results else 0.0
    input_word_count = len(payload.split())
    unique_entities = len({item.get("word", "") for item in raw_results if item.get("word")})

    return PredictResponse(
        prediction=formatted_predictions,
        inference_time_ms=round(inference_time_ms, 2),
        metadata={
            "model_version": NER_MODEL_NAME,
            "language": "en",
            "entities_found": len(formatted_predictions),
            "entity_counts": entity_counts,
            "input_word_count": input_word_count,
            "average_confidence": average_confidence,
            "unique_entities": unique_entities,
        },
    )
