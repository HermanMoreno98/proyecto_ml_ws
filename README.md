# NER API: Despliegue de Modelo ML con FastAPI y Docker
Autores: 
- Franco Lazo Acuña
- Herman Paul Moreno Alvarado

Este proyecto implementa una API RESTful robusta para el reconocimiento de entidades nombradas (NER) utilizando el modelo pre-entrenado `dslim/bert-base-NER` de Hugging Face. Está diseñado siguiendo buenas prácticas de MLOps, optimizado para ejecución en CPU y empaquetado en contenedores Docker.

## Cumplimiento de Requisitos

La arquitectura de la API garantiza el cumplimiento de los siguientes patrones de diseño:

- **Modelo en memoria:** Utiliza el manejador de ciclo de vida (`lifespan`) de FastAPI para cargar el modelo en la RAM del servidor en el momento del arranque, evitando el sobrecosto de carga en cada petición HTTP.
- **Validación de inputs:** Implementada mediante esquemas estrictos de **Pydantic**, garantizando que los textos de entrada cumplan con tipos de datos y longitudes específicas antes de tocar el modelo ML.
- **Health check (`GET /health`):** Endpoint dedicado que no solo responde código 200, sino que verifica activamente si los pesos del modelo se encuentran cargados en el diccionario global.
- **Endpoint de predicción (`POST /predict`):** Recibe el texto, orquesta la inferencia y retorna entidades reconocidas.
- **Respuesta estructurada:** Devuelve un JSON tipado que incluye la predicción (agrupando sub-tokens), el tiempo de inferencia exacto en milisegundos (`inference_time_ms`) y metadatos relevantes (versión del modelo, idioma y cantidad de entidades encontradas).

## Stack Tecnológico

- **Framework Web:** FastAPI
- **Machine Learning:** Hugging Face `transformers`, PyTorch (CPU-optimized)
- **Gestión de Paquetes:** Pip y `pyproject.toml`
- **Infraestructura:** Docker y Docker Compose

## Estructura del Proyecto

```text
proyecto_ml_ws/
├── main.py                # Lógica central de la API, ciclo de vida y endpoints
├── pyproject.toml         # Manifiesto de dependencias (gestionado por uv)
├── Dockerfile             # Receta de construcción de la imagen optimizada
├── docker-compose.yml     # Orquestación de servicios y volúmenes de caché
└── README.md              # Documentación del proyecto
```

## Instrucciones de Despliegue (Local)

Requisitos previos
- Docker y Docker Compose instalados.
- Git.

Pasos para levantar el servicio

1. Clonar el repositorio y entrar a la carpeta:

```
git clone 
cd proyecto_ml_ws
```

2. Construir y levantar el contenedor en segundo plano:

```
docker compose up -d --build
```

## Característica adicional interesante

La API no solo extrae entidades, sino que también devuelve conteos por tipo de entidad y la cantidad de palabras del texto de entrada. Esto permite analizar rápidamente qué tipos de entidades aparecen con mayor frecuencia en cada petición.

## Uso de la API (Ejemplos)

Puedes interactuar con la API mediante curl, Postman o utilizando la interfaz gráfica interactiva (Swagger UI) que FastAPI genera automáticamente.

1. Interfaz Gráfica (Swagger)

Abre tu navegador y visita: http://localhost:8000/docs
Desde allí podrás probar ambos endpoints sin necesidad de la terminal.

2. Health Check

Verifica el estado del modelo en el servidor.

Request
```
curl -X 'GET' 'http://localhost:8000/health'
```

Response (200 OK):
```
{
  "status": "healthy",
  "model_loaded": true
}
```

3. Predicción NER

Envía un texto en inglés para extraer personas (PER), organizaciones (ORG), localizaciones (LOC) y misceláneos (MISC).

Request
```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Tim Cook is the CEO of Apple and he lives in California."
}'
```

Response (200 OK):
```
{
  "prediction": [
    {
      "entity_group": "PER",
      "word": "Tim Cook",
      "confidence": 0.9987
    },
    {
      "entity_group": "ORG",
      "word": "Apple",
      "confidence": 0.9991
    },
    {
      "entity_group": "LOC",
      "word": "California",
      "confidence": 0.9995
    }
  ],
  "inference_time_ms": 145.32,
  "metadata": {
    "model_version": "dslim/bert-base-NER",
    "language": "en",
    "entities_found": 3,
    "entity_counts": {
      "PER": 1,
      "ORG": 1,
      "LOC": 1
    },
    "input_word_count": 10
  }
}
```

### Entidades soportadas

También puedes consultar las etiquetas de entidad soportadas por el modelo con:

```
curl -X 'GET' 'http://localhost:8000/supported-entities'
```

Respuesta:

```
{
  "supported_entity_groups": ["PER", "LOC", "ORG", "MISC"]
}
```
