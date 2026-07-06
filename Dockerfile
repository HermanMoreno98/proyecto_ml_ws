# Usar la imagen oficial de Python 3.12 ligera
FROM python:3.12-slim

WORKDIR /app

# Actualizar pip y copiar metadatos de dependencias
RUN python -m pip install --upgrade pip
COPY pyproject.toml ./

# Instalar PyTorch (CPU) primero
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias de la aplicación
RUN python -m pip install --no-cache-dir fastapi pydantic transformers uvicorn

# Copiar el código fuente
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
