# Usar la imagen oficial de Python 3.12 ligera
FROM python:3.12-slim

WORKDIR /app

# Instalar uv
RUN pip install --no-cache-dir uv

# 1. Copiar tu pyproject.toml (y el uv.lock si ya se generó)
COPY pyproject.toml uv.lock* ./

# 2. Instalar PyTorch (CPU) primero
RUN uv pip install --system --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar el resto de dependencias directamente desde el pyproject.toml
RUN uv pip install --system --no-cache-dir -r pyproject.toml

# Copiar el código fuente
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]