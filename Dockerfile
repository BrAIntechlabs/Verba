FROM python:3.11-slim

WORKDIR /app

# Copiamos el contenido del repo
COPY . .

# Instalamos dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Exponemos el puerto por defecto
EXPOSE 8000

# Comando para levantar FastAPI
CMD ["uvicorn", "goldenverba.server.api:app", "--host", "0.0.0.0", "--port", "8000"]
