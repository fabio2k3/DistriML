# Imagen única de DistriML El mismo contenedor sirve de nodo head o de nodo worker
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAY_USAGE_STATS_ENABLED=0

WORKDIR /app

# Capa de dependencias: solo se reconstruye si cambia pyproject.toml
COPY pyproject.toml ./
RUN mkdir -p distriml && touch distriml/__init__.py \
    && pip install ".[dev]" \
    && pip uninstall -y distriml

# Código del proyecto (instalación editable: el paquete se importa desde /app en todos los procesos de Ray)
COPY . .
RUN pip install --no-deps -e .

# Datos de la BD (CSV) y artefactos: se monta un volumen aquí en el contenedor head
VOLUME ["/data"]
EXPOSE 6379 8080

# Sin ENTRYPOINT: el rol (head/worker) lo decide el comando de `docker run`
CMD ["python", "-c", "print('DistriML: indica un comando, p. ej. ray start --head ... (ver docs/DOCKER.md)')"]
