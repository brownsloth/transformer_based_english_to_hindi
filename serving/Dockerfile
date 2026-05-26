FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY serving/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/transformer_lib ./src/transformer_lib
COPY distil/student ./distil/student
COPY serving/api ./serving/api
COPY serving/config ./serving/config
COPY serving/download_artifacts.py ./serving/download_artifacts.py

ARG HF_ARTIFACTS_REPO=1starun8-research/en-hi-translation
ENV HF_ARTIFACTS_REPO=${HF_ARTIFACTS_REPO}
ENV SERVE_ARTIFACTS_DIR=/app/serving/artifacts
ENV PYTHONPATH=/app/src:/app
ENV SERVE_CONFIG=/app/serving/config/serve.yaml
ENV SERVE_DEVICE=cpu
ENV CORS_ORIGINS=https://projects.tarun-ssharma.com

# Pull ~242 MB weights from Hugging Face at build time (no local .pt copy needed).
RUN python serving/download_artifacts.py

EXPOSE 8000

WORKDIR /app/serving/api
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
