# EHR/Wearables → FHIR R4 pipeline + dashboard
# Single-stage image. Default command runs the Flask dashboard on port 8000;
# the CLI (main.py) can be run by overriding the command.

FROM python:3.11-slim

# System deps: build-essential + python3-dev are needed to build chromadb's
# hnswlib extension. Removed from the final layer to keep the image slim.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# build-essential/python3-dev: build chromadb's hnswlib extension.
# default-jre-headless: required by the official HL7 FHIR validator (Layer 1).
# curl: fetches the validator jar below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential python3-dev default-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*

# Official HL7 FHIR validator (reference implementation) used for conformance
# validation. Baked in so the container validates offline; set
# USE_OFFICIAL_VALIDATOR=false to skip Layer 1 and use the built-in checks.
RUN mkdir -p /app/tools \
    && curl -sSL -o /app/tools/validator_cli.jar \
       https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar

# Install the CPU-only build of torch FIRST so sentence-transformers does not
# pull the much larger default (CUDA) wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Python deps (cached separately from source for faster rebuilds).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application source. The .dockerignore keeps out venv/, .git/, generated
# artifacts and the bulky sample datasets; only the small terminology seeds
# under data/terminology are copied so the vector store can build on first run.
COPY . .

# Run as a non-root user; give it ownership of the writable runtime dirs.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/output /app/.chromadb /app/overrides \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Lightweight liveness check against the dashboard's /models endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/models')" || exit 1

# Default: the dashboard. Override to run the CLI, e.g.
#   docker run --rm --env-file .env fhir-pipeline python main.py --dataset ace --clusters all
CMD ["python", "server.py"]
