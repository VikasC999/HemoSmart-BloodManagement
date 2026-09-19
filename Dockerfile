# HemoSmart backend -- single container, no LSTM sidecar (see README's
# "what's simulated vs real": the LSTM forecaster needs a separate
# TensorFlow venv, which isn't worth containerizing for a free-tier
# single-service host. HEMOSMART_LSTM_PYTHON stays unset here, so
# _forecast_with_lstm() returns None and Prophet -- already trained,
# already on disk -- serves every forecast.

FROM python:3.11-slim

WORKDIR /app

# libgomp1: XGBoost's compiled extension needs OpenMP at runtime on
# Linux, the same way it needed `brew install libomp` on macOS -- this
# is the Debian-slim equivalent of that same dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt

# Install CPU-only torch FIRST -- sentence-transformers (via rag/explain.py's
# HuggingFace embeddings) and crewai's dependency tree both pull in torch,
# and pip's default wheel is the CUDA/GPU build: several GB of nvidia-*
# packages this container never uses (CPU-only inference throughout).
# Satisfying torch here first means the later install finds it already
# present and never reaches for the GPU wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r backend/requirements.txt

# Hugging Face Spaces runs containers as uid 1000 (and other hosts are
# better off not running as root either), so create that user and give
# it a writable HOME.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface
USER user

# Bake the RAG embedding model (rag/explain.py's EMBEDDING_MODEL) into
# the image. Otherwise the first request after every cold start would
# download it from the Hub -- slow, and it needs a writable cache dir.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
