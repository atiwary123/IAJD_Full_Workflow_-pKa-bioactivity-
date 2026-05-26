FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Pin numpy<2 globally FIRST (chemprop 1.6.1 needs it)
RUN pip install --no-cache-dir "numpy<2"

# Main app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CPU (shared by all)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# torch-geometric (AGILE needs it) — skip scatter/sparse, not required for inference
RUN pip install --no-cache-dir torch-geometric

# chemprop 1.6.1 (LION) — install globally, no separate venv needed
RUN pip install --no-cache-dir tensorboard hyperopt typed-argument-parser && \
    pip install --no-cache-dir chemprop==1.6.1 && \
    python -c "from chemprop.train.make_predictions import make_predictions; print('chemprop OK')"

# admet-ai — install globally
RUN pip install --no-cache-dir admet-ai && \
    python -c "from admet_ai import ADMETModel; print('admet-ai OK')"

COPY . .

# Point LION/ADMET to the global python (no separate venvs needed)
ENV LION_REPO=/app/lion_repo
ENV LION_VENV_PYTHON=/usr/local/bin/python3
ENV ADMET_VENV_PYTHON=/usr/local/bin/python3
ENV IAJD_OUT_DIR=/app/IAJD_master/bundles_caches
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
