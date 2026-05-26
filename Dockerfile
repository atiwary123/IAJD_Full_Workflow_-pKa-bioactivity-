FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ── Main app deps ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir --no-build-isolation torch-scatter torch-sparse

# ── LION venv (chemprop 1.6.1) ──
RUN python -m venv /app/lion_env && \
    /app/lion_env/bin/pip install --no-cache-dir \
        chemprop==1.6.1 \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    /app/lion_env/bin/pip install --no-cache-dir \
        numpy pandas scikit-learn rdkit

# ── ADMET venv (admet-ai) ──
RUN python -m venv /app/admet_env && \
    /app/admet_env/bin/pip install --no-cache-dir \
        admet-ai \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    /app/admet_env/bin/pip install --no-cache-dir \
        numpy pandas

# Copy app code
COPY . .

# Set env vars for LION/ADMET paths
ENV LION_REPO=/app/lion_repo
ENV LION_VENV_PYTHON=/app/lion_env/bin/python3
ENV ADMET_VENV_PYTHON=/app/admet_env/bin/python3
ENV IAJD_OUT_DIR=/app/IAJD_master/bundles_caches
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
