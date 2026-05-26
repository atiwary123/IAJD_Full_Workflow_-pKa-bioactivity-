FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Main app: gradio + rdkit + xgboost + torch + torch-geometric
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric

# LION: chemprop 1.6.1 in its own venv (conflicts with admet-ai)
RUN python -m venv --system-site-packages /app/lion_env && \
    /app/lion_env/bin/pip install --no-cache-dir \
        chemprop==1.6.1 tensorboard hyperopt typed-argument-parser && \
    /app/lion_env/bin/python3 -c \
        "from chemprop.train.make_predictions import make_predictions; print('LION OK')"

# ADMET: admet-ai in its own venv (conflicts with chemprop)
RUN python -m venv --system-site-packages /app/admet_env && \
    /app/admet_env/bin/pip install --no-cache-dir admet-ai && \
    /app/admet_env/bin/python3 -c \
        "from admet_ai import ADMETModel; print('ADMET OK')"

COPY . .

ENV LION_REPO=/app/lion_repo
ENV LION_VENV_PYTHON=/app/lion_env/bin/python3
ENV ADMET_VENV_PYTHON=/app/admet_env/bin/python3
ENV IAJD_OUT_DIR=/app/IAJD_master/bundles_caches
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
