FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Main app deps + PyTorch CPU
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

# LION venv — chemprop 1.6.1 needs numpy<2 for np.VisibleDeprecationWarning
RUN python -m venv /app/lion_env && \
    /app/lion_env/bin/pip install --no-cache-dir \
        "numpy<2" pandas scikit-learn rdkit scipy && \
    /app/lion_env/bin/pip install --no-cache-dir \
        torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    /app/lion_env/bin/pip install --no-cache-dir \
        tensorboard hyperopt flask typed-argument-parser && \
    /app/lion_env/bin/pip install --no-cache-dir chemprop==1.6.1 && \
    /app/lion_env/bin/python3 -c "from chemprop.train.make_predictions import make_predictions; print('chemprop OK')"

# ADMET venv
RUN python -m venv /app/admet_env && \
    /app/admet_env/bin/pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    /app/admet_env/bin/pip install --no-cache-dir \
        admet-ai numpy pandas

COPY . .

ENV LION_REPO=/app/lion_repo
ENV LION_VENV_PYTHON=/app/lion_env/bin/python3
ENV ADMET_VENV_PYTHON=/app/admet_env/bin/python3
ENV IAJD_OUT_DIR=/app/IAJD_master/bundles_caches
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
