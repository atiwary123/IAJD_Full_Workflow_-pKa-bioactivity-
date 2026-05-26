FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "gradio>=5.0,<6" pandas "scikit-learn>=1.3" xgboost joblib openpyxl scipy rdkit

RUN pip install --no-cache-dir \
    chemprop==1.6.1 tensorboard hyperopt typed-argument-parser && \
    python -c "from chemprop.train.make_predictions import make_predictions; print('LION OK')"

RUN rm -rf /root/.cache/pip /tmp/*

FROM python:3.11-slim

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

ENV LION_REPO=/app/lion_repo
ENV LION_VENV_PYTHON=/usr/local/bin/python3
ENV ADMET_VENV_PYTHON=/usr/local/bin/python3
ENV IAJD_OUT_DIR=/app/IAJD_master/bundles_caches
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
