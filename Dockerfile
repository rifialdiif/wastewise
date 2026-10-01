FROM python:3.12-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user models/mobilenetv2_waste_classifier.tflite ./models/
COPY --chown=user knowledge ./knowledge
COPY --chown=user config ./config

# Hosting platforms such as Render set PORT; default to 8000 elsewhere.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
