# One image for both services (API and dashboard); docker-compose picks the command.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf
WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
# bake the embedding model into the image, so the container needs no download at start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY reviews ./reviews
COPY ui ./ui
COPY demo ./demo
COPY models ./models
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 camp && chown -R camp /app/.hf
USER camp
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["uvicorn", "reviews.api:app", "--host", "0.0.0.0", "--port", "8000"]
