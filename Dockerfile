# Dockerfile: the API for Cloud Run.  Built by Cloud Build with `gcloud run deploy --source .` (see README / chat).
# Synthetic demo data ships inside the image; real customer data will live in Postgres, never in an image.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the server runs.
COPY dj_inbox ./dj_inbox
COPY evals/__init__.py evals/run_baseline.py evals/score.py ./evals/
COPY synthetic_inbox/data/inbox.mbox ./synthetic_inbox/data/inbox.mbox
COPY data/processed/leads.jsonl ./data/processed/leads.jsonl
COPY data/embeddings/vectors.json ./data/embeddings/vectors.json

# Don't run as root.
RUN useradd --create-home app && chown -R app /app
USER app

# Cloud Run sets $PORT (8080 by default).
CMD ["sh", "-c", "exec uvicorn dj_inbox.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
