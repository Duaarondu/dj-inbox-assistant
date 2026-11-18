#!/usr/bin/env bash
# scripts/deploy_worker.sh: deploy the PRIVATE worker (dj-inbox-worker) from the same image as the API.
#
#     bash scripts/deploy_worker.sh            (update the existing worker)
#     bash scripts/deploy_worker.sh --create   (first time only: allowed to create the service)
#
# Same guards as deploy.sh (Bug Journal #20): repo root, main, clean, in sync with GitHub, pinned project.
# The worker:
#   - is not public (--no-allow-unauthenticated): only dj-inbox-tasks (Cloud Tasks) gets run.invoker on it
#   - runs as dj-inbox-worker: KMS decrypt-only on the Gmail token key, Gemini, two secrets
#   - connects to Postgres as app_ingest (secret database-ingest-url): writes only the company of each transaction
#   - handles one import at a time per instance, up to 30 minutes (Cloud Tasks' longest deadline)
set -euo pipefail

PROJECT="dj-inbox-assistant"
REGION="us-east1"
SERVICE="dj-inbox-worker"
SA="dj-inbox-worker@${PROJECT}.iam.gserviceaccount.com"

cd "$(dirname "$0")/.."
if [[ ! -f Dockerfile || ! -d dj_inbox/worker || ! -d .git ]]; then
    echo "STOP: $(pwd) is not the dj-inbox-assistant repo (with the worker)." >&2; exit 1
fi
if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "STOP: deploy only from main." >&2; exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "STOP: uncommitted changes. Commit them first." >&2; exit 1
fi
git fetch -q origin main
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "STOP: local main differs from GitHub's main. Run: git pull (or push)." >&2; exit 1
fi
if ! gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format="value(name)" >/dev/null 2>&1; then
    if [[ "${1:-}" != "--create" ]]; then
        echo "STOP: $SERVICE doesn't exist. First deploy: bash scripts/deploy_worker.sh --create" >&2; exit 1
    fi
fi

# The Gmail client id is not secret; take it from the API's settings so the two can't drift apart.
CLIENT_ID="$(gcloud run services describe dj-inbox-api --region "$REGION" --project "$PROJECT" --format=json \
    | python3 -c 'import json,sys; env=json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0].get("env",[]); print(next((e.get("value","") for e in env if e["name"]=="GMAIL_CLIENT_ID"), ""))')"
if [[ ! "$CLIENT_ID" =~ ^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$ ]]; then
    echo "STOP: the API has no valid GMAIL_CLIENT_ID to share with the worker." >&2; exit 1
fi

echo "Deploying worker $(git rev-parse --short HEAD) to $PROJECT / $REGION / $SERVICE"
gcloud run deploy "$SERVICE" --source . --region "$REGION" --project "$PROJECT" --quiet \
    --service-account "$SA" \
    --no-allow-unauthenticated \
    --concurrency 1 --max-instances 2 --timeout 1800 --memory 1Gi --cpu 1 \
    --set-env-vars "APP_MODULE=dj_inbox.worker.app:app,GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=global,GMAIL_CLIENT_ID=${CLIENT_ID}" \
    --set-secrets "DATABASE_URL=database-ingest-url:latest,GMAIL_CLIENT_SECRET=gmail-client-secret:latest"
