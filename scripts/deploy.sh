#!/usr/bin/env bash
# scripts/deploy.sh: deploy the API to Cloud Run, always from the repo root and always to the right project.
#
#     bash scripts/deploy.sh
#
# Why (Bug Journal #20, Nov 17): `gcloud run deploy --source .` uploads whatever folder you're in and deploys
# to gcloud's current default project. One run started zipping the home folder; the next created a stray
# service in another project. This script pins the folder, the project and the branch.
set -euo pipefail

PROJECT="dj-inbox-assistant"
REGION="us-east1"
SERVICE="dj-inbox-api"

cd "$(dirname "$0")/.."                                    # the repo root, wherever you run this from

if [[ ! -f Dockerfile || ! -d dj_inbox || ! -d .git ]]; then
    echo "STOP: $(pwd) is not the dj-inbox-assistant repo." >&2; exit 1
fi
branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
    echo "STOP: production deploys only from main (you're on '$branch')." >&2; exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "STOP: uncommitted changes. Commit them first, so production matches a commit." >&2; exit 1
fi
git fetch -q origin main
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "STOP: local main differs from GitHub's main. Run: git pull (or push)." >&2; exit 1
fi
if ! gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format="value(name)" >/dev/null 2>&1; then
    echo "STOP: $SERVICE not found in $PROJECT/$REGION. Refusing to create a new service." >&2; exit 1
fi

echo "Deploying $(git rev-parse --short HEAD) from $(pwd) to $PROJECT / $REGION / $SERVICE"
gcloud run deploy "$SERVICE" --source . --region "$REGION" --project "$PROJECT" --quiet
