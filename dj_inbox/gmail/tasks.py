# dj_inbox/gmail/tasks.py
"""Queue "import this company" for the private worker, through Cloud Tasks.

The public API never runs an import itself (Cloud Run throttles CPU after a response; an import takes minutes).
It adds a task; Cloud Tasks calls the worker with an OIDC token for the `dj-inbox-tasks` service account,
the only identity allowed to invoke the worker. The task body holds only the company id.

Settings (env): TASKS_QUEUE = projects/<p>/locations/<r>/queues/gmail-import, WORKER_URL = the worker's
https URL, TASKS_INVOKER = dj-inbox-tasks@<p>.iam.gserviceaccount.com. Missing -> enqueue logs and does nothing.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Callable

log = logging.getLogger("dj_inbox.api")
TASKS_URL = "https://cloudtasks.googleapis.com/v2/{queue}/tasks"
DISPATCH_DEADLINE = "1800s"                  # Cloud Tasks' maximum for HTTP targets: 30 minutes


def task_body(company_id: str, worker_url: str, invoker: str) -> dict:
    return {"task": {
        "dispatchDeadline": DISPATCH_DEADLINE,
        "httpRequest": {
            "httpMethod": "POST",
            "url": worker_url.rstrip("/") + "/tasks/import",
            "headers": {"Content-Type": "application/json"},
            "body": base64.b64encode(json.dumps({"company_id": company_id}).encode()).decode(),
            "oidcToken": {"serviceAccountEmail": invoker, "audience": worker_url.rstrip("/")},
        }}}


class CloudTasksQueue:
    def __init__(self, queue: str, worker_url: str, invoker: str, http):
        self.queue, self.worker_url, self.invoker, self.http = queue, worker_url, invoker, http

    def __call__(self, company_id: str) -> None:
        resp = self.http.post(TASKS_URL.format(queue=self.queue),
                              json=task_body(company_id, self.worker_url, self.invoker), timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Cloud Tasks refused the import task (HTTP {resp.status_code})")
        log.info(json.dumps({"event": "gmail_import_queued", "company_id": company_id}))


def enqueue_from_env() -> Callable[[str], None]:
    queue, worker, invoker = (os.environ.get(n, "").strip() for n in ("TASKS_QUEUE", "WORKER_URL", "TASKS_INVOKER"))
    if not (queue and worker and invoker):
        def log_only(company_id: str) -> None:
            log.info(json.dumps({"event": "gmail_import_requested", "company_id": company_id, "queued": False}))
        return log_only

    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return CloudTasksQueue(queue, worker, invoker, AuthorizedSession(creds))
