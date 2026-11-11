# web/: the demo page (Next.js on Vercel)

One page: ask the inbox a question (answer + the exact source emails), and the lead tracker.
It only talks to the FastAPI service on Cloud Run; it holds no data and no secrets.

## Run locally
```
npm install
NEXT_PUBLIC_API_URL=http://localhost:8080 npm run dev
```
(with the API running: `uvicorn dj_inbox.api.app:app --port 8080`)

## Deploy (Vercel)
- Import the GitHub repo, set **Root Directory = `web`**.
- Environment variable: `NEXT_PUBLIC_API_URL = https://dj-inbox-api-1090658566619.us-east1.run.app`
- Then add the Vercel URL to the API's `ALLOWED_ORIGINS` (Cloud Run env var), or the browser blocks the calls (CORS).

Email text is untrusted: it is rendered as plain text, never as HTML.
