-- db/migrations/001_email_chunks.sql
-- Email chunks + their embeddings in Postgres (pgvector). Replaces data/embeddings/vectors.json + NumPy.

create extension if not exists vector;

create table if not exists email_chunks (
    company_id text        not null,              -- every row belongs to one company; every query filters on it
    chunk_id   text        not null,              -- "<thread_id>-<n>": what answers cite
    thread_id  text        not null,
    message_id text        not null,
    sent_at    timestamptz not null,
    text       text        not null,              -- header line + cleaned, redacted body (what was embedded)
    embedding  vector(768) not null,              -- gemini-embedding-2, task RETRIEVAL_DOCUMENT
    primary key (company_id, chunk_id)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
create index if not exists email_chunks_embedding_hnsw
    on email_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists email_chunks_company_thread on email_chunks (company_id, thread_id);

-- Supabase publishes tables in the public schema through its REST API (with the public "anon" key).
-- Row-level security with NO policies makes that API return nothing. Our server connects as the table's
-- owner, which bypasses RLS, and filters every query on company_id itself. Real per-user policies come
-- with Google sign-in (after Nov 14).
alter table email_chunks enable row level security;
