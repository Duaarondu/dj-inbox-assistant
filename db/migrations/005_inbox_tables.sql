-- db/migrations/005_inbox_tables.sql
-- Each company's inbox in Postgres: threads, emails, extracted leads. Until now only the vectors were here
-- (email_chunks, 001) and everything else came from files for the one demo company.
--
-- Two roles touch these tables:
--   app_api     the public API: READ ONLY (as before). Policies from 002 style: rows of app.company_id only.
--   app_ingest  the import worker: reads AND writes, but also only rows of app.company_id. A worker that is
--               fooled (or buggy) while importing company A can't write into company B: the WITH CHECK
--               clause rejects the row. Like app_api: no superuser, no BYPASSRLS, login given by hand.
--
-- Dedupe: an email's RFC Message-ID is unique per company (primary key), and Gmail's own message id too
-- (unique index), so a re-import or a replayed push notification can't create duplicates.
-- Lead STATUS (awaiting reply / cold / ...) is NOT stored: it depends on today's date (decision Oct 28);
-- it's computed when the leads are read. Only what the model extracted is stored.

do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'app_ingest') then
        create role app_ingest nologin noinherit nobypassrls;
    end if;
end
$$;

create table if not exists threads (
    company_id text not null references companies (company_id) on delete cascade,
    thread_id  text not null,                      -- Gmail's threadId for imported mail
    subject    text not null,
    primary key (company_id, thread_id)
);

create table if not exists emails (
    company_id   text        not null,
    message_id   text        not null,             -- RFC 5322 Message-ID header (or a hash when missing)
    gmail_id     text,                             -- Gmail's message id (null for mbox imports)
    thread_id    text        not null,
    sent_at      timestamptz not null,
    sent_offset  smallint    not null default 0,   -- the Date header's UTC offset in minutes (timestamptz drops
                                                   -- it): the model and the chunks see the sender's local time
    sender_role  text        not null check (sender_role in ('client', 'company')),
    sender_name  text        not null,
    sender_email text        not null,
    subject      text        not null,
    body_raw     text        not null,             -- shown to the owner, never sent to a model
    body_clean   text        not null,             -- quotes/signatures removed, contacts redacted
    contacts     jsonb       not null default '{}',
    primary key (company_id, message_id),
    foreign key (company_id, thread_id) references threads (company_id, thread_id) on delete cascade
);

create unique index if not exists emails_gmail_id on emails (company_id, gmail_id) where gmail_id is not null;
create index if not exists emails_thread on emails (company_id, thread_id, sent_at);

create table if not exists leads (
    company_id     text        not null,
    thread_id      text        not null,
    lead_index     smallint    not null check (lead_index >= 0),   -- 0 = the main event, 1 = a second event
    fields         jsonb       not null,           -- ExtractedLead as JSON (validated by Pydantic on write and read)
    source         text        not null check (source in ('email', 'website_form')),
    model          text        not null,
    prompt_version text        not null,
    extracted_at   timestamptz not null default now(),
    primary key (company_id, thread_id, lead_index),
    foreign key (company_id, thread_id) references threads (company_id, thread_id) on delete cascade
);

alter table threads enable row level security;
alter table emails  enable row level security;
alter table leads   enable row level security;

do $$
begin
    execute format('grant usage on schema %I to app_ingest', current_schema());
    if exists (select 1 from pg_namespace where nspname = 'extensions') then      -- Supabase: pgvector lives here
        grant usage on schema extensions to app_ingest;
        alter role app_ingest set search_path = public, extensions;
    end if;
end
$$;

grant select on threads, emails, leads to app_api;
grant select, insert, update, delete on threads, emails, leads, email_chunks to app_ingest;
grant select on companies to app_ingest;

-- app_api: read its own company's rows (same rule as 002).
drop policy if exists threads_same_company on threads;
create policy threads_same_company on threads for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists emails_same_company on emails;
create policy emails_same_company on emails for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists leads_same_company on leads;
create policy leads_same_company on leads for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );

-- app_ingest: read and write its own company's rows. USING filters what it can see/update/delete;
-- WITH CHECK rejects any row it tries to write for another company.
drop policy if exists threads_ingest on threads;
create policy threads_ingest on threads for all to app_ingest
    using      ( company_id = nullif(current_setting('app.company_id', true), '') )
    with check ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists emails_ingest on emails;
create policy emails_ingest on emails for all to app_ingest
    using      ( company_id = nullif(current_setting('app.company_id', true), '') )
    with check ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists leads_ingest on leads;
create policy leads_ingest on leads for all to app_ingest
    using      ( company_id = nullif(current_setting('app.company_id', true), '') )
    with check ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists chunks_ingest on email_chunks;
create policy chunks_ingest on email_chunks for all to app_ingest
    using      ( company_id = nullif(current_setting('app.company_id', true), '') )
    with check ( company_id = nullif(current_setting('app.company_id', true), '') );
drop policy if exists companies_ingest on companies;
create policy companies_ingest on companies for select to app_ingest
    using ( company_id = nullif(current_setting('app.company_id', true), '') );
