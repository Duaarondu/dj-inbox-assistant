-- db/migrations/006_gmail_connections.sql
-- One Gmail connection per company: which mailbox, its refresh token (ENCRYPTED with Cloud KMS), import status.
--
-- Who may do what:
--   app_api     (public API)   reads the STATUS columns of its own company only, never token_ciphertext
--                              (column-level grant). Saves a connection only through save_gmail_connection().
--   app_ingest  (the worker)   reads its company's row INCLUDING the ciphertext (it alone may decrypt it, via
--                              KMS decrypt-only), and updates the import status / Gmail history id.
--
-- save_gmail_connection() follows 004's rules: who and which company come from app.user_id / app.company_id
-- (set by the API after verifying the sign-in token and the membership), never from arguments; only the
-- company's OWNER may connect; SECURITY DEFINER with a fixed search_path; EXECUTE for app_api only.
--
-- The ciphertext is useless without KMS, and KMS was asked to bind it to the company id (additional
-- authenticated data): a ciphertext copied into another company's row fails to decrypt.

create table if not exists gmail_connections (
    company_id      text        primary key references companies (company_id) on delete cascade,
    gmail_address   text        not null,
    token_ciphertext bytea      not null,
    kms_key         text        not null,          -- which key encrypted it (keys can be rotated later)
    history_id      text,                          -- Gmail's mailbox position, for "what's new since" (updates)
    status          text        not null check (status in ('importing', 'ready', 'error', 'disconnected')),
    error           text,                          -- short, safe message for the owner (never a token)
    connected_by    uuid        not null,
    connected_at    timestamptz not null default now(),
    last_import_at  timestamptz
);

-- A mailbox belongs to one company: connecting it to a second company is refused.
create unique index if not exists gmail_connections_one_company_per_mailbox
    on gmail_connections (lower(gmail_address));

alter table gmail_connections enable row level security;

grant select (company_id, gmail_address, history_id, status, error, connected_at, last_import_at)
    on gmail_connections to app_api;
grant select, update on gmail_connections to app_ingest;

drop policy if exists gmail_status_same_company on gmail_connections;
create policy gmail_status_same_company on gmail_connections for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );

drop policy if exists gmail_ingest_same_company on gmail_connections;
create policy gmail_ingest_same_company on gmail_connections for all to app_ingest
    using      ( company_id = nullif(current_setting('app.company_id', true), '') )
    with check ( company_id = nullif(current_setting('app.company_id', true), '') );

do $outer$
begin
    execute format($fn$
        create or replace function save_gmail_connection(address text, ciphertext bytea, key_name text,
                                                         start_history_id text)
        returns void
        language plpgsql
        security definer
        set search_path = %I, pg_temp
        as $body$
        declare
            uid uuid := nullif(current_setting('app.user_id', true), '')::uuid;
            cid text := nullif(current_setting('app.company_id', true), '');
        begin
            if uid is null or cid is null then
                raise exception 'not signed in' using errcode = '28000';
            end if;
            if not exists (select 1 from company_members
                            where user_id = uid and company_id = cid and role = 'owner') then
                raise exception 'only the company owner can connect Gmail' using errcode = '42501';
            end if;
            if address is null or position('@' in address) = 0 or ciphertext is null or length(ciphertext) = 0 then
                raise exception 'invalid connection' using errcode = '22023';
            end if;
            insert into gmail_connections (company_id, gmail_address, token_ciphertext, kms_key, history_id,
                                           status, error, connected_by, connected_at, last_import_at)
            values (cid, lower(address), ciphertext, key_name, start_history_id, 'importing', null, uid, now(), null)
            on conflict (company_id) do update
               set gmail_address = excluded.gmail_address, token_ciphertext = excluded.token_ciphertext,
                   kms_key = excluded.kms_key, history_id = excluded.history_id, status = 'importing',
                   error = null, connected_by = excluded.connected_by, connected_at = now();
        end;
        $body$
    $fn$, current_schema());
end
$outer$;

revoke all on function save_gmail_connection(text, bytea, text, text) from public;
grant execute on function save_gmail_connection(text, bytea, text, text) to app_api;
