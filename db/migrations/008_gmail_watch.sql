-- db/migrations/008_gmail_watch.sql
-- Automatic updates (milestone 5): Gmail watch -> Pub/Sub push -> worker -> history.list from history_id.
--
-- 1. gmail_connections.watch_expires_at: when Gmail stops sending notifications for this mailbox (a watch lasts
--    ~7 days; Cloud Scheduler renews every connected mailbox daily).
-- 2. emails.imported_at: when the row was written. imported_at - sent_at = how fast a new email showed up
--    (the milestone's success bar: median < 30 s).
-- 3. Two lookups the worker needs BEFORE it knows the company, so RLS (app.company_id) can't help:
--      gmail_company_for(address)   a push notification names only the mailbox
--      gmail_companies_to_watch()   the daily renewal walks every connected mailbox
--    Both are SECURITY DEFINER, return company ids only (never tokens), and only app_ingest may call them.

alter table gmail_connections add column if not exists watch_expires_at timestamptz;
grant select (watch_expires_at) on gmail_connections to app_api;

alter table emails add column if not exists imported_at timestamptz default now();

do $outer$
begin
    execute format($fn$
        create or replace function gmail_company_for(address text)
        returns text
        language sql
        stable
        security definer
        set search_path = %I, pg_temp
        as $body$
            select company_id from gmail_connections
             where gmail_address = lower(address) and status <> 'disconnected'
        $body$
    $fn$, current_schema());

    execute format($fn$
        create or replace function gmail_companies_to_watch()
        returns setof text
        language sql
        stable
        security definer
        set search_path = %I, pg_temp
        as $body$
            select company_id from gmail_connections where status in ('ready', 'importing') order by company_id
        $body$
    $fn$, current_schema());
end
$outer$;

revoke all on function gmail_company_for(text) from public;
revoke all on function gmail_companies_to_watch() from public;
grant execute on function gmail_company_for(text) to app_ingest;
grant execute on function gmail_companies_to_watch() to app_ingest;
