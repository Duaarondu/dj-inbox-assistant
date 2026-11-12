-- db/migrations/003_app_api_extensions.sql
-- Supabase installs pgvector in the `extensions` schema. app_api needs to see it there for the `vector`
-- type and the <=> operator, or every search fails once the API connects as app_api.
-- Plain Postgres (CI, tests) has no `extensions` schema: then this does nothing.
do $$
begin
    if exists (select 1 from pg_namespace where nspname = 'extensions') then
        grant usage on schema extensions to app_api;
        alter role app_api set search_path = public, extensions;
    end if;
end
$$;
