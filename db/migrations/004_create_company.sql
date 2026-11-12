-- db/migrations/004_create_company.sql
-- Onboarding: the ONE write app_api may do. A signed-in user with no company creates one and becomes its owner.
--
-- Why a function and not INSERT rights: with INSERT on company_members, even a policy "only rows for yourself"
-- lets a user add THEMSELVES to ANY existing company ('northshore', a competitor) and read its email.
-- This function decides everything itself:
--   - who: app.user_id, set by the API after it verified the sign-in token (never an argument)
--   - which company: a new random id (never an argument)
--   - at most one owned company per user (unique index, so two simultaneous clicks can't make two)
--
-- SECURITY DEFINER = runs with the rights of its owner (the migration runner). Two rules that make that safe:
--   - a FIXED search_path, so nobody can slip in their own "companies" table or function via theirs
--   - EXECUTE revoked from everyone, then granted to app_api only

create unique index if not exists one_owned_company_per_user
    on company_members (user_id) where role = 'owner';

do $outer$
begin
    -- search_path pinned to the schema this migration runs in (public in production, a temp schema in tests).
    execute format($fn$
        create or replace function create_company(company_name text)
        returns text
        language plpgsql
        security definer
        set search_path = %I, pg_temp
        as $body$
        declare
            uid    uuid := nullif(current_setting('app.user_id', true), '')::uuid;
            clean  text := btrim(company_name);
            new_id text;
        begin
            if uid is null then
                raise exception 'not signed in' using errcode = '28000';
            end if;
            if clean is null or length(clean) not between 1 and 120 then
                raise exception 'company name must be 1 to 120 characters' using errcode = '22023';
            end if;
            if exists (select 1 from company_members where user_id = uid and role = 'owner') then
                raise exception 'you already have a company' using errcode = '23505';
            end if;
            new_id := 'c_' || substr(replace(gen_random_uuid()::text, '-', ''), 1, 16);
            insert into companies (company_id, name) values (new_id, clean);
            insert into company_members (user_id, company_id, role) values (uid, new_id, 'owner');
            return new_id;
        end;
        $body$
    $fn$, current_schema());
end
$outer$;

revoke all on function create_company(text) from public;
grant execute on function create_company(text) to app_api;
