-- db/migrations/002_companies_rls.sql
-- Accounts (companies + members) and the database-level company lock.
--
-- The API will connect as `app_api`: NOT the table owner and without BYPASSRLS, so every row it reads
-- passes the policies below. Each request runs in a transaction that starts with
--     select set_config('app.company_id', <company>, true)      -- true = only for this transaction
-- and the policies compare each row's company_id with that setting. A query that forgets
-- "where company_id = ..." still sees only that company's rows; a request that forgets set_config sees
-- nothing (NULL never equals anything: fail closed). Transaction-local settings matter because Supabase's
-- pooler hands one connection to many requests.
--
-- No password here (no secrets in git): in production, give the role a login once, by hand, in the
-- Supabase SQL editor:  alter role app_api with login password '<from your password manager>';

-- Roles are cluster-wide (tests run this file once per temporary schema), so create it only once.
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'app_api') then
        create role app_api nologin noinherit nobypassrls;
    end if;
end
$$;

create table if not exists companies (
    company_id text        primary key,          -- 'northshore' = the public demo; new ones get a random id
    name       text        not null check (length(name) between 1 and 120),
    created_at timestamptz not null default now()
);

insert into companies (company_id, name)
values ('northshore', 'Northshore Sound Co. (demo)')
on conflict (company_id) do nothing;

create table if not exists company_members (
    user_id    uuid        not null,             -- Supabase Auth user id (the JWT's "sub")
    company_id text        not null references companies (company_id) on delete cascade,
    role       text        not null check (role in ('owner', 'staff')),
    created_at timestamptz not null default now(),
    primary key (user_id, company_id)
);

-- On Supabase, tie members to real users (deleting a user removes their memberships). Plain Postgres
-- (CI, local tests) has no auth schema, so the foreign key is added only where auth.users exists.
do $$
begin
    if to_regclass('auth.users') is not null
       and not exists (select 1 from pg_constraint where conname = 'company_members_user_fk'
                       and conrelid = 'company_members'::regclass) then
        alter table company_members
            add constraint company_members_user_fk foreign key (user_id) references auth.users (id) on delete cascade;
    end if;
end
$$;

alter table companies       enable row level security;
alter table company_members enable row level security;
-- email_chunks already has RLS on (001). Without a policy for a role, that role sees no rows.

-- What app_api may do: read, nothing else (writes come later, through narrow functions).
do $$
begin
    execute format('grant usage on schema %I to app_api', current_schema());
end
$$;
grant select on email_chunks, companies, company_members to app_api;

-- The company lock. (Aaron's policies.)
drop policy if exists chunks_same_company on email_chunks;
create policy chunks_same_company on email_chunks
    for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );

drop policy if exists companies_same_company on companies;
create policy companies_same_company on companies
    for select to app_api
    using ( company_id = nullif(current_setting('app.company_id', true), '') );

-- Before the company is known, the API sets app.user_id to find the user's memberships.
drop policy if exists members_own_rows on company_members;
create policy members_own_rows on company_members
    for select to app_api
    using ( user_id = nullif(current_setting('app.user_id', true), '')::uuid );
