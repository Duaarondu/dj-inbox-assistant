-- db/migrations/007_company_domains.sql
-- Which email domains are "us" for a company, so the importer can tell our replies from clients' emails
-- (sender_role drives lead status: awaiting OUR reply vs waiting on the client).
--
-- The connected Gmail address always counts as the company (CompanyConfig.addresses). This adds the company's
-- own domain(s) for mail sent from other addresses, e.g. the demo mailbox (a gmail.com account) holds mail
-- sent from northshoresound.example. Never a shared domain like gmail.com: every client there would become
-- "the company". Set by hand for now (Supabase SQL editor); a settings page later.

alter table companies add column if not exists email_domains text[] not null default '{}';

alter table companies drop constraint if exists companies_no_shared_domains;
alter table companies add constraint companies_no_shared_domains check (
    not (email_domains && array['gmail.com', 'googlemail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                                'icloud.com', 'aol.com', 'live.com', 'msn.com', 'me.com']));
