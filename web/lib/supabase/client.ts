// web/lib/supabase/client.ts: the one Supabase client the browser uses (sign-in and the session only).
// The browser never reads company data from Supabase: data goes through FastAPI with the access token.
// The publishable key is public by design (RLS + no grants to anon), so NEXT_PUBLIC_ is fine here.
import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (!client) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
    if (!url || !key) throw new Error("Sign-in isn't configured (NEXT_PUBLIC_SUPABASE_URL / _PUBLISHABLE_KEY).");
    client = createBrowserClient(url, key);
  }
  return client;
}

/** The current access token, refreshed by supabase-js if it's about to expire. null = signed out. */
export async function accessToken(): Promise<string | null> {
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}
