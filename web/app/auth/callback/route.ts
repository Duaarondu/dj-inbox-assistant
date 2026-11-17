// web/app/auth/callback/route.ts: Google sends the user back here (via Supabase) with a one-time ?code.
// We swap the code for a session (PKCE: the verifier cookie was set by the browser client) and store the
// session in cookies, then go to /setup. This URL must be in Supabase's redirect allow list exactly.
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { NextResponse, type NextRequest } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams, origin } = request.nextUrl;
  const code = searchParams.get("code");
  if (!code) {
    // Supabase/Google send ?error=...&error_description=... instead of a code when sign-in fails.
    // Pass the description through so the sign-in page can show it (rendered as plain text).
    const why = searchParams.get("error_description") ?? searchParams.get("error");
    const to = new URL("/signin", origin);
    to.searchParams.set("error", "missing_code");
    if (why) to.searchParams.set("why", why.slice(0, 200));
    return NextResponse.redirect(to);
  }

  const cookieStore = await cookies();
  const client = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!,
    {
      cookies: {
        getAll: () => cookieStore.getAll(),
        setAll: (toSet) => toSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options)),
      },
    },
  );

  const { error } = await client.auth.exchangeCodeForSession(code);
  // Fixed destination, never taken from the URL: no open redirect.
  return NextResponse.redirect(error ? `${origin}/signin?error=sign_in_failed` : `${origin}/setup`);
}
