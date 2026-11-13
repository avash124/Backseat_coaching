import { createClient, type SupabaseClient } from "@supabase/supabase-js";

import { AUTH_ENABLED } from "@/lib/config";

let client: SupabaseClient | undefined;

// Created on first use so the app builds and runs without Supabase settings
// while sign-in is turned off.
export function getSupabase(): SupabaseClient {
  if (!client) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    if (!url || !anonKey) {
      throw new Error(
        "NEXT_PUBLIC_AUTH_ENABLED is true but NEXT_PUBLIC_SUPABASE_URL or " +
          "NEXT_PUBLIC_SUPABASE_ANON_KEY is not set.",
      );
    }
    client = createClient(url, anonKey);
  }
  return client;
}

export async function getAccessToken(): Promise<string | null> {
  if (!AUTH_ENABLED) return null;
  const { data } = await getSupabase().auth.getSession();
  return data.session?.access_token ?? null;
}
