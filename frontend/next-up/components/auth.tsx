"use client";

import type { Session } from "@supabase/supabase-js";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";

import { AUTH_ENABLED } from "@/lib/config";
import { getSupabase } from "@/lib/supabase";

interface AuthState {
  /** False when sign-in is turned off (NEXT_PUBLIC_AUTH_ENABLED is not "true"). */
  enabled: boolean;
  loading: boolean;
  session: Session | null;
}

const AuthContext = createContext<AuthState>({ enabled: false, loading: false, session: null });

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    enabled: AUTH_ENABLED,
    loading: AUTH_ENABLED,
    session: null,
  });

  useEffect(() => {
    if (!AUTH_ENABLED) return;
    // Fires straight away with the stored session, then on every change.
    const { data } = getSupabase().auth.onAuthStateChange((_event, session) => {
      setState({ enabled: true, loading: false, session });
    });
    return () => data.subscription.unsubscribe();
  }, []);

  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>;
}

/** Sends signed-out visitors to the sign-in page. Does nothing when sign-in is off. */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { enabled, loading, session } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const signedOut = enabled && !loading && !session;

  useEffect(() => {
    if (signedOut) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [signedOut, router, pathname]);

  if (enabled && !session) return <p className="text-muted-foreground">Loading…</p>;
  return children;
}
