"use client";

import Link from "next/link";

import { useAuth } from "@/components/auth";
import { Button, buttonVariants } from "@/components/ui/button";
import { getSupabase } from "@/lib/supabase";

export function SiteHeader() {
  const { enabled, session } = useAuth();

  const signOut = async () => {
    await getSupabase().auth.signOut();
    // A full page load: it wins over the sign-in gate's own redirect and drops
    // everything cached for this coach.
    window.location.assign("/login");
  };

  return (
    <header className="border-b">
      <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between px-4">
        <Link href="/" className="font-heading text-lg font-semibold">
          Next-Up
        </Link>
        {enabled &&
          (session ? (
            <div className="flex items-center gap-3 text-sm">
              <span className="hidden text-muted-foreground sm:inline">{session.user.email}</span>
              <Button variant="outline" onClick={signOut}>
                Sign out
              </Button>
            </div>
          ) : (
            <Link href="/login" className={buttonVariants({ variant: "outline" })}>
              Sign in
            </Link>
          ))}
      </div>
    </header>
  );
}
