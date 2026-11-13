"use client";

import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/components/auth";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getSupabase } from "@/lib/supabase";

type Mode = "sign-in" | "sign-up";

export function LoginForm() {
  const { enabled, session } = useAuth();
  const router = useRouter();
  const requested = useSearchParams().get("next");
  // Only follow same-site paths, never "//other.site" or "/\other.site".
  const next = requested && /^\/(?![/\\])/.test(requested) ? requested : "/upload";

  const [mode, setMode] = useState<Mode>("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [checkEmail, setCheckEmail] = useState(false);

  useEffect(() => {
    if (session) router.replace(next);
  }, [session, next, router]);

  const submit = useMutation({
    mutationFn: async () => {
      const auth = getSupabase().auth;
      const { data, error } =
        mode === "sign-in"
          ? await auth.signInWithPassword({ email, password })
          : await auth.signUp({ email, password });
      if (error) throw error;
      return data.session;
    },
    // With a session, the effect above moves on. Without one, the project
    // wants the email address confirmed first.
    onSuccess: (newSession) => setCheckEmail(!newSession),
  });

  if (!enabled) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No sign-in needed</CardTitle>
          <CardDescription>Sign-in is turned off on this server.</CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/upload" className={buttonVariants()}>
            Upload a game
          </Link>
        </CardContent>
      </Card>
    );
  }

  const switchMode = () => {
    setMode(mode === "sign-in" ? "sign-up" : "sign-in");
    setCheckEmail(false);
    submit.reset();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{mode === "sign-in" ? "Sign in" : "Create an account"}</CardTitle>
        <CardDescription>
          {mode === "sign-in"
            ? "Sign in to upload games and see your results."
            : "Use your email address and choose a password."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            submit.mutate();
          }}
        >
          <div className="flex flex-col gap-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              autoComplete={mode === "sign-in" ? "current-password" : "new-password"}
              required
              minLength={mode === "sign-up" ? 6 : undefined}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </div>

          {submit.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {submit.error.name === "AuthRetryableFetchError"
                  ? "We couldn't reach the sign-in service. Check your internet connection and try again."
                  : submit.error.message || "We couldn't sign you in. Please try again."}
              </AlertDescription>
            </Alert>
          )}
          {checkEmail && (
            <Alert>
              <AlertDescription>
                Check your email for a link to confirm your account, then sign in here.
              </AlertDescription>
            </Alert>
          )}

          <Button type="submit" disabled={submit.isPending}>
            {submit.isPending ? "Please wait…" : mode === "sign-in" ? "Sign in" : "Create account"}
          </Button>
          <Button type="button" variant="link" onClick={switchMode}>
            {mode === "sign-in" ? "New here? Create an account" : "Already have an account? Sign in"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
