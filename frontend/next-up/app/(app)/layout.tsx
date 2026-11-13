import { RequireAuth } from "@/components/auth";

// Every page in this group needs a signed-in coach when sign-in is turned on.
export default function SignedInLayout({ children }: { children: React.ReactNode }) {
  return <RequireAuth>{children}</RequireAuth>;
}
