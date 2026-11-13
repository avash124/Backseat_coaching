import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";

export default function HomePage() {
  return (
    <section className="mx-auto flex max-w-2xl flex-col items-start gap-6 py-12">
      <h1 className="font-heading text-4xl font-semibold tracking-tight">
        Break down your game film
      </h1>
      <p className="text-lg text-muted-foreground">
        Upload a game video and Next-Up lists every play: when it happened, which team had the
        ball, what they ran, and how it ended. Click any play to jump straight to it.
      </p>
      <Link href="/upload" className={buttonVariants({ size: "lg" })}>
        Upload a game
      </Link>
    </section>
  );
}
