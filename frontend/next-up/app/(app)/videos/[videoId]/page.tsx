import type { Metadata } from "next";

import { Results } from "./results";

export const metadata: Metadata = { title: "Game results" };

export default async function ResultsPage({ params }: { params: Promise<{ videoId: string }> }) {
  const { videoId } = await params;
  return (
    <div className="flex flex-col gap-6">
      <h1 className="font-heading text-2xl font-semibold">Game results</h1>
      <Results videoId={videoId} />
    </div>
  );
}
