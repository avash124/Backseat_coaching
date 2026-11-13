import type { Metadata } from "next";

import { JobStatus } from "./job-status";

export const metadata: Metadata = { title: "Analyzing your game" };

export default async function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <h1 className="font-heading text-2xl font-semibold">Analyzing your game</h1>
      <JobStatus jobId={jobId} />
    </div>
  );
}
