"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ApiError, api, errorMessage } from "@/lib/api";

const POLL_MS = 2000;

export function JobStatus({ jobId }: { jobId: string }) {
  const router = useRouter();
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "done" || status === "failed") return false;
      // Keep polling through network blips and server errors, but not refusals.
      const error = query.state.error;
      if (error instanceof ApiError && error.status < 500) return false;
      return POLL_MS;
    },
  });

  const status = job.data?.status;
  const videoId = job.data?.video_id;
  useEffect(() => {
    if (status === "done" && videoId) router.replace(`/videos/${videoId}`);
  }, [status, videoId, router]);

  const refused = job.error instanceof ApiError && job.error.status < 500;
  if (job.isError && (refused || !job.data)) {
    return (
      <Alert variant="destructive">
        <AlertTitle>We couldn&apos;t check on this game</AlertTitle>
        <AlertDescription>{errorMessage(job.error)}</AlertDescription>
      </Alert>
    );
  }

  if (status === "failed") {
    return (
      <div className="flex flex-col gap-4">
        <Alert variant="destructive">
          <AlertTitle>We couldn&apos;t analyze this game</AlertTitle>
          <AlertDescription>
            {job.data?.error || "Something went wrong while analyzing the video."}
          </AlertDescription>
        </Alert>
        <Link href="/upload" className={buttonVariants({ className: "self-start" })}>
          Upload a game
        </Link>
      </div>
    );
  }

  const message =
    status === "processing"
      ? "Finding the plays in your game. This page will update on its own."
      : status === "done"
        ? "All done. Opening your results…"
        : "Your game is in line to be analyzed. This page will update on its own.";

  return (
    <Card>
      <CardContent className="flex items-center gap-3">
        <Loader2 className="size-5 shrink-0 animate-spin text-muted-foreground" />
        <p>{job.isPending ? "Checking on your game…" : message}</p>
      </CardContent>
    </Card>
  );
}
