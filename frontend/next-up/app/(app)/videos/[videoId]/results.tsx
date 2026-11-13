"use client";

import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, errorMessage } from "@/lib/api";
import { formatTime, humanize } from "@/lib/format";
import { cn } from "@/lib/utils";

export function Results({ videoId }: { videoId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const results = useQuery({
    queryKey: ["results", videoId],
    queryFn: () => api.getResults(videoId),
    // A refetch could hand back a new video link and restart playback.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  if (results.isPending) return <p className="text-muted-foreground">Loading your game…</p>;
  if (results.isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>We couldn&apos;t load this game</AlertTitle>
        <AlertDescription>{errorMessage(results.error)}</AlertDescription>
      </Alert>
    );
  }

  const { video_url, plays, status } = results.data;

  const jumpTo = (index: number) => {
    setSelected(index);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = plays[index].start_s;
    video.play().catch(() => {}); // the coach can still press play
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <div className="flex flex-col gap-3">
        {video_url ? (
          <video
            ref={videoRef}
            src={video_url}
            controls
            playsInline
            preload="metadata"
            className="aspect-video w-full rounded-xl bg-black"
          >
            Your browser can&apos;t play this video.
          </video>
        ) : (
          <div className="flex aspect-video w-full items-center justify-center rounded-xl bg-muted p-6 text-center text-muted-foreground">
            The video isn&apos;t available to play right now.
          </div>
        )}
        {status !== "done" && (
          <p className="text-sm text-muted-foreground">
            This game is still being analyzed, so the list of plays may be incomplete.
          </p>
        )}
      </div>

      <Card size="sm">
        <CardHeader>
          <CardTitle>Plays ({plays.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {plays.length === 0 ? (
            <p className="text-muted-foreground">No plays were found in this game.</p>
          ) : (
            <ol className="flex flex-col gap-1">
              {plays.map((play, index) => (
                <li key={`${play.start_s}-${index}`}>
                  <button
                    type="button"
                    onClick={() => jumpTo(index)}
                    className={cn(
                      "flex w-full flex-col gap-1 rounded-lg px-3 py-2 text-left transition-colors outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50",
                      selected === index && "bg-muted",
                    )}
                  >
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="font-medium">
                        {play.play_type ? humanize(play.play_type) : "Play"}
                      </span>
                      <span className="text-sm text-muted-foreground tabular-nums">
                        {formatTime(play.start_s)} – {formatTime(play.end_s)}
                      </span>
                    </span>
                    <span className="flex flex-wrap gap-x-3 text-sm text-muted-foreground">
                      <span>{play.team === null ? "Team unknown" : `Team ${play.team}`}</span>
                      {play.outcome && <span>{humanize(play.outcome)}</span>}
                      {play.confidence !== null && (
                        <span>{Math.round(play.confidence * 100)}% sure</span>
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
