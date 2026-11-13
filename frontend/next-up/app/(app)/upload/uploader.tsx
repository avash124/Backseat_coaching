"use client";

import { useMutation } from "@tanstack/react-query";
import { FileVideo, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ApiError, api, uploadErrorMessage } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  isAbort,
  putPart,
  sleep,
  startUpload,
  storageKey,
  type UploadHandle,
} from "@/lib/upload";

export function Uploader() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [sentBytes, setSentBytes] = useState(0);
  const [resuming, setResuming] = useState(false);
  const [dragging, setDragging] = useState(false);
  const handle = useRef<UploadHandle | null>(null);

  const upload = useMutation({
    mutationFn: (file: File) => {
      const h = startUpload(file, { api, putPart, sleep, storage: localStorage }, setSentBytes);
      handle.current = h;
      return h.result;
    },
    onSuccess: ({ jobId }) => router.push(`/jobs/${jobId}`),
  });

  const cancel = useMutation({
    mutationFn: () => handle.current?.cancel() ?? Promise.resolve(),
    onSettled: () => {
      handle.current = null;
      upload.reset();
      setFile(null);
    },
  });

  // Leaving the page pauses the upload; choosing the same file again resumes it.
  useEffect(() => () => handle.current?.stop(), []);

  useEffect(() => {
    if (!upload.isPending) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [upload.isPending]);

  const choose = (chosen: File | undefined) => {
    if (!chosen || upload.isPending) return;
    setFile(chosen);
    setSentBytes(0);
    setResuming(localStorage.getItem(storageKey(chosen)) !== null);
    cancel.reset();
    upload.mutate(chosen);
  };

  const reset = () => {
    upload.reset();
    setFile(null);
  };

  if (!file) {
    return (
      <div className="flex flex-col gap-4">
        <label
          htmlFor="video-file"
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            choose(event.dataTransfer.files[0]);
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors hover:bg-muted/50 has-focus-visible:ring-3 has-focus-visible:ring-ring/50",
            dragging && "border-primary bg-muted/50",
          )}
        >
          <Upload className="size-8 text-muted-foreground" />
          <span className="font-medium">Drag your game video here, or click to choose a file</span>
          <span className="text-sm text-muted-foreground">MP4 or MOV, one game at a time</span>
          <input
            id="video-file"
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov"
            className="sr-only"
            onChange={(event) => {
              choose(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </label>
        {cancel.isError && (
          <Alert variant="destructive">
            <AlertDescription>
              We stopped the upload, but couldn&apos;t tell the server to discard it. You can
              ignore this.
            </AlertDescription>
          </Alert>
        )}
      </div>
    );
  }

  const percent = file.size > 0 ? Math.floor((sentBytes / file.size) * 100) : 0;
  const finishing = upload.isPending && sentBytes >= file.size;
  const failed = upload.isError && !isAbort(upload.error) && !cancel.isPending;
  // Retrying won't help when the server has rejected the file itself.
  const canRetry = !(upload.error instanceof ApiError && [413, 415, 422].includes(upload.error.status));

  return (
    <Card>
      <CardContent className="flex flex-col gap-5">
        <div className="flex items-center gap-3">
          <FileVideo className="size-6 shrink-0 text-muted-foreground" />
          <div className="min-w-0">
            <p className="truncate font-medium">{file.name}</p>
            <p className="text-sm text-muted-foreground">{formatBytes(file.size)}</p>
          </div>
        </div>

        {!failed && (
          <div className="flex flex-col gap-2">
            <Progress value={percent} aria-label="Upload progress" />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>
                {cancel.isPending
                  ? "Cancelling…"
                  : finishing || upload.isSuccess
                    ? "Finishing up…"
                    : resuming
                      ? "Picking up where you left off…"
                      : "Uploading…"}
              </span>
              <span className="tabular-nums">
                {formatBytes(sentBytes)} of {formatBytes(file.size)} ({percent}%)
              </span>
            </div>
          </div>
        )}

        {upload.isPending && !cancel.isPending && (
          <p className="text-sm text-muted-foreground">
            Keep this page open until the upload finishes. If your connection drops or you close
            the page, choose the same video again and it will continue from where it stopped.
          </p>
        )}

        {failed && (
          <Alert variant="destructive">
            <AlertTitle>The upload didn&apos;t finish</AlertTitle>
            <AlertDescription>{uploadErrorMessage(upload.error)}</AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2">
          {upload.isPending && (
            <Button variant="outline" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
              Cancel upload
            </Button>
          )}
          {failed && canRetry && <Button onClick={() => upload.mutate(file)}>Try again</Button>}
          {failed && (
            <Button variant="outline" onClick={reset}>
              Choose a different video
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
