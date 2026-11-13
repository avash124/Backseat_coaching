import { ApiError, type api, type UploadStatus } from "@/lib/api";

export const CONCURRENCY = 4;
export const MAX_ATTEMPTS = 5;
/** Upload links requested per call (the server allows up to 100). */
export const LINK_BATCH = 10;
const LINK_MAX_AGE_MS = 20 * 60_000;
const STALL_TIMEOUT_MS = 60_000;

/** Sends one slice of the file to storage. Rejects on any failure. */
export type PutPart = (
  url: string,
  body: Blob,
  signal: AbortSignal,
  onProgress: (sentBytes: number) => void,
) => Promise<void>;

export interface UploadDeps {
  api: Pick<
    typeof api,
    "createUpload" | "getUploadStatus" | "getPartUrls" | "confirmUpload" | "cancelUpload"
  >;
  putPart: PutPart;
  storage: Pick<Storage, "getItem" | "setItem" | "removeItem">;
  sleep: (ms: number, signal: AbortSignal) => Promise<void>;
}

export interface UploadResult {
  videoId: string;
  jobId: string;
}

export interface UploadHandle {
  result: Promise<UploadResult>;
  /** Stop sending. The upload stays on the server and can be resumed later. */
  stop(): void;
  /** Stop sending and delete the upload from the server. */
  cancel(): Promise<void>;
}

/** Identifies a file across page reloads, so an interrupted upload can resume. */
export function storageKey(file: File): string {
  return `nextup:upload:${file.name}:${file.size}:${file.lastModified}`;
}

/** Byte range [start, end) of a 1-based part. The last part may be short. */
export function partRange(partNumber: number, partSize: number, fileSize: number) {
  return {
    start: (partNumber - 1) * partSize,
    end: Math.min(partNumber * partSize, fileSize),
  };
}

/** Parts the server does not have yet, or has at the wrong size. */
export function missingParts(fileSize: number, status: UploadStatus): number[] {
  const sizes = new Map(status.uploaded.map((p) => [p.part_number, p.size_bytes]));
  const missing: number[] = [];
  for (let n = 1; n <= status.part_count; n++) {
    const { start, end } = partRange(n, status.part_size, fileSize);
    if (sizes.get(n) !== end - start) missing.push(n);
  }
  return missing;
}

export function backoffMs(attempt: number): number {
  return Math.min(1000 * 2 ** (attempt - 1), 30_000);
}

// Browsers report an empty type for some video files, so fall back to the
// extension. The server makes the real decision.
export function contentType(file: File): string {
  if (file.type) return file.type;
  const name = file.name.toLowerCase();
  if (name.endsWith(".mp4") || name.endsWith(".m4v")) return "video/mp4";
  if (name.endsWith(".mov")) return "video/quicktime";
  return "application/octet-stream";
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

// A request the server understood and refused will be refused again.
function isPermanent(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  );
}

export function startUpload(
  file: File,
  deps: UploadDeps,
  onProgress: (uploadedBytes: number) => void,
): UploadHandle {
  const controller = new AbortController();
  const result = run(file, deps, controller.signal, onProgress).catch((error) => {
    controller.abort(); // stop the parts still in flight
    throw error;
  });
  return {
    result,
    stop: () => controller.abort(),
    cancel: async () => {
      controller.abort();
      await result.catch(() => {}); // settle first, so the video id is saved if one was issued
      const key = storageKey(file);
      const videoId = deps.storage.getItem(key);
      if (videoId) {
        deps.storage.removeItem(key);
        await deps.api.cancelUpload(videoId);
      }
    },
  };
}

async function run(
  file: File,
  deps: UploadDeps,
  signal: AbortSignal,
  onProgress: (uploadedBytes: number) => void,
): Promise<UploadResult> {
  const { api, storage } = deps;
  const key = storageKey(file);

  let videoId = storage.getItem(key);
  let status: UploadStatus | undefined;
  if (videoId) {
    try {
      status = await api.getUploadStatus(videoId);
    } catch (error) {
      if (!(error instanceof ApiError && (error.status === 404 || error.status === 409))) {
        throw error;
      }
      // 409: the upload is no longer in progress. If an earlier confirm got through but
      // its answer was lost, confirming again returns that job: don't re-send the file.
      if (error.status === 409) {
        const job = await api.confirmUpload(videoId).catch((e) => {
          if (e instanceof ApiError && e.status === 409) return null;
          throw e;
        });
        if (job) {
          storage.removeItem(key);
          return { videoId, jobId: job.job_id };
        }
      }
      // The saved upload no longer exists (or is not ours): start a new one.
      storage.removeItem(key);
    }
  }
  if (!videoId || !status) {
    const session = await api.createUpload({
      filename: file.name,
      content_type: contentType(file),
      size_bytes: file.size,
    });
    videoId = session.video_id;
    storage.setItem(key, videoId);
    status = { part_size: session.part_size, part_count: session.part_count, uploaded: [] };
  }
  signal.throwIfAborted();

  await uploadParts(file, videoId, status, deps, signal, onProgress);
  signal.throwIfAborted();

  // Confirming twice is safe: the server answers with the same job.
  try {
    const { job_id } = await api.confirmUpload(videoId);
    storage.removeItem(key);
    return { videoId, jobId: job_id };
  } catch (error) {
    // 422: the file is not a video, so forget it rather than resume into the
    // same rejection. 409 (a part missing or the wrong size) stays resumable:
    // trying again asks which parts the server has and sends the rest.
    if (error instanceof ApiError && error.status === 422) storage.removeItem(key);
    throw error;
  }
}

async function uploadParts(
  file: File,
  videoId: string,
  status: UploadStatus,
  deps: UploadDeps,
  signal: AbortSignal,
  onProgress: (uploadedBytes: number) => void,
): Promise<void> {
  const queue = missingParts(file.size, status);
  const sizeOf = (n: number) => {
    const { start, end } = partRange(n, status.part_size, file.size);
    return end - start;
  };

  let finishedBytes = file.size - queue.reduce((sum, n) => sum + sizeOf(n), 0);
  const inFlight = new Map<number, number>();
  const report = () => {
    let bytes = finishedBytes;
    for (const sent of inFlight.values()) bytes += sent;
    onProgress(bytes);
  };
  report();

  // Upload links are fetched a few parts ahead, shared by all workers, and
  // dropped well before they expire (about an hour).
  const links = new Map<number, { url: string; fetchedAt: number }>();
  let fetching: Promise<unknown> | undefined;
  const linkFor = async (n: number): Promise<string> => {
    for (;;) {
      const link = links.get(n);
      links.delete(n);
      if (link && Date.now() - link.fetchedAt < LINK_MAX_AGE_MS) return link.url;
      if (fetching) {
        // Another worker is already fetching; it may be fetching this part.
        await fetching.catch(() => {});
        continue;
      }
      const batch = [n, ...queue.filter((m) => !links.has(m)).slice(0, LINK_BATCH - 1)];
      const request = deps.api.getPartUrls(videoId, batch);
      fetching = request;
      try {
        const { parts } = await request;
        const fetchedAt = Date.now();
        for (const p of parts) links.set(p.part_number, { url: p.url, fetchedAt });
      } finally {
        fetching = undefined;
      }
      const url = links.get(n)?.url;
      links.delete(n);
      if (!url) throw new Error(`No upload link returned for part ${n}`);
      return url;
    }
  };

  const uploadPart = async (n: number) => {
    const { start, end } = partRange(n, status.part_size, file.size);
    for (let attempt = 1; ; attempt++) {
      try {
        // Each attempt takes a new link: an expired link is a likely reason
        // for the previous attempt to have failed.
        const url = await linkFor(n);
        signal.throwIfAborted();
        await deps.putPart(url, file.slice(start, end), signal, (sent) => {
          inFlight.set(n, sent);
          report();
        });
        break;
      } catch (error) {
        inFlight.delete(n);
        report();
        if (signal.aborted) throw signal.reason;
        if (attempt >= MAX_ATTEMPTS || isPermanent(error)) throw error;
        await deps.sleep(backoffMs(attempt), signal);
      }
    }
    inFlight.delete(n);
    finishedBytes += end - start;
    report();
  };

  const worker = async () => {
    for (let n = queue.shift(); n !== undefined; n = queue.shift()) {
      await uploadPart(n);
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
}

export const putPart: PutPart = (url, body, signal, onProgress) =>
  new Promise((resolve, reject) => {
    // XMLHttpRequest, because fetch cannot report upload progress. No headers
    // are set: the link is signed for a bare PUT.
    const xhr = new XMLHttpRequest();
    let stalled = false;
    let stallTimer: ReturnType<typeof setTimeout>;
    const watchForStall = () => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        stalled = true;
        xhr.abort();
      }, STALL_TIMEOUT_MS);
    };
    const onAbort = () => xhr.abort();

    xhr.upload.onprogress = (event) => {
      watchForStall();
      onProgress(event.loaded);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(`Storage refused the upload (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.onabort = () =>
      reject(stalled ? new Error("Upload stalled") : signal.reason);
    xhr.onloadend = () => {
      clearTimeout(stallTimer);
      signal.removeEventListener("abort", onAbort);
    };

    signal.addEventListener("abort", onAbort);
    xhr.open("PUT", url);
    xhr.send(body);
    watchForStall();
    if (signal.aborted) xhr.abort();
  });

export function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}
