import { describe, expect, it, vi } from "vitest";

import { ApiError, type UploadedPart } from "@/lib/api";
import {
  CONCURRENCY,
  LINK_BATCH,
  MAX_ATTEMPTS,
  backoffMs,
  contentType,
  isAbort,
  missingParts,
  partRange,
  startUpload,
  storageKey,
  type PutPart,
  type UploadDeps,
} from "@/lib/upload";

// Each byte holds its own offset mod 251, so a slice can be checked by content.
function makeFile(size: number, name = "game.mp4", type = "video/mp4") {
  const bytes = new Uint8Array(size);
  for (let i = 0; i < size; i++) bytes[i] = i % 251;
  return new File([bytes], name, { type, lastModified: 1700000000000 });
}

function memoryStorage(initial: Record<string, string> = {}) {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
  };
}

interface Sent {
  part: number;
  start: number;
  length: number;
}

function partOf(url: string): number {
  return Number(new URL(url).pathname.split("/").pop());
}

function setup(opts: {
  size: number;
  partSize: number;
  uploaded?: UploadedPart[];
  storage?: Record<string, string>;
  putPart?: PutPart;
}) {
  const partCount = Math.ceil(opts.size / opts.partSize);
  let signature = 0;
  const sent: Sent[] = [];

  type Api = UploadDeps["api"];
  const api = {
    createUpload: vi.fn<Api["createUpload"]>(async () => ({
      video_id: "video-new",
      storage_key: "raw/video-new.mp4",
      part_size: opts.partSize,
      part_count: partCount,
    })),
    getUploadStatus: vi.fn<Api["getUploadStatus"]>(async () => ({
      part_size: opts.partSize,
      part_count: partCount,
      uploaded: opts.uploaded ?? [],
    })),
    getPartUrls: vi.fn<Api["getPartUrls"]>(async (_videoId, partNumbers) => ({
      parts: partNumbers.map((n) => ({
        part_number: n,
        url: `https://storage.test/part/${n}?sig=${++signature}`,
      })),
    })),
    confirmUpload: vi.fn<Api["confirmUpload"]>(async () => ({ job_id: "job-1" })),
    cancelUpload: vi.fn<Api["cancelUpload"]>(async () => {}),
  };

  const defaultPut: PutPart = async (url, body, _signal, onProgress) => {
    const bytes = new Uint8Array(await body.arrayBuffer());
    sent.push({ part: partOf(url), start: bytes[0], length: bytes.length });
    onProgress(body.size);
  };
  const putPart = vi.fn(opts.putPart ?? defaultPut);
  const sleep = vi.fn<UploadDeps["sleep"]>(async () => {});
  const storage = memoryStorage(opts.storage);
  const progress: number[] = [];

  const deps: UploadDeps = { api, putPart, storage, sleep };
  return { api, putPart, sleep, storage, sent, progress, deps, onProgress: (b: number) => progress.push(b) };
}

describe("partRange", () => {
  it("splits the file into server-sized parts with a short last part", () => {
    expect(partRange(1, 100, 250)).toEqual({ start: 0, end: 100 });
    expect(partRange(2, 100, 250)).toEqual({ start: 100, end: 200 });
    expect(partRange(3, 100, 250)).toEqual({ start: 200, end: 250 });
  });

  it("gives a full last part when the size is an exact multiple", () => {
    expect(partRange(3, 100, 300)).toEqual({ start: 200, end: 300 });
  });

  it("stays exact for multi-gigabyte files", () => {
    const size = 6 * 1024 ** 3 + 123;
    const partSize = 64 * 1024 ** 2;
    const count = Math.ceil(size / partSize);
    expect(partRange(count, partSize, size)).toEqual({ start: (count - 1) * partSize, end: size });
    expect(partRange(count, partSize, size).end - partRange(count, partSize, size).start).toBe(123);
  });
});

describe("missingParts", () => {
  const status = (uploaded: UploadedPart[]) => ({ part_size: 100, part_count: 3, uploaded });

  it("lists every part when nothing is uploaded", () => {
    expect(missingParts(250, status([]))).toEqual([1, 2, 3]);
  });

  it("skips parts the server already has", () => {
    expect(
      missingParts(
        250,
        status([
          { part_number: 1, size_bytes: 100 },
          { part_number: 3, size_bytes: 50 },
        ]),
      ),
    ).toEqual([2]);
  });

  it("re-sends a part the server has at the wrong size", () => {
    expect(
      missingParts(
        250,
        status([
          { part_number: 1, size_bytes: 100 },
          { part_number: 2, size_bytes: 40 },
          { part_number: 3, size_bytes: 100 },
        ]),
      ),
    ).toEqual([2, 3]);
  });
});

describe("helpers", () => {
  it("backs off exponentially with a cap", () => {
    expect([1, 2, 3, 4].map(backoffMs)).toEqual([1000, 2000, 4000, 8000]);
    expect(backoffMs(20)).toBe(30_000);
  });

  it("falls back to the extension when the browser gives no type", () => {
    expect(contentType(makeFile(1, "a.MOV", ""))).toBe("video/quicktime");
    expect(contentType(makeFile(1, "a.mp4", ""))).toBe("video/mp4");
    expect(contentType(makeFile(1, "a.mov", "video/quicktime"))).toBe("video/quicktime");
  });

  it("keys a file on name, size and last-modified time", () => {
    expect(storageKey(makeFile(10))).toBe("nextup:upload:game.mp4:10:1700000000000");
  });
});

describe("startUpload", () => {
  it("uploads a new file part by part and confirms it", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100 });

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result).toEqual({ videoId: "video-new", jobId: "job-1" });
    expect(t.api.createUpload).toHaveBeenCalledWith({
      filename: "game.mp4",
      content_type: "video/mp4",
      size_bytes: 250,
    });
    expect(t.sent.sort((a, b) => a.part - b.part)).toEqual([
      { part: 1, start: 0, length: 100 },
      { part: 2, start: 100, length: 100 },
      { part: 3, start: 200 % 251, length: 50 },
    ]);
    expect(t.api.confirmUpload).toHaveBeenCalledWith("video-new");
    expect(t.progress.at(-1)).toBe(250);
    expect(t.storage.map.size).toBe(0);
  });

  it("remembers the upload while it is running", async () => {
    const file = makeFile(250);
    let seen: string | null = null;
    const t = setup({
      size: 250,
      partSize: 100,
      putPart: async () => {
        seen = t.storage.getItem(storageKey(file));
      },
    });
    await startUpload(file, t.deps, t.onProgress).result;
    expect(seen).toBe("video-new");
  });

  it("resumes a saved upload and sends only the missing parts", async () => {
    const file = makeFile(450);
    const t = setup({
      size: 450,
      partSize: 100,
      storage: { [storageKey(file)]: "video-old" },
      uploaded: [
        { part_number: 1, size_bytes: 100 },
        { part_number: 3, size_bytes: 100 },
        { part_number: 5, size_bytes: 50 },
      ],
    });

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result.videoId).toBe("video-old");
    expect(t.api.createUpload).not.toHaveBeenCalled();
    expect(t.api.getUploadStatus).toHaveBeenCalledWith("video-old");
    expect(t.sent.map((s) => s.part).sort()).toEqual([2, 4]);
    const requested = t.api.getPartUrls.mock.calls.flatMap(([, nums]) => nums);
    expect(requested.sort()).toEqual([2, 4]);
    expect(t.progress[0]).toBe(250); // starts from what the server already has
    expect(t.api.confirmUpload).toHaveBeenCalledWith("video-old");
  });

  it("starts over when the saved upload is gone", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100, storage: { [storageKey(file)]: "video-old" } });
    t.api.getUploadStatus.mockRejectedValueOnce(new ApiError(404, "Not found"));

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result.videoId).toBe("video-new");
    expect(t.api.createUpload).toHaveBeenCalledOnce();
    expect(t.sent).toHaveLength(3);
  });

  it("picks up the job when an earlier confirm got through but its answer was lost", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100, storage: { [storageKey(file)]: "video-old" } });
    t.api.getUploadStatus.mockRejectedValueOnce(new ApiError(409, "Upload is processing"));
    t.api.confirmUpload.mockResolvedValueOnce({ job_id: "job-earlier" });

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result).toEqual({ videoId: "video-old", jobId: "job-earlier" });
    expect(t.api.createUpload).not.toHaveBeenCalled();
    expect(t.sent).toHaveLength(0);
    expect(t.storage.map.size).toBe(0);
  });

  it("starts over when the saved upload ended without a job", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100, storage: { [storageKey(file)]: "video-old" } });
    t.api.getUploadStatus.mockRejectedValueOnce(new ApiError(409, "Upload is failed"));
    t.api.confirmUpload.mockRejectedValueOnce(new ApiError(409, "Upload is failed"));

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result.videoId).toBe("video-new");
    expect(t.api.createUpload).toHaveBeenCalledOnce();
    expect(t.sent).toHaveLength(3);
  });

  it("retries a failed part with backoff and a fresh link", async () => {
    const file = makeFile(300);
    const failures = new Map([[2, 2]]);
    const urls: string[] = [];
    const t = setup({
      size: 300,
      partSize: 100,
      putPart: async (url, body, _signal, onProgress) => {
        const part = partOf(url);
        if (part === 2) urls.push(url);
        const left = failures.get(part) ?? 0;
        if (left > 0) {
          failures.set(part, left - 1);
          onProgress(30); // partial progress, then the connection drops
          throw new Error("Network error during upload");
        }
        onProgress(body.size);
      },
    });

    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(result.jobId).toBe("job-1");
    expect(t.putPart).toHaveBeenCalledTimes(5);
    expect(t.sleep.mock.calls.map(([ms]) => ms)).toEqual([1000, 2000]);
    expect(new Set(urls).size).toBe(3); // a new link for every attempt
    expect(t.progress.at(-1)).toBe(300);
    expect(Math.max(...t.progress)).toBe(300); // failed bytes are not double-counted
  });

  it("gives up after repeated failures and keeps the upload resumable", async () => {
    const file = makeFile(200);
    const t = setup({
      size: 200,
      partSize: 100,
      putPart: async (url) => {
        if (partOf(url) === 2) throw new Error("Network error during upload");
      },
    });

    await expect(startUpload(file, t.deps, t.onProgress).result).rejects.toThrow("Network error");

    expect(t.putPart.mock.calls.filter(([url]) => partOf(url) === 2)).toHaveLength(MAX_ATTEMPTS);
    expect(t.api.confirmUpload).not.toHaveBeenCalled();
    expect(t.storage.getItem(storageKey(file))).toBe("video-new");
  });

  it("does not retry when the server refuses the request", async () => {
    const file = makeFile(200);
    const t = setup({ size: 200, partSize: 100 });
    t.api.getPartUrls.mockRejectedValue(new ApiError(401, "Not signed in"));

    await expect(startUpload(file, t.deps, t.onProgress).result).rejects.toMatchObject({ status: 401 });
    expect(t.sleep).not.toHaveBeenCalled();
  });

  it("never has more than the allowed number of parts in flight", async () => {
    const file = makeFile(2000);
    let active = 0;
    let peak = 0;
    const t = setup({
      size: 2000,
      partSize: 100,
      putPart: async () => {
        active++;
        peak = Math.max(peak, active);
        await new Promise((r) => setTimeout(r, 2));
        active--;
      },
    });

    await startUpload(file, t.deps, t.onProgress).result;

    expect(t.putPart).toHaveBeenCalledTimes(20);
    expect(peak).toBe(CONCURRENCY);
  });

  it("requests upload links in small batches as it goes", async () => {
    const file = makeFile(3000);
    const t = setup({ size: 3000, partSize: 100 });

    await startUpload(file, t.deps, t.onProgress).result;

    const batches = t.api.getPartUrls.mock.calls.map(([, nums]) => nums);
    expect(batches.every((b) => b.length <= LINK_BATCH)).toBe(true);
    expect(batches.length).toBeGreaterThan(1);
    expect(batches.length).toBeLessThan(30);
    expect(t.sent.map((s) => s.part).sort((a, b) => a - b)).toEqual(
      Array.from({ length: 30 }, (_, i) => i + 1),
    );
  });

  it("cancels on the server and forgets the upload", async () => {
    const file = makeFile(250);
    const signals: AbortSignal[] = [];
    const t = setup({
      size: 250,
      partSize: 100,
      putPart: (_url, _body, signal) =>
        new Promise((_, reject) => {
          signals.push(signal);
          signal.addEventListener("abort", () => reject(signal.reason));
        }),
    });

    const handle = startUpload(file, t.deps, t.onProgress);
    await vi.waitFor(() => expect(t.putPart).toHaveBeenCalled());
    await handle.cancel();

    expect(t.api.cancelUpload).toHaveBeenCalledWith("video-new");
    expect(t.storage.map.size).toBe(0);
    expect(signals.every((s) => s.aborted)).toBe(true);
    await expect(handle.result).rejects.toSatisfy(isAbort);
    expect(t.api.confirmUpload).not.toHaveBeenCalled();
  });

  it("keeps the upload when parts are missing at the end, and resends only those", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100 });
    t.api.confirmUpload.mockRejectedValueOnce(new ApiError(409, "Part 2 missing"));

    await expect(startUpload(file, t.deps, t.onProgress).result).rejects.toMatchObject({ status: 409 });
    expect(t.storage.getItem(storageKey(file))).toBe("video-new");

    // Try again: the server reports part 2 missing and part 3 at the wrong size.
    t.api.getUploadStatus.mockResolvedValueOnce({
      part_size: 100,
      part_count: 3,
      uploaded: [
        { part_number: 1, size_bytes: 100 },
        { part_number: 3, size_bytes: 10 },
      ],
    });
    t.sent.length = 0;
    const result = await startUpload(file, t.deps, t.onProgress).result;

    expect(t.api.createUpload).toHaveBeenCalledOnce();
    expect(t.sent.map((s) => s.part).sort()).toEqual([2, 3]);
    expect(result).toEqual({ videoId: "video-new", jobId: "job-1" });
    expect(t.storage.map.size).toBe(0);
  });

  it("forgets an upload the server rejects at the end", async () => {
    const file = makeFile(250);
    const t = setup({ size: 250, partSize: 100 });
    t.api.confirmUpload.mockRejectedValueOnce(new ApiError(422, "Not a valid MP4/MOV"));

    await expect(startUpload(file, t.deps, t.onProgress).result).rejects.toMatchObject({ status: 422 });
    expect(t.storage.map.size).toBe(0);
  });
});
