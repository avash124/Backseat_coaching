import { API_URL } from "@/lib/config";
import { getAccessToken } from "@/lib/supabase";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string | null,
  ) {
    super(detail ?? `Request failed (${status})`);
    this.name = "ApiError";
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = await getAccessToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!response.ok) {
    // FastAPI sends {"detail": "..."}; for request-validation errors detail is
    // a list, which is no use as a message.
    const payload = await response.json().catch(() => null);
    const detail = typeof payload?.detail === "string" ? payload.detail : null;
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export interface UploadSession {
  video_id: string;
  storage_key: string;
  part_size: number;
  part_count: number;
}

export interface UploadedPart {
  part_number: number;
  size_bytes: number;
}

export interface UploadStatus {
  part_size: number;
  part_count: number;
  uploaded: UploadedPart[];
}

export interface PartUrl {
  part_number: number;
  url: string;
}

export type JobStatus = "queued" | "processing" | "done" | "failed";

export interface Job {
  id: string;
  video_id: string;
  status: JobStatus;
  pipeline_version: string;
  error: string | null;
}

export interface Play {
  start_s: number;
  end_s: number;
  team: number | null;
  play_type: string | null;
  outcome: string | null;
  confidence: number | null;
}

export interface VideoResults {
  video_id: string;
  status: string;
  video_url: string | null;
  plays: Play[];
}

export const api = {
  createUpload: (file: { filename: string; content_type: string; size_bytes: number }) =>
    request<UploadSession>("POST", "/uploads", file),
  getUploadStatus: (videoId: string) =>
    request<UploadStatus>("GET", `/uploads/${videoId}/parts`),
  getPartUrls: (videoId: string, partNumbers: number[]) =>
    request<{ parts: PartUrl[] }>("POST", `/uploads/${videoId}/parts`, {
      part_numbers: partNumbers,
    }),
  confirmUpload: (videoId: string) =>
    request<{ job_id: string }>("POST", `/uploads/${videoId}/confirm`),
  cancelUpload: (videoId: string) => request<void>("DELETE", `/uploads/${videoId}`),
  getJob: (jobId: string) => request<Job>("GET", `/jobs/${jobId}`),
  getResults: (videoId: string) =>
    request<VideoResults>("GET", `/videos/${videoId}/results`),
};

/** A plain-language message for anything thrown while loading a game or job. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.status) {
      case 401:
        return "Your session has expired. Please sign in again.";
      case 404:
      case 422: // a malformed id in the address
        return "We couldn't find that game. It may have been removed, or it belongs to a different account.";
    }
    return "Something went wrong on our end. Please try again in a moment.";
  }
  return "We couldn't reach the server. Check your internet connection and try again.";
}

/** Like errorMessage, plus the reasons the server can refuse an upload. */
export function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.status) {
      case 409:
        return "Part of the video didn't arrive. Click Try again to send the missing pieces.";
      case 413:
        return "This video is too large to upload. Try a shorter or lower-quality recording.";
      case 415:
        return "That file type isn't supported. Please choose an MP4 or MOV video.";
      case 422:
        return "This file doesn't look like a playable video. Please export it again as MP4 or MOV and retry.";
    }
  }
  return errorMessage(error);
}
