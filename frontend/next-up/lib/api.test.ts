import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, errorMessage, uploadErrorMessage } from "@/lib/api";
import { getAccessToken } from "@/lib/supabase";

vi.mock("@/lib/supabase", () => ({ getAccessToken: vi.fn(async () => null) }));

function respond(status: number, body?: unknown) {
  const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(async () =>
    new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api requests", () => {
  it("sends JSON to the API and no token when signed out", async () => {
    const fetchMock = respond(200, { video_id: "v", storage_key: "k", part_size: 10, part_count: 1 });

    await api.createUpload({ filename: "g.mp4", content_type: "video/mp4", size_bytes: 10 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/uploads");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      filename: "g.mp4",
      content_type: "video/mp4",
      size_bytes: 10,
    });
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("attaches the access token when there is one", async () => {
    vi.mocked(getAccessToken).mockResolvedValueOnce("token-123");
    const fetchMock = respond(200, { id: "j", video_id: "v", status: "queued", pipeline_version: "0", error: null });

    await api.getJob("j");

    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({ Authorization: "Bearer token-123" });
  });

  it("handles an empty 204 response", async () => {
    respond(204);
    await expect(api.cancelUpload("v")).resolves.toBeUndefined();
  });

  it("turns error responses into ApiError with the server's detail", async () => {
    respond(413, { detail: "File exceeds 8 GB" });
    const error = await api.createUpload({ filename: "g.mp4", content_type: "video/mp4", size_bytes: 1 }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 413, message: "File exceeds 8 GB" });
  });

  it("copes with validation errors whose detail is a list", async () => {
    respond(422, { detail: [{ loc: ["body"], msg: "field required" }] });
    const error = await api.confirmUpload("v").catch((e) => e);
    expect(error).toMatchObject({ status: 422, message: "Request failed (422)" });
  });
});

describe("uploadErrorMessage", () => {
  it.each([
    [413, "too large"],
    [415, "MP4 or MOV"],
    [422, "playable video"],
    [409, "Try again"],
    [401, "sign in again"],
    [500, "on our end"],
  ])("explains a %i in plain language", (status, text) => {
    expect(uploadErrorMessage(new ApiError(status, "detail"))).toContain(text);
  });

  it("explains a network failure", () => {
    expect(uploadErrorMessage(new Error("Network error during upload"))).toContain("internet connection");
  });
});

describe("errorMessage", () => {
  it.each([
    [404, "couldn't find that game"],
    [422, "couldn't find that game"],
    [401, "sign in again"],
    [503, "on our end"],
  ])("explains a %i when loading a game", (status, text) => {
    expect(errorMessage(new ApiError(status, "detail"))).toContain(text);
  });

  it("explains a network failure", () => {
    expect(errorMessage(new TypeError("Failed to fetch"))).toContain("internet connection");
  });
});
