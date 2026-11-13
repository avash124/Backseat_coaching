import { describe, expect, it } from "vitest";

import { formatBytes, formatTime, humanize } from "@/lib/format";

describe("format", () => {
  it("humanizes snake_case labels", () => {
    expect(humanize("pick_and_roll")).toBe("Pick and roll");
    expect(humanize("made_2")).toBe("Made 2");
  });

  it("formats game clock times", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(75.9)).toBe("1:15");
    expect(formatTime(3725)).toBe("1:02:05");
  });

  it("formats file sizes", () => {
    expect(formatBytes(4.5 * 1024 ** 3)).toBe("4.5 GB");
    expect(formatBytes(300 * 1024 ** 2)).toBe("300 MB");
    expect(formatBytes(10)).toBe("1 MB");
  });
});
