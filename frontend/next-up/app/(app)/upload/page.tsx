import type { Metadata } from "next";

import { Uploader } from "./uploader";

export const metadata: Metadata = { title: "Upload a game" };

export default function UploadPage() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <h1 className="font-heading text-2xl font-semibold">Upload a game</h1>
      <Uploader />
    </div>
  );
}
