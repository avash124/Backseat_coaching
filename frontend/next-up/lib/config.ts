// NEXT_PUBLIC_ values are inlined at build time, so each one has to be read
// with a literal `process.env.NAME` expression.
export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export const AUTH_ENABLED = process.env.NEXT_PUBLIC_AUTH_ENABLED === "true";
