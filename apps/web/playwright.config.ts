import { defineConfig } from "@playwright/test";
import { BACKEND_PORT } from "./src/lib/backend";

// The journeys run against both halves running for real: the Next shell and
// the FastAPI control plane, with the provider stubbed out by the app itself
// (no credentials needed to upload and validate). Nothing here crosses a
// network boundary beyond loopback, which is why they can run on every push.
const webPort = 3000;
const backendUrl = process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${BACKEND_PORT}`;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  forbidOnly: !!process.env.CI,
  use: {
    baseURL: `http://127.0.0.1:${webPort}`,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "corepack pnpm dev",
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
    {
      command: `uv run uvicorn temper_control_plane.main:app --port ${BACKEND_PORT} --log-level warning`,
      cwd: "../..",
      url: `${backendUrl}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
});
