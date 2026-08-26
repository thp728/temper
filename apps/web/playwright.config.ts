import { defineConfig } from "@playwright/test";
import { BACKEND_PORT } from "./src/lib/backend";

// The journeys run against both halves running for real: the Next shell and
// the FastAPI control plane, with the provider stubbed out by the app itself
// (TEMPER_FAKE_PROVIDER swaps its own FakeProvider in for launched jobs).
// Nothing here crosses a network boundary beyond loopback, and no journey can
// reach the billing account -- which is why they can run on every push.
const webPort = Number(process.env.TEMPER_WEB_PORT ?? 3100);
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
      // Not the developer-facing 3000: several agents (and humans) share this
      // machine and its default-port dev servers. A foreign server on 3000
      // would otherwise pass reuseExistingServer and the journeys would drive
      // whatever code that process happens to serve. PORT is what `next dev`
      // reads.
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: { ...process.env, PORT: String(webPort) },
    },
    {
      command: `uv run uvicorn temper_control_plane.main:app --port ${BACKEND_PORT} --log-level warning`,
      cwd: "../..",
      url: `${backendUrl}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      // Read by the app at import; /health advertises the result, and the
      // launch journeys refuse to run against a process where it did not
      // take effect (a dev server started without it would be able to
      // provision real machines -- the one mistake never made cheaply).
      env: { ...process.env, TEMPER_FAKE_PROVIDER: "1" },
    },
  ],
});
