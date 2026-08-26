import { defineConfig } from "@playwright/test";
import { BACKEND_PORT } from "./src/lib/backend";

// The journeys run against both halves running for real: the Next shell and
// the FastAPI control plane, with the provider stubbed out by the app itself
// The journeys run against both halves running for real: the Next shell and
// the FastAPI control plane, with the provider stubbed out by the app itself
// (TEMPER_FAKE_PROVIDER swaps its own FakeProvider in for launched jobs).
// Nothing here crosses a network boundary beyond loopback, and no journey can
// reach the billing account -- which is why they can run on every push.
//
// Both ports are overridable by environment so two checkouts can run the
// journeys side by side on one machine -- without this, a second checkout
// silently reuses the first's servers (`reuseExistingServer`) and drives
// whatever code those processes happen to serve. 3100 is the journeys' own
// default rather than the developer-facing 3000 for the same reason.
const webPort = Number(process.env.TEMPER_WEB_PORT ?? 3100);
const backendPort = Number(process.env.TEMPER_BACKEND_PORT ?? BACKEND_PORT);
const backendUrl =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${backendPort}`;

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
      // The port is passed explicitly rather than left to next dev's default
      // plus auto-relocation: a relocated port is one nothing health-checks.
      // PORT is set as well because `next dev` reads either.
      command: `corepack pnpm dev -p ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: { ...process.env, PORT: String(webPort) },
    },
    {
      command: `uv run uvicorn temper_control_plane.main:app --port ${backendPort} --log-level warning`,
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
