import { defineConfig } from "@playwright/test";
import path from "node:path";
import { E2E_BACKEND_PORT, E2E_WEB_PORT } from "./src/lib/backend";

// The journeys run against both halves running for real: the Next shell and
// the FastAPI control plane, with the provider stubbed out by the app itself
// (TEMPER_FAKE_PROVIDER swaps its own FakeProvider in for launched jobs).
// Nothing here crosses a network boundary beyond loopback, and no journey can
// reach the billing account -- which is why they can run on every push.
//
// Both ports are overridable by environment, and they default to values
// derived from this issue's number (39xx) rather than the developer-facing
// 3000/8000. Two things make a second checkout's run safe on one machine: the
// distinct default ports (a sibling worktree drives its own 3xxx range) and
// `reuseExistingServer: false`, so a journey can never silently reuse a
// process serving another branch's code -- the failure mode that made an
// earlier agent pass every test against the wrong tree. Each webServer
// confirms the process it booted via its own health URL, and the launch
// journeys additionally refuse to run unless /health says the fake is in
// force.
//
// The control plane the journeys boot resets its database at startup
// (TEMPER_DB_RESET): a run that was killed mid-job cannot leave an orphaned
// row that poisons the next gate run's startup. The reset lives in the
// backend's own init rather than here because this config is loaded more
// than once per run, and a wipe at load time would delete the backend's
// database out from under it.
//
// Issue #43: unlike the SQLite file this replaced, the journeys do not get a
// database of their own for free -- creating one requires a live connection
// to create it over, which `just db-up`'s single PostgreSQL container
// provides but does not itself split into a per-purpose database. The
// journeys point at that same container's default database
// (TEMPER_DATABASE_URL), which `just dev` also uses: running `just e2e`
// against a developer's own `just dev` session now resets that developer's
// data, where the SQLite-era separate file could not. Recorded as a known
// regression in isolation rather than hidden; the fix is a dedicated
// journey database, which needs either an init script on the `db-up`
// container or `db.py` gaining the ability to create one -- both outside
// this file and outside issue #43's boundary (persistence's function
// bodies, not deployment tooling).
const webPort = Number(process.env.TEMPER_WEB_PORT ?? E2E_WEB_PORT);
const backendPort = Number(
  process.env.TEMPER_BACKEND_PORT ?? E2E_BACKEND_PORT,
);
const backendUrl =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${backendPort}`;
const repoRoot = path.resolve(import.meta.dirname, "..", "..");
const e2eDatabaseUrl =
  process.env.TEMPER_DATABASE_URL ??
  "postgresql://temper:temper@localhost:5432/temper";
const e2eObjectsPath = path.join(repoRoot, "data", "objects-e2e");

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
      // PORT is set as well because `next dev` reads either. TEMPER_BACKEND_URL
      // points the app's rewrites *and* its server-side fetches at the control
      // plane this config just booted on backendPort -- without it, the app
      // defaults to the developer-facing 8000 and a journey would drive
      // whatever process happens to sit there, which is exactly the wrong
      // server on the wrong port that reuseExistingServer was meant to rule
      // out.
      command: `corepack pnpm dev -p ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: false,
      timeout: 180_000,
      env: {
        ...process.env,
        PORT: String(webPort),
        TEMPER_BACKEND_URL: backendUrl,
      },
    },
    {
      command: `uv run uvicorn temper_control_plane.main:app --port ${backendPort} --log-level warning`,
      cwd: "../..",
      url: `${backendUrl}/health`,
      reuseExistingServer: false,
      timeout: 180_000,
      // Read by the app at import; /health advertises the result, and the
      // launch journeys refuse to run against a process where it did not
      // take effect (a dev server started without it would be able to
      // provision real machines -- the one mistake never made cheaply).
      //
      // TEMPER_FAKE_LINE_DELAY_S slows the simulated machine's output so a
      // launched job lives long enough to watch: output arrives over the
      // stream while the job is still working, and a running job can be
      // cancelled mid-run. Every journey tolerates the extra seconds.
      env: {
        ...process.env,
        TEMPER_FAKE_PROVIDER: "1",
        TEMPER_FAKE_LINE_DELAY_S: "0.6",
        TEMPER_DATABASE_URL: e2eDatabaseUrl,
        TEMPER_DB_RESET: "1",
        TEMPER_STORAGE_ROOT: e2eObjectsPath,
      },
    },
    {
      // Issue #51: orchestration now lives in the worker, not in the
      // control plane. The backend above just inserts the row; this
      // worker claims ``queued`` jobs with ``SELECT ... FOR UPDATE SKIP
      // LOCKED`` and drives them. Without it a journey's launch would
      // stay queued forever. No URL to wait for -- the worker is not
      // HTTP, so it starts after the backend is healthy and stays up
      // for the suite's duration.
      // The package, not the submodule -- see compose.yaml's worker service
      // for why `temper_worker.worker` silently does nothing.
      command: `uv run python -m temper_worker`,
      cwd: "../..",
      reuseExistingServer: false,
      timeout: 180_000,
      env: {
        ...process.env,
        TEMPER_FAKE_PROVIDER: "1",
        TEMPER_FAKE_LINE_DELAY_S: "0.6",
        TEMPER_DATABASE_URL: e2eDatabaseUrl,
        TEMPER_DB_RESET: "1",
        TEMPER_STORAGE_ROOT: e2eObjectsPath,
      },
    },
  ],
});
