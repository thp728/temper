// Where the control plane runs locally. One definition, read by every place
// that needs an origin or a port: next.config.ts (the dev rewrite target),
// mutator.ts (server-side fetches, which cannot use relative URLs) and
// playwright.config.ts (which boots the control plane itself).
export const BACKEND_PORT = 8000;
export const DEFAULT_BACKEND = `http://127.0.0.1:${BACKEND_PORT}`;

// The ports the e2e journeys boot their own servers on, distinct from the
// developer-facing 8000/3000 so a journey never silently drives a dev server.
// Derived from an issue number because the parallel worktrees
// share one machine: each wave's journeys must default to ports only its own
// webServers use, or a sibling's run reuses this one's processes. One
// definition, read by playwright.config.ts and by every spec that talks to
// the backend directly.
export const E2E_WEB_PORT = 4730;
export const E2E_BACKEND_PORT = 4731;
