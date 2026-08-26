// Where the control plane runs locally. One definition, read by every place
// that needs an origin or a port: next.config.ts (the dev rewrite target),
// mutator.ts (server-side fetches, which cannot use relative URLs) and
// playwright.config.ts (which boots the control plane itself).
export const BACKEND_PORT = 8000;
export const DEFAULT_BACKEND = `http://127.0.0.1:${BACKEND_PORT}`;
