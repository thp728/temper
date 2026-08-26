// The control plane's local address. One definition, read by both places
// that need an absolute origin: next.config.ts (the dev rewrite target) and
// mutator.ts (server-side fetches, which cannot use relative URLs). In the
// browser neither applies -- requests are same-origin and proxied.
export const DEFAULT_BACKEND = "http://127.0.0.1:8000";
