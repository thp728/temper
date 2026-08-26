# Web application

The browser surface: a Next.js App Router shell over the control plane's API.
Spec 007; the first screen ported into it is upload + validation report (#28).

## Rules

**The client is generated, never hand-written.** `pnpm generate:client` reads
`packages/contracts/openapi.json`; output lands in `src/lib/api/generated/`,
which is gitignored. If a type you need is missing, fix the API's published
model and regenerate — do not declare it by hand. The one exception is
`src/lib/api/mutator.ts`, which owns transport only (base URL, multipart
boundary handling, typed error extraction) and no path or shape knowledge.

**Errors keep their stable codes.** The mutator turns every non-2xx response
into an `ApiError` carrying `status` and the API's `detail.code`. Render the
code on the page; never launder a refusal into a generic "something went
wrong" while a stable code exists.

**Tests drive the app like a person.** Find controls by accessible name, act,
assert on what the user then sees. Component tests mock the generated client;
Playwright journeys (`e2e/`) run against both halves with the provider stubbed
by the app itself — they must never need hardware or credentials.

## Commands

Everything runs through `just` from the repo root (`web-install`, `web-client`,
`web-lint`, `web-types`, `web-test`, `e2e`, `dev-web`). Without `just`, each
recipe is a single command under `apps/web`. To run the whole journey locally:
`just dev` in one terminal and `just dev-web` in another. Both sides take the
control plane's address from one definition, `src/lib/backend.ts`.

## Notes

- Backend URL for the dev rewrite: `TEMPER_BACKEND_URL`, defaulting to the
  address in `src/lib/backend.ts`.
- Unported old screens are reachable through rewrites (`/jobs/*`) so a journey
  stays in one origin; each ported screen deletes its proxy entry in the same
  change. See [ADR-0013](../../docs/adr/0013-the-interface-consumes-a-client-generated-from-the-api-contract.md).
- jsdom enforces form constraint validation but never sets a file input's
  fakepath value, so submit-blocked-by-`required` cannot be exercised in
  component tests; the empty-file refusal is the component's own guard instead.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
