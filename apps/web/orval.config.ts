import { defineConfig } from "orval";

// The client is generated from the checked-in contract, never hand-written.
// A response shape that changes in packages/contracts/openapi.json breaks
// this build instead of a page. Regenerate with `just web-client`.
export default defineConfig({
  temper: {
    input: "../../packages/contracts/openapi.json",
    output: {
      target: "./src/lib/api/generated/client.ts",
      // The transport is ours (src/lib/api/mutator.ts); the types, paths and
      // call signatures are the generator's.
      override: {
        mutator: {
          path: "./src/lib/api/mutator.ts",
          name: "apiFetch",
        },
      },
      mode: "single",
    },
  },
});
