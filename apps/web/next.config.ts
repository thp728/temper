import type { NextConfig } from "next";
import { DEFAULT_BACKEND } from "./src/lib/backend";

// The control plane is the API; this app is its browser surface. Browser code
// calls `/v1/...` same-origin and the rewrite forwards it, so there is no CORS
// surface to maintain and no backend origin leaking into the client bundle.
// The target must be absolute for the proxy, so it is read here on the server
// only -- never NEXT_PUBLIC_, which would bake it into the bundle.
const backend = process.env.TEMPER_BACKEND_URL ?? DEFAULT_BACKEND;

const nextConfig: NextConfig = {
  // Dev-server asset requests are host-checked; Playwright drives the app
  // through 127.0.0.1, which Next does not assume.
  allowedDevOrigins: ["127.0.0.1"],
  // Off because this server proxies a server-sent event stream, and gzip
  // buffers it. `GET /v1/jobs/:id/stream` is the live job surface: the
  // browser's EventSource opened fine (readyState 1) and then received
  // nothing for the whole run, while the same URL under `curl` replayed
  // immediately -- the difference being the `Accept-Encoding: gzip` every
  // browser sends and curl does not. The stream is what the running-job page
  // is, so compressing it costs the feature entirely.
  //
  // The control plane already sets `X-Accel-Buffering: no` for nginx, but
  // that is a hint to a different proxy and Next does not read it, and Next
  // offers no per-route control over its built-in compression. Assets lose
  // gzip from this server; a CDN or reverse proxy in front is where that
  // belongs anyway, and a buffered stream is a broken feature rather than a
  // slower one.
  compress: false,
  async rewrites() {
    return [
      { source: "/v1/:path*", destination: `${backend}/v1/:path*` },
    ];
  },
};

export default nextConfig;
