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
  async rewrites() {
    return [
      { source: "/v1/:path*", destination: `${backend}/v1/:path*` },
      // Until a screen is ported, its existing server-rendered page is
      // proxied through this origin: the journey stays in one place and no
      // link crosses origins mid-flow. Each entry below is one unported
      // screen (or its form post); a ported screen deletes its entry in the
      // same change. `/jobs/new` (#38), the job list and `/jobs/:id` (#40)
      // are ported and therefore absent -- the shell's own pages serve them.
      // What remains is the old watch page's cancel form post, reachable
      // only by typing its address until #39 ports live watching, and the
      // stylesheet that page ships with -- an unstyled page is a broken
      // page, whatever test asserts on headings.
      { source: "/jobs/:id/cancel", destination: `${backend}/jobs/:id/cancel` },
      { source: "/static/:path*", destination: `${backend}/static/:path*` },
    ];
  },
};

export default nextConfig;
