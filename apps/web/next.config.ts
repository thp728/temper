import type { NextConfig } from "next";

// The control plane is the API; this app is its browser surface. Browser code
// calls `/v1/...` same-origin and the rewrite forwards it, so there is no CORS
// surface to maintain and no backend origin leaking into the client bundle.
// The target must be absolute for the proxy, so it is read here on the server
// only -- never NEXT_PUBLIC_, which would bake it into the bundle.
const backend =
  process.env.TEMPER_BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Dev-server asset requests are host-checked; Playwright drives the app
  // through 127.0.0.1, which Next does not assume.
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    return [
      { source: "/v1/:path*", destination: `${backend}/v1/:path*` },
      // Until a screen is ported, its existing server-rendered page is
      // proxied through this origin: the journey stays in one place and no
      // link crosses origins mid-flow. Ported screens replace these entries.
      { source: "/jobs/:path*", destination: `${backend}/jobs/:path*` },
    ];
  },
};

export default nextConfig;
