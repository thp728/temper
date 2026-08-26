// The one place this application touches `fetch` directly. Everything above
// this file -- every path, parameter and response type -- comes from the
// generated client, which is why a contract change fails here at build time
// rather than in front of a user.
//
// The API's error contract is stable codes with user-safe messages
// (`{"detail": {"code": ..., "message": ...}}`), so errors surface typed
// rather than as opaque status numbers. A body that is not that shape still
// becomes an ApiError, never an unhandled parse failure.

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

// In the browser: empty base -- requests are same-origin and the Next
// rewrite proxies them to the control plane. On the server (the report page
// renders there): an absolute URL is required by fetch, so the rewrite's
// target itself is used. Both places read the same variable next.config.ts
// reads; the literal below is only the unset default.
const BACKEND_DEFAULT = "http://127.0.0.1:8000";
const BASE_URL =
  process.env.TEMPER_BACKEND_URL ??
  (typeof window === "undefined" ? BACKEND_DEFAULT : "");

interface ParsedDetail {
  code?: string;
  message?: string;
}

function parseDetail(body: unknown): ParsedDetail {
  const detail = (body as { detail?: unknown } | null)?.detail;

  if (typeof detail === "string") {
    return { message: detail };
  }
  if (Array.isArray(detail)) {
    // FastAPI's own request-validation errors: a list of field problems,
    // not our stable-code shape. Named distinctly so a caller can tell a
    // malformed request from a product refusal.
    const first = detail[0] as { msg?: unknown } | undefined;
    return {
      code: "request_invalid",
      message:
        typeof first?.msg === "string" ? first.msg : "The request was rejected.",
    };
  }
  if (detail && typeof detail === "object") {
    const d = detail as { code?: unknown; message?: unknown };
    return {
      code: typeof d.code === "string" ? d.code : undefined,
      message: typeof d.message === "string" ? d.message : undefined,
    };
  }
  return {};
}

async function toApiError(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let message = res.statusText || "The request failed.";
  try {
    const parsed = parseDetail(await res.json());
    if (parsed.code) code = parsed.code;
    if (parsed.message) message = parsed.message;
  } catch {
    // A non-JSON error body keeps the status-derived fallback.
  }
  return new ApiError(res.status, code, message);
}

export interface ApiRequestConfig {
  url: string;
  method: string;
  params?: Record<string, unknown>;
  headers?: Record<string, string>;
  data?: unknown;
}

export const apiFetch = async <T>(
  config: ApiRequestConfig,
  options?: RequestInit,
): Promise<T> => {
  const { url, method, params, headers, data } = config;

  const query = params
    ? new URLSearchParams(
        Object.entries(params)
          .filter(([, v]) => v !== undefined)
          .map(([k, v]) => [k, String(v)]),
      ).toString()
    : "";
  const qs = query ? `?${query}` : "";

  const finalHeaders = new Headers(headers);
  let body: BodyInit | undefined;
  if (data !== undefined) {
    if (data instanceof FormData) {
      // The generated client stamps multipart/form-data without a boundary;
      // only the runtime knows it, so drop the header and let fetch set it.
      finalHeaders.delete("Content-Type");
      body = data;
    } else if (typeof data === "string") {
      body = data;
    } else {
      body = JSON.stringify(data);
    }
  }

  const res = await fetch(`${BASE_URL}${url}${qs}`, {
    ...options,
    method,
    headers: finalHeaders,
    body,
  });
  if (!res.ok) {
    throw await toApiError(res);
  }
  return (await res.json()) as T;
};
