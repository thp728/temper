import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The shape every page's fetches share when they fail: the data is null and
// the refusal is typed, exactly as the mutator raises it. One definition,
// because pages rendering refusals differently would be several behaviours
// wearing one name.
export async function load<T>(
  fetch: () => Promise<T>,
): Promise<{
  data: T | null;
  error: ApiError | null;
}> {
  try {
    return { data: await fetch(), error: null };
  } catch (err) {
    return {
      data: null,
      error: err instanceof ApiError ? err : NETWORK_ERROR,
    };
  }
}
