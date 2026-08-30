// Whether the zero-cost tier is in force, read from the same single setting
// the control plane reads (ADR-0067). One rule, shared with config.py's
// `_flag` reader, so the shell's marking and the backend's provider can never
// disagree about which mode is in force: `1`/`true`/`yes`/`on` are the
// zero-cost tier; `0`/`false`/`no`/`off` (or absent) are the real tier.
//
// A value neither half recognises reads as the real tier here. The control
// plane refuses such a value at boot rather than guessing, so a running stack
// cannot be configured that way; the shell just never claims it is a
// demonstration it cannot prove.
export function isZeroCostMode(): boolean {
  const raw = process.env.TEMPER_FAKE_PROVIDER;
  if (!raw || raw.trim() === "") return false;
  const value = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(value)) return true;
  return false;
}
