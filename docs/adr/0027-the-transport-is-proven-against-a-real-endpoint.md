# ADR-0027 — The transport is proven against a real connection endpoint

- **Status:** accepted
- **Date:** 2026-08-26
- **Spec:** `docs/specs/006-streaming-transport-progress-and-artifacts.md`
- **Issue:** [#21](https://github.com/thp728/temper/issues/21)

## Context

The defect that cost this project a real run was a line-ending translation
applied on the way to a remote shell: `stream` opened its SSH pipe in text
mode, Windows translated every `\n` into `\r\n`, and the remote bash refused
the script with `$'\r': command not found` before running a line of it. Two
existing test tiers both missed it, for structural reasons:

* **The fake provider** (`FakeProvider`) records pushes in a list and never
  crosses a connection. No configuration of it can observe what a transport
  does to bytes, because it does not move bytes.
* **The local-process tests** (`test_provider.py`) replace `_ssh` with a local
  Python process, which crosses pipes but not a connection. They caught the
  deadlock class and the backstop class, and after the fix were extended to
  assert stdin arrived untranslated — but they exercise none of what a real
  connection adds: the `ssh` client itself, channel chunking and flow control,
  host-key and identity-file handling, stderr folding, exit-status plumbing.

A hardware test would have caught it, at the price of a provisioning cycle —
which is why no such test existed. The gap sat between the two cheap tiers and
the one expensive tier, and nothing covered it.

## Decision

**A transport tier exists between the doubles and the hardware: an in-process
SSH endpoint on 127.0.0.1 that the real provider implementation is driven
against, with no GPU, no credentials, and no Docker.**

`apps/control-plane/tests/transport_endpoint.py` starts a real SSH server
(`asyncssh`) on a loopback port inside its own asyncio thread. It answers
exactly the three command shapes `JarvisLabsProvider` sends —
`mkdir -p <parent> && cat > <dest>`, `sudo cat <path>`, `bash -s` — holding
pushed bytes in memory, returning them on fetch, and echoing each script line
as it is consumed so streamed output can be observed arriving line by line.
Authentication is an ephemeral Ed25519 keypair minted per endpoint; nothing
touches an agent or an API key.

Tests drive `JarvisLabsProvider.push`, `.fetch` and `.stream` through the real
`ssh` binary against this endpoint, asserting:

* a payload pushed and fetched back is byte-identical, including line endings;
* binary content (all 256 byte values, sized past any buffer) round-trips;
* streamed output arrives while the script is still being produced, not at
  EOF;
* a clean LF-only script runs without newline translation. **This test is the
  one the acceptance criterion means**: against the pre-fix implementation it
  fails, verified by temporarily restoring the old text-mode stdin write
  before this record landed; and
* a CRLF-mangled script produces the exact `$'\r': command not found`
  signature the real machine produced, proving the endpoint can see the
  defect the double could not.

Two product changes fell out of building the tier, both in transport code the
tier now owns:

* `_ssh` parses handles with `shlex`, not `str.split`, because the endpoint's
  handle quotes an identity path containing spaces, and a naive split hands
  `ssh` broken arguments. A provider handle is whatever `create` returned;
  quoting is part of its grammar.
* The tests surfaced that `stream` folds ssh *client* chatter (the
  "Permanently added to known hosts" warning) into the yielded lines — true of
  production too, since production uses the same flags. Documented in the
  tests rather than filtered away: the noise is real transport behaviour.

The tier runs unmarked on every push: `just test` already deselects only
`hardware`, so these tests are in the default gate by construction.

One deliberate deviation from the spec's wording, recorded rather than glossed:
the spec says a real connection endpoint "runs alongside the other services".
Today it starts per test, in-process, because there is no compose stack yet —
that arrives with issue #29, at which point the endpoint can graduate to a
service beside Postgres and MinIO without its tests changing shape.

## Alternatives considered

**An sshd container via testcontainers.** Rejected. It adds a Docker
dependency to the default gate, costs seconds per test on image pull and
container start, and buys nothing this tier needs: the bytes still cross the
same client and channel code. Phase B's stack (issue #29) will bring
containers in for Postgres, Redis and MinIO; if an sshd image ever becomes
free there, this endpoint is behind one interface and swappable.

**Extend the local-process tests instead.** Rejected as sufficient — they are
kept, because they run faster and cover scheduling/deadlock classes well — but
they cannot see connection-level behaviour by construction, which is the whole
lesson of the original defect.

**Drive the fake provider against the endpoint.** Meaningless: the double's
interface does not produce connections, so there is nothing to point at an
endpoint.

**Only verify on hardware.** Rejected: it is the status quo that let the
defect ship. Verification clauses on real hardware stay, but as the last line
of defence, not the only one.

## Consequences

- The suite requires an OpenSSH-compatible client on PATH (present by default
  on Windows 10+, macOS, and effectively all Linux CI images).
- The endpoint emulates three command shapes, not a shell. New transport
  features need matching endpoint support — deliberate, because it keeps each
  new wire behaviour paired with the test that proves it.
- The streaming assertion bounds wall-clock gaps from below (≥ 0.4 s against a
  0.5 s directive), generous enough that CI scheduling noise cannot flake it.
- Windows-specific friction (identity-file ACLs) is handled once, inside the
  endpoint, not in every test.
- The fake provider remains the right stub for orchestration tests; this tier
  adds coverage, it does not replace the cheap tier.

## Rollback

Delete `test_transport_endpoint.py` and `transport_endpoint.py`; restore
`_ssh`'s split if desired (behaviour for unquoted handles is unchanged). The
gate stays green, and the coverage silently regresses to the pre-#21 state —
which is exactly what the deleted tests exist to prevent going unnoticed.
