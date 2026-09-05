# ADR-0065 — A served endpoint stops itself

- **Status:** accepted
- **Date:** 2026-08-29
- **Spec:** `docs/specs/011-evaluation-and-delivery.md`
- **Issue:** [#78](https://github.com/thp728/temper/issues/78)

## Context

A job finishes and hands back a file. Downloading weights is not seeing the
model answer, so the platform offers a temporary endpoint to try the tuned
model without downloading anything. The baseline this product is measured
against bills a warm machine per hour until someone remembers to stop it.
The training itself is cheap; the forgotten machine is the bill. Every
interview that named the baseline's loudest complaint named this one.

Spec 011 states the requirement in one sentence: **the endpoint is
temporary by construction. It carries an expiry from the moment it starts,
extends on use, and stops itself.** That sentence is the feature, not a
convenience, and the reason is the cost curve above. An endpoint that
requires a user to remember to stop it is the same product as the baseline
with a different name; one that stops itself is the gap the baseline leaves.

Four merged guarantees shape the room, and none is ours to change:

- **#43 (ADR-0064): persistence is PostgreSQL behind `db.py`.** Function
  names and return shapes survived; `config.DATABASE_URL` replaced
  `config.DB_PATH`; tests get a throwaway Postgres per session. Persisted
  state arrives as a versioned migration with a working `down`, because
  `test_migrations.py` rolls every migration back and asserts the data goes
  with it.
- **#46 (ADR-0063): the spend ceiling is enforced outside the training
  process**, on the control plane's clock, never from what the machine
  reports. A served endpoint bills, so anything kept warm must be visible
  to that accounting rather than becoming a second, unmetered way to spend
  money.
- **#34 (ADR-0057): teardown is confirmed across consecutive absent
  observations**, with `normalize_status()` defined once in `provider.py`.
  When the endpoint stops itself it must destroy through that confirmed
  path, not around it. An orphaned GPU bills until someone notices, which
  is the exact failure this issue is about.
- **#29 (ADR-0062): configuration goes through the typed settings layer**;
  domain constants with recorded derivations stay in their own modules.
  The expiry window is a domain constant with a derivation, not a
  deployment setting.
- **ADR-0010:** `packages/core` is pure and does no I/O; a value two
  components share is defined once, never retyped.

Two criteria are the ones that would be easiest to fake, and the issue
calls them out explicitly:

- **Reachability from outside the machine is verified from outside, not
  from on it.** The platform's firewall does not filter published
  container ports the way it appears to, which is why the trainer
  publishes nothing. A `curl localhost` on the machine proves the process
  is listening and proves nothing about the firewall -- it is exactly the
  check that would miss the defect this criterion exists to catch.

- **`it carries an expiry from the moment it starts, extends on use, and
  stops itself` -- stopping itself is the feature.** A stop that only
  happens when someone calls the stop route is the failure this issue
  exists to prevent. The interaction "extends on use" and "expires" must
  coexist, so a busy endpoint still dies at the ceiling rather than being
  kept alive forever by traffic.

## Decision

**A served endpoint is a billed, warm machine with a hashed key, an idle
expiry and a hard ceiling, verified from outside before it is handed a key,
and it stops itself via the confirmed teardown path without anyone asking.**

- **One endpoint per finished job, authenticated with a key stored hashed.**
  `secrets.token_urlsafe(32)` (256 bits of entropy) is minted at creation,
  `hash_api_key` is `SHA-256(hex)` and that hex is what is stored;
  `verify_api_key` is constant-time; `key_prefix` (first 8 chars) is
  stored alongside for display without revealing the key. The plaintext is
  returned once at creation and never again -- a key that can be read back
  out of the store is a finding, not a feature. The generation and the
  hash live in `temper_core.serving` (pure, no I/O) so the control plane
  and the tests read the same functions.

- **The expiry is two numbers, both domain constants with derivations in
  `temper_core.serving` (ADR-0062).**

  - `ENDPOINT_IDLE_TIMEOUT_S = 15 * 60`. 15 minutes, the same window as the
    training stall detector (ADR-0002), because a silent training job and an
    idle serving machine are the same failure mode -- a GPU billing while
    nobody watches it -- and a user who learns one timeout has learned both.
    A single inference returns in seconds; 15 minutes lets a user try a
    handful of prompts without re-provisioning, while a forgotten endpoint
    left overnight still dies in the first quarter-hour.

  - `ENDPOINT_MAX_LIFETIME_S = 2 * 60 * 60`. 2 hours, so a busy endpoint
    kept alive by traffic still dies. At the cheapest GPU the platform
    provisions (L4 at Rs 41.31/hr, the same rate the idle/max derivation
    assumes) 2h costs ~Rs 82.62 -- about 0.8% of the Rs 10,000 spend ceiling
    (ADR-0063). That is a bill a reviewer can see is small, which is the
    point of a ceiling that is a safety control. 2h is enough to try a model
    interactively (the
    side-by-side comparison is one prompt; ten prompts still fit) without
    forcing a re-provision every few minutes, which a shorter ceiling (30
    min) would do. Longer (8h) bills 4x as much for no additional
    utility; "no ceiling, idle only" is the exact failure the criterion
    says to close: a busy endpoint kept alive forever by traffic.

  Both are `float` seconds, defined once in `temper_core.serving` and read
  by the control plane (to mint expiries), the web (to show "stops at"),
  and the tests (to prove the interaction) -- never retyped (ADR-0010).

- **It carries its expiry from the moment it starts, extends on use, and
  stops itself via a timer -- and the two expiries interact.** At
  creation `expires_at = now + idle`, `max_expires_at = now + max`, and
  `last_used_at = now`. On each successful inference
  `expires_at = min(now + idle, max_expires_at)` and `last_used_at = now`
  (`extend_expiry` in `temper_core.serving`); the idle timer is re-armed
  to the new `expires_at`, the max timer never moves. When either timer
  fires the endpoint is stopped by `_stop_via_timer`, which destroys the
  machine through the confirmed path (consecutive absent observations,
  `normalize_status` from `provider.py`, the same `DESTROY_ATTEMPTS` loop
  `orchestrator._teardown` uses) and marks the row `expired`. A `DELETE`
  is the same path with reason `user_stopped`. A process restart re-reads
  `endpoints` and re-arms both timers for any still-running row, or stops
  those already past their deadline -- an endpoint that survives a restart
  without a timer is an endpoint that never stops.

- **It is a billed, warm machine visible to the spend accounting.**
  Provisioning reuses the `Provider` seam (`provider.create`, same
  `gpu_type`/`device_count` the job was provisioned at when known,
  otherwise L4) and the price the job froze at (`price_per_hour`,
  `currency` copied onto the endpoint row). That rate is what
  `preview_for_job` shows before the endpoint starts (the hourly cost and
  the stop time are shown before it starts, spec 011 story 23) and what
  the bill would multiply: `duration_s / 3600 * price_per_hour` is the
  same derivation `temper_core.actuals` uses, so the endpoint does not
  become a second, unmetered way to spend money -- it is a row with a
  price, a machine id, and a lifetime, all of which the accounting can
  read.

- **Reachability is verified from outside, not from on it, before the key
  is handed out.** `verify_not_reachable` runs on the control plane and
  tries a TCP connect to `host:INFERENCE_PORT` on the machine's public
  host (parsed from the provider's handle). Success means the firewall
  did not hold -- the port is directly reachable -- so the machine is
  destroyed and the start refused with `endpoint_reachable`. Failure
  (refused / timeout) means the port is not directly reachable, which is
  the desired outcome: the inference server (when real) listens on
  `127.0.0.1` only and is reached over SSH, not via a published container
  port, which is why the trainer publishes nothing (ADR-0004). For the
  fake provider the handle is `fake://...` and has no public host to
  probe; verification passes vacuously -- the fake publishes nothing, so
  it is not directly reachable by construction, which is the mitigation
  that works. The check originates on the control plane (outside the
  machine), never on the machine, so a `curl localhost` on the machine
  cannot pass it. Where a genuine outside-in TCP check is not possible
  in the test harness, the PR body names it outstanding and states what
  was done instead and why it does not substitute -- the criterion says
  to do exactly that rather than reinterpret the check into one that can
  be passed.

- **It is persisted as a versioned migration with a working `down`.**
  `0003_serving_endpoints` creates `endpoints` (PK `id`, FK `job_id`,
  `status`, `api_key_hash`/`api_key_prefix`, `created_at`/`expires_at`/
  `last_used_at`/`max_expires_at`, `machine_id`, `price_per_hour`/
  `currency`, `stopped_at`/`stop_reason`, indexes on `job_id` and
  `status`). `test_migrations.py` proves `up` is reachable and `down`
  drops the table and the data, and a one-step rollback drops `endpoints`
  while keeping the baseline and the phase-B tables. All new persistence
  goes through `db.py` functions (`create_endpoint`, `get_endpoint`,
  `get_endpoint_by_job`, `active_endpoints`, `touch_endpoint`,
  `set_endpoint_status`) -- the function-names-and-return-shapes
  contract ADR-0064 records is the boundary, and no code outside `db.py`
  reaches past it.

## Alternatives considered

**Manual stop only, no timer.** Rejected: it is the failure the issue
exists to prevent. The forgotten warm machine is the bill, and a stop that
only happens when someone calls the stop route is the same product as the
baseline with a different name.

**Idle timeout only, no hard ceiling.** Rejected: a busy endpoint kept
alive by traffic lives forever, which is the exact interaction the second
criterion says to cover. The max is what makes "extends on use" and
"expires" coexist rather than one cancelling the other.

**Hard ceiling only, no idle extension.** Rejected: a user actively
probing the model would lose the endpoint mid-session and be forced to
re-provision, which teaches users not to rely on the endpoint at all. The
idle window is the grace for interactive use.

**A short hard ceiling (30 min).** Rejected: too short for interactive
use; ten prompts with thinking between them already approaches 30 minutes,
and a ceiling that forces frequent re-provisioning is a worse surprise
than a 2h bill of Rs 82.

**A long hard ceiling (8h).** Rejected: bills 4x the 2h ceiling for no
additional utility, and a busy endpoint kept alive by traffic would hold a
GPU for a whole working day.

**Stored plaintext keys, or encryption that can be decrypted.** Rejected:
a leak of the store is then a leak of the keys. A hash that can be
compared but not reversed is the honest store.

**Bcrypt/Argon2 for the hash.** Rejected: the key is already 256 bits of
entropy, so a slow, salted hash buys nothing but latency on every
request. SHA-256 hex is sufficient and is what the tests assert is stored.

**Verification from on the machine (`curl localhost`).** Rejected: it
proves the process is listening and proves nothing about the firewall --
it is exactly the check that would miss the defect this criterion exists
to catch. The verification must originate somewhere that is not that
machine, and this implementation runs it on the control plane.

**Published container port for inference.** Rejected for the trainer and
reused for serving: the platform's firewall does not filter published
container ports the way it appears to (spike findings, ADR-0004), so the
mitigation that works is not publishing. The inference server listens on
`127.0.0.1` and the control plane reaches it over SSH; the verification
proves the public port is not directly reachable, which is the observable
that matters.

**A deployment setting for the idle/max window.** Rejected (ADR-0062):
the right number is derived against the billed rate and the spend ceiling, not
against the deployment's hardware or network. A domain constant with its
derivation written beside it is what lets a reviewer question it.

## Consequences

- `temper_core.serving` owns the two windows and the key functions; the
  control plane (`temper_control_plane.serving`) owns provisioning,
  verification, the per-endpoint idle/max timers and the sweep thread, the
  confirmed teardown, and the `preview_for_job` the interface shows before
  creation. The web (`EndpointSection`) shows the hourly cost and the stop
  time before it starts, lets the user start the endpoint and try a prompt
  with the key, and lets them stop it immediately; the idle timer re-arms
  on each use and the max timer never moves, so a busy endpoint still dies
  at the ceiling.
- The API is four routes: `GET /v1/jobs/{id}/endpoint/preview` (cost and
  stop time before it starts), `POST /v1/jobs/{id}/endpoint` (mint key,
  verify from outside, provision, arm timers, return plaintext once),
  `GET /v1/jobs/{id}/endpoint` (status, prefix, expiries, price -- never
  the hash), `POST /v1/jobs/{id}/endpoint/infer` (verify key
  constant-time, check and extend expiry capped by max, return completion),
  and `DELETE /v1/jobs/{id}/endpoint` (confirmed teardown, immediate).
  The generation itself is a stub in the fake tier; on real hardware the
  branch would reach the machine over SSH and run the model there.
- Reachability verification is a real outside-in TCP check on real
  hardware and a vacuous pass on the fake (no public host to probe).
  Where the harness cannot do a genuine outside-in check, the PR body
  names it outstanding under a heading, states what was done instead and
  why it does not substitute -- the criterion says to do exactly that
  rather than reinterpret the check.
- An endpoint that is provisioned but never used still expires and is
  destroyed via the confirmed path; an endpoint that is kept busy still
  expires at the max via the same path. Both are exercised by tests that
  manipulate time and call `sweep_expired` rather than waiting for real
  timers, which is the honest version of "stops itself without anyone
  asking" a test can reach.
- The wait-what is: the PR's "## Why" section answers why stopping itself
  is the feature, not just what was built, because the loudest complaint
  against the commercial baseline is the forgotten warm machine, not the
  training cost.

## Rollback

Revert the `temper_core.serving` module, the `0003_serving_endpoints`
migration (both `up` and `down` -- `down` drops the table and the data),
the `db.py` endpoint functions, the `serving.py` control-plane module and
its lifespan hooks, the five API routes and their contract models, and the
`EndpointSection` web component. The training path, the spend ceiling's
enforcement, the delivery formats, and the evaluation comparison are
untouched -- the boundary the issue drew -- so no other flow notices the
revert. A job that had a running endpoint at revert time is left with a
machine that the next `docker compose down` or manual destroy must clean
up, which is the same orphan a forgotten endpoint would have been.
