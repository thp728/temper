# ADR-0062 — The configuration boundary sits at deployment settings

- **Status:** accepted
- **Date:** 2026-08-28
- **Spec:** `docs/specs/008-the-stack-foundation.md`
- **Issue:** [#29](https://github.com/thp728/temper/issues/29)

## Context

Issue #29's criterion is *"every configuration value is read from the
environment through typed settings, with no literals in code"*, and parallel
branches keep landing domain thresholds as named module-level constants with a
derivation written beside them. The two sound like the same rule and are not:
applying the first to the second would hoist every number into an environment
variable, which makes the product worse (a divergence threshold derived against
a measured report is not a deployment knob) and would break the branches that
own those constants. The boundary is the decision most likely to be revisited,
which is why it is recorded here as it lands.

Two standing rules bear on it. ADR-0010 says *"a value two components must
agree on is defined once and read, never retyped"* — and a settings layer is
exactly where that rule gets broken by accident, when the control plane and
the web half each re-declare an address they share. ADR-0051 says the
fault-injection surface is off by default and refuses on the real tier, so the
question of which tier a start lands in is a safety property, not a
convenience.

## Decision

**The boundary sits at "does the right number depend on the deployment?"**

- A value whose right number depends on the deployment's hardware, catalog,
  network or tolerance is a **deployment setting**. It lives in
  `apps/control-plane/src/temper_control_plane/config.py`, is read from a
  `TEMPER_*` environment variable through a typed reader with a documented
  local default, and is bundled into the typed `Settings` object. A database
  path, a port, a base URL, a provider credential, a mode flag, a safety
  limit, a retention bound: these.
- A value whose right number is derived against a measured report, a dataset,
  or the trainer's own schema is a **domain constant**. It stays as a named
  module-level constant in the module that owns the derivation, with the
  reasoning written beside it — `TEARDOWN_CONFIRM_SAMPLES=3`,
  `DEFAULT_MOE_NUM_EXPERTS=8`, a divergence threshold derived against a
  report. These are never collected into the settings layer: moving them would
  lose the derivation and break the branches that own them.

The settings layer is deliberately small and stable. `config.py` keeps the
typed readers it has had (`_seconds`, `_megabytes`, `_count`, `_text`): each
reads one environment variable, refuses a value that cannot be honoured at
import (an operator who set `TEMPER_STALL_TIMEOUT_S=15m` believes a limit is
in force that is not, so it is refused rather than silently replaced by the
default), and documents its default — each documented default defined once in
a `DEFAULT_*` constant, read by the reader and by the typed view. The whole
set is exposed as one typed `Settings` pydantic object whose bare form is
exactly a start with nothing configured.

**Whether real compute is used is a single setting.** `TEMPER_FAKE_PROVIDER`
is the switch, read by every seam that decides whether this process may reach
something external (the provider, model facts, remote datasets, the tokenizer)
and by `/health`, which advertises the mode. Nothing else differs between the
modes: same API, same database, same storage, same orchestrator. A
no-configuration start lands on the real tier with the fault surface off —
the safe mode of ADR-0051 — and the fault surface has no second, easier way to
switch on, because there is no second path; `TEMPER_FAULT_SURFACE` remains the
deliberate operator tier, unchanged.

**One command brings up every service the product needs.** `compose.yaml`
starts the control plane and the web shell (the worker app is empty by design
and joins when orchestration ships something to run). It defaults to the
zero-cost tier so a reviewer with no account and no secrets can walk the whole
journey, because the issue's own words are *"a reviewer's first run should not
require a decision"*. Data lives on a named volume so stopping and restarting
preserves it. Each service declares a healthcheck, and the control plane's own
`/health` reports the database and the object store **separately**, answering
non-200 when either is down — the distinction between a broken dependency and
a broken application, made observable.

## Alternatives considered

- **Hoist every number into an environment variable.** Rejected: it treats a
  derived domain threshold as a deployment knob, loses the recorded
  derivation, and breaks the branches that own those constants. The issue's
  own brief names this as the failure mode to avoid.
- **Adopt pydantic-settings for the typed layer.** ADR-0011 named
  pydantic-settings as a Phase B tool; it was tried and not adopted, for a
  recorded reason: the repository must populate `os.environ` from `.env`
  itself anyway, because the provider SDK resolves `JL_API_KEY` from
  `os.environ`, not from a settings object — so pydantic-settings' dotenv
  feature is redundant, and its field binding would replace a tested
  refuse-not-ignore reader with no behavioural gain.
- **Make the one command run real compute by default.** Rejected: a reviewer
  with no account would watch every launch fail at provisioning — the exact
  bad first ten minutes the issue exists to prevent. The zero-cost tier is
  the documented front door (spec 012); real compute is one documented line
  away.
- **A single flat `/health` ok.** Rejected: it is the exact failure the
  criterion names — a broken dependency indistinguishable from a broken
  application.
- **Fake provider as the no-configuration default.** Rejected for the process
  default: a no-configuration start must land in the safe mode (real tier,
  faults refused, nothing provisionable without credentials). The compose file
  sets the zero-cost tier explicitly and visibly, which is a configured start,
  not an accident.

## Consequences

- `config.py` is the single place a deployment setting is declared; the
  boundary is now recorded so a later change is a deliberate act, not a drift.
- `just up` (or `docker compose up`) is the documented first-run command:
  starts with nothing configured, walks the whole journey for free, and
  preserves data across a restart.
- `/health` names each dependency and the mode; anything that drives launches
  programmatically can ask before spending.
- A new deployment setting is one field in `config.py`; a new domain threshold
  is one constant beside its derivation in its own module.

## Rollback

Each piece unwinds independently. Reverting the one command is deleting
`compose.yaml`, the two Dockerfiles and the `up`/`down`/`logs` recipes (a
process start via `just dev` remains). Reverting the settings layer restores
the module's earlier shape while keeping the `DEFAULT_*` constants and the
typed `Settings` object. Reverting per-dependency health restores the flat
`/health` payload — the journeys only consume the `provider` field, which is
unchanged in both shapes.
