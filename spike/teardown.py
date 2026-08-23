"""Destroying a machine, and then proving it.

Correction C17: **a destroy call's return value is not evidence, and neither is
one absent listing.** An instance can report `Destroying` and still be billing,
and a listing can miss a machine that is still there. So teardown here is three
things, not one: call destroy, retry it if it fails, then watch the listing
across CONSECUTIVE samples until the machine is absent every time.

Shared by spikes 5 and 6 rather than copied into each, because the one thing
that must not vary between two spikes that both provision GPUs is the code that
turns them off.

Every path that creates a VM calls `confirm_destroyed` in a `finally` block.
An orphaned GPU bills until somebody notices.
"""

from __future__ import annotations

import time

DESTROY_ATTEMPTS = 3
CONFIRM_SAMPLES = 3
CONFIRM_INTERVAL_S = 20
CONFIRM_TIMEOUT_S = 300


def confirm_destroyed(client, machine_id: int, f) -> bool:
    """Destroy `machine_id` and prove it is gone. Returns True only if proven.

    On failure the machine id is printed with the command to remove it by hand,
    because the alternative to a loud failure here is a GPU nobody is watching.
    """
    from spike import header

    header("TEARDOWN")
    destroyed = False
    for attempt in range(1, DESTROY_ATTEMPTS + 1):
        try:
            client.instances.destroy(machine_id)
            f.record("destroy called", True, f"attempt {attempt}")
            destroyed = True
            break
        except Exception as e:  # noqa: BLE001 - retry any failure
            f.note(f"destroy attempt {attempt}", f"{type(e).__name__}: {e}")
            time.sleep(5)

    if not destroyed:
        f.record("destroy called", False,
                 f"DESTROY MANUALLY: jl destroy {machine_id}")

    # The proof, which is separate from the call and does not trust it.
    header("TEARDOWN CONFIRMATION -- consecutive samples (C17)")
    trace: list[dict] = []
    consecutive_absent = 0
    t0 = time.time()
    while time.time() - t0 < CONFIRM_TIMEOUT_S:
        try:
            rows = client.instances.list()
            match = next((i for i in rows
                          if getattr(i, "machine_id", None) == machine_id), None)
            status = getattr(match, "status", None) if match else "ABSENT"
        except Exception as e:  # noqa: BLE001
            status = f"LIST FAILED: {type(e).__name__}"
        sample = {"at_seconds": round(time.time() - t0), "status": status}
        trace.append(sample)
        print(f"  sample {len(trace)}: {status}")

        # `Destroying` is not gone. Treating it as gone is exactly the mistake
        # C17 records.
        if status == "ABSENT":
            consecutive_absent += 1
        else:
            consecutive_absent = 0
        if consecutive_absent >= CONFIRM_SAMPLES:
            f.record("teardown confirmed", True,
                     f"absent in {CONFIRM_SAMPLES} consecutive listings",
                     teardown_trace=trace)
            return True
        time.sleep(CONFIRM_INTERVAL_S)

    f.record("teardown confirmed", False,
             f"still present or unconfirmed after {CONFIRM_TIMEOUT_S}s -- "
             f"CHECK MANUALLY: jl list; jl destroy {machine_id}",
             teardown_trace=trace)
    return False


def sweep_by_name(client, name_prefix: str, f, keep: bool = False) -> list[int]:
    """Destroy any instance whose name starts with `name_prefix`. Belt and braces.

    `confirm_destroyed` can only destroy a machine whose id the caller managed
    to hold on to, and that is a narrower guarantee than it looks. The provider
    SDK's `create()` polls internally until the machine is `Running`, so an
    exception raised inside that call -- or anywhere between the create
    returning and the caller assigning it -- leaves a billing machine with no
    variable pointing at it. The `finally` block then runs and finds `None`.

    So the sweep does not ask what the caller remembers. It asks the platform
    what exists, which is the only source that knows. Naming every instance a
    spike creates with a fixed prefix is what makes that possible, and it is
    why the prefix is a constant rather than a formatted string.

    **An orphaned GPU bills until somebody notices**, and "somebody notices" is
    not a mechanism.
    """
    from spike import header

    header(f"ORPHAN SWEEP -- anything still named {name_prefix}*")
    try:
        rows = client.instances.list()
    except Exception as e:  # noqa: BLE001
        f.record("orphan sweep", False,
                 f"could not list instances ({type(e).__name__}: {e}) -- "
                 f"CHECK MANUALLY: jl list")
        return []

    strays = [i for i in rows
              if (getattr(i, "name", "") or "").startswith(name_prefix)]
    if not strays:
        f.record("orphan sweep", True, "nothing left behind")
        return []

    ids = [i.machine_id for i in strays]
    if keep:
        f.record("orphan sweep", False,
                 f"--keep, so {ids} LEFT RUNNING AND BILLING")
        return ids

    f.note("orphans found", f"{ids} -- destroying")
    for machine_id in ids:
        confirm_destroyed(client, machine_id, f)
    return ids
