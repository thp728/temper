"""Watch what the reconciler would decide, without deciding anything.

A live run was destroyed as an orphan while its owner job sat in `preparing`
with the machine id already on the row -- the pass logged `listed: 1,
owned: 0`. Every half of that in isolation is correct: the ownership query
returns the id when asked directly, and `set_state` persists it. So the
disagreement is between the two halves at runtime, and this prints both
sides with their types on an interval so the mismatch is visible when it
happens rather than inferred afterwards.

Read-only: it lists and queries, and never destroys.
"""

from __future__ import annotations

import time

from temper_control_plane import db
from temper_control_plane.provider import new_provider, normalize_status


def main() -> None:
    provider = new_provider()
    try:
        while True:
            listed = provider.list_machines()
            owned = db.list_non_terminal_machine_ids()
            stamp = time.strftime("%H:%M:%S")
            for m in listed:
                mid = m.machine_id
                print(
                    f"{stamp} machine={mid!r} type={type(mid).__name__} "
                    f"status={normalize_status(m.status)!r} "
                    f"owned_set={owned!r} "
                    f"matches={mid in set(owned)}",
                    flush=True,
                )
            if not listed:
                print(f"{stamp} no machines; owned_set={owned!r}", flush=True)
            time.sleep(5)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
