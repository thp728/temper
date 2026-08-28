"""Suite-wide setup that has to happen before any test module imports.

**One job.** `apps/trainer/entrypoint.py` runs as a flat script at
`/opt/trainer` inside the image, where `thinking.py` sits beside it, so its
import reads `from thinking import ...` and must keep reading that way -- the
container is the truth and a host-only rewrite would test a shape that does not
ship. On this side of the boundary the same module lives in `temper_core`,
because the control plane validates thinking mode with it and the domain cannot
import an application (ADR-0010).

Registering the alias here is what lets one file serve both, and it is the
reason `packages/core/src/temper_core/thinking.py` is copied into the image
rather than duplicated into the trainer directory. `split.py` and
`checkpoint.py` are the same: the entrypoint imports them flat, and the
control plane imports them from `temper_core`, so one copy ships into the
image and one alias serves the host suite.
"""

import sys

import temper_core.checkpoint
import temper_core.split
import temper_core.thinking

sys.modules.setdefault("thinking", temper_core.thinking)
sys.modules.setdefault("split", temper_core.split)
sys.modules.setdefault("checkpoint", temper_core.checkpoint)
