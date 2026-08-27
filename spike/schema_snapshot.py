"""Turn a fresh introspection into the checked-in schema contract.

Issue #33's generated surface is derived from
`packages/contracts/axolotl-schema.json`, the configuration schema of the
pinned trainer image. That file is deterministic for a given image digest, so
it is checked in and regenerated only when the pinned image moves (issue #44),
at which point the digest-pin test fails and this is the documented path to a
fresh snapshot.

Regenerating:

1. Run `spike/introspect_axolotl.py` inside the pinned image (it prints
   `---SPIKE7-JSON---` followed by the schema as JSON) and save the whole
   output as a findings file like `spike/findings-spike7.json`.
2. Run this script with that findings file as an argument; it writes the
   contract file.
3. Regenerate the advanced surface (`uv run python -m temper_core.surface`)
   and review the diff -- a digest change is a deliberate, visible act.

Usage:
    python spike/schema_snapshot.py spike/findings-spike7.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages" / "contracts" / "axolotl-schema.json"

KEEP = (
    "name",
    "type",
    "required",
    "default",
    "free_form",
    "enum_values",
    "constraints",
    "infrastructure",
    "description",
)


def build(findings_path: Path) -> Path:
    raw = json.loads(findings_path.read_text(encoding="utf-8"))
    intro = raw["introspection"]
    fields = [{k: f[k] for k in KEEP} for f in intro["fields"]]
    fields.sort(key=lambda f: f["name"])

    doc = {
        "_comment": (
            "The configuration schema of the pinned trainer image, as "
            "introspected from AxolotlInputConfig. This is the universe the "
            "advanced surface (#33) is generated from: a field not listed here "
            "is a field the trainer does not know, and a key unknown to the "
            "trainer is refused loudly and echoed back, at every level. "
            "Deterministic for a given image digest, which is what makes the "
            "generated surface stable. Regenerate by running "
            "spike/introspect_axolotl.py inside the pinned image and "
            "transforming the findings with spike/schema_snapshot.py."
        ),
        "_source": f"spike introspection, {findings_path.name}",
        "_source_image": raw["image"],
        "_axolotl_version": intro["axolotl_version"],
        "_config_model": intro["config_model"],
        "counts": {
            "total": len(fields),
            "infrastructure": sum(1 for f in fields if f["infrastructure"]),
            "free_form": sum(1 for f in fields if f["free_form"]),
        },
        "runtime_validators": {
            "note": (
                "Constraints Axolotl enforces in arbitrary Python rather than "
                "in the schema. A generated form cannot know what these will "
                "refuse until the job is already running, so they are the "
                "'combinations that cannot be known ahead of time' named in "
                "Spec 009. Each model_validator commonly encodes several "
                "rules, so these are floors on the runtime-only surface, not "
                "estimates. The model-validator names were captured by the "
                "spike-7 introspection; the field-validator names were not "
                "(pydantic exposes them as a count), so only that count is "
                "recorded."
            ),
            "model_validator_names": intro["constraints"][
                "model_validator_names"
            ],
            "field_validator_count": intro["constraints"]["field_validators"],
        },
        "fields": fields,
    }

    OUT.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return OUT


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    print(build(Path(sys.argv[1])))
