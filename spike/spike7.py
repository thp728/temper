"""Spike 7 - how wide is Axolotl's config schema, and can Advanced mode be derived from it?

Advanced mode's design rests on one sentence: *the exposed surface is generated
from Axolotl's own config schema, so it is exactly as wide as the trainer and
no wider.* That is what reconciles "expose every dial" with "unknown keys are
refused loudly" -- the refusal is inherited from the trainer rather than
invented by the platform, and it cannot drift when the pinned digest moves,
because it is derived from the pinned digest.

**Whether that is an afternoon or a week depends on a number nobody has looked
up: how many fields does that schema have?** Forty is a UI. Four hundred is a
different product.

The risk this exists to surface is narrower than the count, though. If Axolotl
expresses its constraints only in runtime validation rather than in the schema,
a generated form will happily accept combinations that fail four minutes into a
paid job -- which is the exact class of failure the predictor exists to
prevent. So this counts fields, and then asks whether the schema knows what it
will refuse.

No GPU, no VM, no money. Runs the pinned image locally and reads its pydantic
model. The image is the contract: reading the schema from a pip-installed
Axolotl would measure a different trainer than the one that runs jobs.

Usage:
    python -u spike7.py                # pull if needed, enumerate, classify
    python -u spike7.py --no-pull      # fail rather than pull (image is large)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header  # noqa: E402

FINDINGS_PATH = Path(__file__).parent / "findings-spike7.json"
TIERS_PATH = (
    Path(__file__).parent.parent / "docs" / "data" / "axolotl-field-tiers.json"
)
DOCKERFILE = Path(__file__).parent.parent / "trainer" / "Dockerfile"
INTROSPECT = Path(__file__).parent / "introspect_axolotl.py"

PULL_TIMEOUT_S = 3600
RUN_TIMEOUT_S = 600


def pinned_image(f: Findings) -> str | None:
    """Read the digest out of trainer/Dockerfile rather than repeating it here.

    A second copy of a pin is a second thing to forget to update, and this
    spike's entire value is that it measured THE image that runs jobs.
    """
    header("THE PIN")
    if not DOCKERFILE.exists():
        f.record("trainer/Dockerfile present", False, str(DOCKERFILE))
        return None
    m = re.search(
        r"^FROM\s+(\S+@sha256:[0-9a-f]{64})",
        DOCKERFILE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not m:
        f.record(
            "digest pin found in Dockerfile",
            False,
            "no FROM ...@sha256: line -- the pin discipline has slipped",
        )
        return None
    image = m.group(1)
    f.record("digest pin found in Dockerfile", True, image)
    return image


def ensure_image(image: str, allow_pull: bool, f: Findings) -> bool:
    header("IMAGE")
    have = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if have.returncode == 0:
        f.record("image present locally", True, "no pull needed")
        return True
    if not allow_pull:
        f.record(
            "image present locally", False, "--no-pull, and it is not here"
        )
        return False
    print(f"  pulling {image} -- this is a multi-GB image")
    t0 = time.perf_counter()
    r = subprocess.run(["docker", "pull", image], timeout=PULL_TIMEOUT_S)
    ok = r.returncode == 0
    f.record("image pulled by digest", ok, f"{time.perf_counter() - t0:.0f}s")
    return ok


def introspect(image: str, f: Findings) -> dict | None:
    """Run the introspection script inside the image and read back its JSON.

    The script goes in on stdin rather than as a bind mount: a bind mount of a
    Windows path into a Linux container brings line endings and permissions
    with it, and both have already cost this project time (C10, and the CRLF
    normalisation spike 4 needed).
    """
    header("INTROSPECTION -- inside the pinned image")
    script = INTROSPECT.read_text(encoding="utf-8").replace("\r\n", "\n")
    t0 = time.perf_counter()
    r = subprocess.run(
        ["docker", "run", "--rm", "-i", "--entrypoint", "python", image, "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        f.record(
            "introspection ran",
            False,
            f"exit {r.returncode}: {r.stderr.strip()[-800:]}",
        )
        return None

    # The script prints a marker so the payload survives anything the image
    # writes to stdout on import -- and Axolotl imports a lot.
    marker = "---SPIKE7-JSON---"
    if marker not in r.stdout:
        f.record(
            "introspection ran",
            False,
            f"no marker in output: {r.stdout.strip()[-800:]}",
        )
        return None
    payload = json.loads(r.stdout.split(marker, 1)[1])
    f.record("introspection ran", True, f"{elapsed:.1f}s inside the container")
    if payload.get("error"):
        f.record("config model imported", False, payload["error"])
        return payload
    return payload


def report(payload: dict, f: Findings) -> None:
    header("FIELD COUNTS")
    c = payload["counts"]
    f.record("total fields", True, str(c["total"]))
    f.record(
        "fields after excluding infrastructure",
        True,
        f"{c['training_relevant']} "
        f"({c['infrastructure']} are paths, output dirs, logging sinks, "
        f"wandb/mlflow/comet sinks, hub push settings)",
    )
    f.record("fields with a default", True, str(c["with_default"]))
    f.record("required fields", True, str(c["required"]))
    f.record(
        "free-form fields (Any / dict / untyped)", True, str(c["free_form"])
    )

    header("CONSTRAINT EXPRESSIVENESS -- the risk this spike exists for")
    ce = payload["constraints"]
    f.record(
        "fields carrying schema-level bounds (ge/le/gt/lt/pattern)",
        True,
        str(ce["with_field_constraints"]),
    )
    f.record("fields typed as an enum or Literal", True, str(ce["enum_typed"]))
    f.note(
        "model_validator hooks (cross-field rules in RUNTIME code)",
        f"{ce['model_validators']} -- these are where mutual exclusions and "
        f"conditional requirements live, and a generated form cannot see "
        f"inside them",
    )
    f.note("field_validator hooks", str(ce["field_validators"]))

    verdict = ce["expressed_in_schema_fraction"]
    print(
        f"\n  Of the constraints that exist, roughly {verdict:.0%} are "
        f"visible to a schema reader."
    )

    header("THE CORRECTNESS SETTINGS FROM AGENTS.md")
    for name, found in payload["correctness_settings"].items():
        f.record(
            f"correctness setting: {name}",
            bool(found),
            found
            if isinstance(found, str)
            else ", ".join(found)
            if found
            else "NOT FOUND under any tried name",
        )


def decide(payload: dict, f: Findings) -> dict:
    """Answer the kill criterion rather than leaving it to a reader."""
    header("DERIVE OR HAND-ENUMERATE?")
    c = payload["counts"]
    ce = payload["constraints"]

    # Two things have to hold for generation to be worth it. The schema has to
    # be structured enough to generate FROM, and the constraints have to be
    # visible enough that the generated form does not accept combinations the
    # trainer will refuse four minutes into a paid job.
    mostly_typed = c["free_form"] / max(c["total"], 1) < 0.30
    constraints_visible = ce["expressed_in_schema_fraction"] >= 0.5

    if mostly_typed and constraints_visible:
        decision = "DERIVE"
        reason = (
            "the schema is typed and carries most of its own "
            "constraints, so a generated form refuses what the trainer "
            "refuses"
        )
    elif mostly_typed:
        decision = "DERIVE THE FORM, HAND-WRITE THE RULES"
        reason = (
            "types and defaults are generable, but the cross-field "
            "rules live in model_validator hooks that a schema reader "
            "cannot see. Generating the fields and hand-enumerating the "
            "combinations is the honest split: the field list cannot "
            "drift from the pin, and the rules are reviewable in a diff."
        )
    else:
        decision = "HAND-ENUMERATE"
        reason = (
            "too much of the schema is free-form to generate a usable "
            "form from"
        )

    out = {
        "decision": decision,
        "reason": reason,
        "mostly_typed": mostly_typed,
        "constraints_visible_in_schema": constraints_visible,
        "surface_size_verdict": "a UI"
        if c["training_relevant"] <= 80
        else "large but tractable"
        if c["training_relevant"] <= 200
        else "a different product -- tiering is mandatory, not optional",
    }
    print(f"  {decision}\n  {reason}")
    f.record(
        "advanced mode is derivable from the schema",
        decision.startswith("DERIVE"),
        reason,
    )
    return out


def check_tiers(payload: dict, f: Findings) -> dict:
    """Read the hand-authored tier classification and check it against the schema.

    The classification is NOT generated. It is a list of judgements about what
    each field does and how it fails, and there is no rule that produces
    "setting pad_token equal to eos_token trains the model never to stop" from
    a type annotation. What CAN be automated is the part that rots: whether
    every classified field still exists in the pinned schema. A tier file that
    names a field the trainer dropped is worse than no tier file, because it
    reads as current.

    The timing lives in the data file rather than here, because it is a
    measurement of human work and this script did not do that work.
    """
    header("TIER CLASSIFICATION -- a sample, timed")
    c = payload["counts"]
    if not TIERS_PATH.exists():
        f.record(
            "tier classification exists",
            False,
            f"{TIERS_PATH} not written -- the sample has not been "
            f"classified, so the full pass cannot be estimated",
        )
        return {"classified": 0}

    data = json.loads(TIERS_PATH.read_text(encoding="utf-8"))
    entries = data["fields"]
    schema_names = {fld["name"] for fld in payload["fields"]}

    unknown = [e["name"] for e in entries if e["name"] not in schema_names]
    f.record(
        "every classified field exists in the pinned schema",
        not unknown,
        "all present"
        if not unknown
        else f"NOT IN THE SCHEMA: {unknown} -- the classification has "
        f"drifted from the pin",
    )

    tiers: dict[str, int] = {}
    for e in entries:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    for tier, n in sorted(tiers.items()):
        f.note(f"tier: {tier}", f"{n} of {len(entries)}")

    minutes = data["_classification_minutes"]
    per_field = minutes / len(entries)
    projected_hours = per_field * c["training_relevant"] / 60
    f.record(
        "sample classified",
        True,
        f"{len(entries)} fields in {minutes} minutes "
        f"= {per_field:.2f} min/field",
    )
    f.note(
        "full pass estimate",
        f"{projected_hours:.1f} hours for all {c['training_relevant']} "
        f"training-relevant fields -- DERIVED from the sample, not measured",
    )

    # The number that changes the estimate, and it is not the field count.
    unsupported_share = tiers.get("known_but_unsupported", 0) / len(entries)
    f.note(
        "share of the sample that is KNOWN BUT UNSUPPORTED",
        f"{unsupported_share:.0%} -- these are fields belonging to methods "
        f"this product does not do (DPO, Q-GaLore, LISA, EAFT). They need a "
        f"one-line reason, not a design. If that share holds across the "
        f"344, the surface that needs real design work is roughly "
        f"{round(c['training_relevant'] * (1 - unsupported_share))} fields, "
        f"not 344 -- and THAT is the number the ticket estimate should use.",
    )

    return {
        "classified": len(entries),
        "sampling": data.get("_sampling"),
        "minutes": minutes,
        "minutes_per_field": round(per_field, 2),
        "tiers": tiers,
        "unknown_to_schema": unknown,
        "projected_hours_for_all_training_fields": round(projected_hours, 1),
        "known_but_unsupported_share": round(unsupported_share, 2),
        "design_relevant_field_estimate": round(
            c["training_relevant"] * (1 - unsupported_share)
        ),
        "estimate_kind": "derived from a 20-field seeded random sample",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-pull",
        action="store_true",
        help="fail if the image is not already local",
    )
    args = ap.parse_args()

    f = Findings()
    header("SPIKE 7 -- Axolotl config schema introspection")

    image = pinned_image(f)
    payload: dict | None = None
    if image and ensure_image(image, not args.no_pull, f):
        payload = introspect(image, f)
        if payload:
            payload["image"] = image

    if payload and not payload.get("error"):
        report(payload, f)
        tiers = check_tiers(payload, f)
        decision = decide(payload, f)
        decision["tier_sample"] = tiers
    else:
        decision = {
            "decision": "UNDECIDED -- introspection did not run",
            "reason": "the fallback to a hand-enumerated list is not chosen "
            "here; it is chosen when the schema is measured and "
            "found unusable. An unmeasured schema is not a finding.",
        }
        f.record("schema measured", False, decision["reason"])

    FINDINGS_PATH.write_text(
        json.dumps(
            {
                "spike": "spike7-axolotl-schema",
                "run_at": datetime.now(timezone.utc).isoformat(),
                "image": image,
                "introspection": payload,
                "decision": decision,
                "steps": f.steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFindings written to {FINDINGS_PATH}")
    return 0 if payload and not payload.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
