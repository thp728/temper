"""Spike 8 - what does the JarvisLabs SDK actually expose?

Phase A used four calls: create, destroy, list, currency. Those were what
proving the loop required, not what exists, and nothing in this repository has
ever enumerated the difference.

**The one to look for is pause.** If a machine can be paused more cheaply than
destroyed and recreated, three flows change at once: cancellation stops being
purely destructive (ADR-0003), OOM retry stops paying a 2-4 minute cold start
per attempt, and resume-from-checkpoint on a fresh machine becomes resume on
the same machine. Each of those is currently specified around a cost that
pausing might remove.

Costs nothing. Every live call it makes is a read.

Three claims are marked SEPARATELY for each capability, because "documented"
and "works on VMs" have already been shown to be different claims on this
platform - spike 1 found startup scripts accepted and silently ignored on VMs:

  documented   - the vendor says it exists
  in_sdk       - resolved by introspection here, right now, not transcribed
  tested_here  - this repo has exercised it, in a spike or in product code

Usage:
    python -u spike8.py             # enumerate + live read-only probes
    python -u spike8.py --offline   # enumerate only, no network at all
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spike import Findings, header, load_dotenv  # noqa: E402

try:
    import jarvislabs
    from jarvislabs import Client
except ImportError:
    sys.exit("jarvislabs SDK not installed. Run: pip install jarvislabs")

FINDINGS_PATH = Path(__file__).parent / "findings-spike8.json"

# The vendor ships its own documentation inside the wheel, at
# jarvislabs/skills/SKILL.md. That is the primary source used here for the
# "documented" column: it travels with the pinned version, so unlike the
# website it cannot drift away from the code under test between one reading and
# the next. Where it and the code disagree, the disagreement is the finding.
SKILL_DOC = Path(jarvislabs.__file__).parent / "skills" / "SKILL.md"


@dataclass
class Capability:
    """One provider capability, with the three claims kept apart."""

    name: str
    group: str
    # Dotted path from the client, e.g. "instances.pause". Introspected, not
    # trusted: if the attribute is gone in a later SDK, in_sdk goes False and
    # the row still prints.
    sdk_path: str | None
    documented: bool
    doc_reference: str
    # Where this repo has exercised it, or "" if it never has.
    tested_in: str = ""
    notes: str = ""
    in_sdk: bool | None = None
    signature: str = ""
    # How the capability is reached. "method" is a call on the client;
    # "parameter" is an argument to create/resume rather than a call of its
    # own; "absent" is a capability the provider does not have. Only a
    # "method" row can disagree with the docs by being missing.
    via: str = "method"

    def resolve(self, client: Client) -> None:
        """Fill in in_sdk and signature by looking at the installed package."""
        if self.sdk_path is None:
            self.in_sdk = False
            return
        target: object = client
        for part in self.sdk_path.split("."):
            target = getattr(target, part, None)
            if target is None:
                self.in_sdk = False
                return
        self.in_sdk = True
        try:
            self.signature = self.sdk_path.split(".")[-1] + str(inspect.signature(target))
        except (TypeError, ValueError):
            self.signature = "<not introspectable>"

    def to_dict(self) -> dict:
        return {
            "capability": self.name,
            "group": self.group,
            "sdk_path": self.sdk_path,
            "documented": self.documented,
            "doc_reference": self.doc_reference,
            "in_sdk": self.in_sdk,
            "via": self.via,
            "signature": self.signature,
            "tested_here": bool(self.tested_in),
            "tested_in": self.tested_in,
            "notes": self.notes,
        }


# The capability table. `documented` and `doc_reference` are transcribed from
# SKILL.md (line numbers are of the version pinned in this venv, and the
# assertions below check the text is still there). `in_sdk` is never
# transcribed - it is resolved at run time.
CAPABILITIES: list[Capability] = [
    # --- instance lifecycle --------------------------------------------
    Capability(
        "create instance", "lifecycle", "instances.create", True,
        "SKILL.md 'jl create'", tested_in="spike1-4, api/provider.py",
    ),
    Capability(
        "destroy instance", "lifecycle", "instances.destroy", True,
        "SKILL.md 'jl destroy'", tested_in="spike1-4, api/provider.py",
    ),
    Capability(
        "list instances", "lifecycle", "instances.list", True,
        "SKILL.md 'jl list'", tested_in="spike1-4, api/provider.py",
    ),
    Capability(
        "get one instance", "lifecycle", "instances.get", True,
        "SKILL.md 'jl get'",
        notes="Polled indirectly through create(); never called directly here.",
    ),
    Capability(
        "PAUSE instance", "lifecycle", "instances.pause", True,
        "SKILL.md: 'Paused (compute billing stopped, storage billing "
        "continues, data persists)'",
        notes="THE FINDING. Compute billing stops, storage billing continues, "
              "the whole VM disk persists. Untested here - pricing below is "
              "the vendor's claim, not a measurement.",
    ),
    Capability(
        "resume instance", "lifecycle", "instances.resume", True,
        "SKILL.md 'jl resume'",
        notes="Region-locked; MAY RETURN A NEW machine_id, so any caller that "
              "stores the id must re-read it. GPU type and storage can both "
              "change on resume, within the original region.",
    ),
    Capability(
        "rename instance", "lifecycle", "instances.rename", True,
        "SKILL.md 'jl rename'",
        notes="Cosmetic. No product use.",
    ),
    Capability(
        "resize a running instance", "lifecycle", None, False,
        "absent from SKILL.md",
        notes="No resize call. Changing GPU or disk goes through pause then "
              "resume with new parameters - which is the same mechanism, at "
              "the cost of an interruption.",
        via="absent",
    ),
    Capability(
        "CPU-only VM", "lifecycle", "instances.create_cpu_vm", True,
        "SKILL.md 'jl cpus'",
        notes="A separate backend path with its own rejections. Not useful for "
              "training; possibly useful as a cheap holder of a filesystem.",
    ),
    Capability(
        "spot instances", "lifecycle", None, True,
        "SKILL.md: 'Spot prices are shown only for GPU containers'",
        notes="is_spot is a create/resume PARAMETER, not a call - and the SDK "
              "raises ValidationError for is_spot with template='vm'. Spot is "
              "containers only, so it is unavailable to this product, which "
              "needs VMs for Docker.",
        via="parameter",
    ),
    # --- storage --------------------------------------------------------
    Capability(
        "instance disk sizing", "storage", None, True,
        "SKILL.md '--storage (default: 100GB)'",
        notes="`storage` is a create parameter with NO client-side upper "
              "bound, so the ceiling is discovered by rejection. Spike 5 "
              "measures where that rejection falls.",
        via="parameter",
    ),
    Capability(
        "persistent filesystem: create", "storage", "filesystems.create", True,
        "SKILL.md 'jl filesystem create'",
        notes="Outlives any instance. The obvious home for a model cache that "
              "would otherwise be re-downloaded per job.",
    ),
    Capability(
        "persistent filesystem: list", "storage", "filesystems.list", True,
        "SKILL.md 'jl filesystem list'",
    ),
    Capability(
        "persistent filesystem: resize", "storage", "filesystems.edit", True,
        "SKILL.md 'jl filesystem'",
        notes="Grows a filesystem in place - the resize the instance disk "
              "does not have.",
    ),
    Capability(
        "persistent filesystem: remove", "storage", "filesystems.remove", True,
        "SKILL.md 'jl filesystem'",
    ),
    Capability(
        "attach filesystem to instance", "storage", None, True,
        "SKILL.md: 'Attach a filesystem at creation with --fs-id'",
        notes="fs_id is a create/resume parameter, not a call.",
        via="parameter",
    ),
    # --- images and templates -------------------------------------------
    Capability(
        "list templates", "images", "account.templates", True,
        "SKILL.md 'jl templates --json'",
        notes="Templates are container images the provider offers. A VM takes "
              "template='vm' and no image - which is why this product ships "
              "its own image through Docker rather than as a template.",
    ),
    Capability(
        "supply a custom image", "images", None, False,
        "absent from SKILL.md",
        tested_in="spike1 (established by absence)",
        notes="No such call. This is the whole reason the product runs Docker "
              "on a VM instead of using a container instance.",
        via="absent",
    ),
    Capability(
        "startup scripts: add/list/update/remove", "images", "scripts.list", True,
        "SKILL.md 'jl scripts add'",
        tested_in="spike1",
        notes="DOCUMENTED AND IN THE SDK, BUT SILENTLY IGNORED ON VMs. "
              "create() accepts script_id without error and cloud-init never "
              "runs it (correction C11). This row is why the three columns are "
              "kept separate.",
    ),
    # --- keys and network ------------------------------------------------
    Capability(
        "SSH keys: list", "keys", "ssh_keys.list", True,
        "SKILL.md 'jl ssh-key'", tested_in="spike1-4 preflight",
        notes="A hard prerequisite of VM creation, and the SDK enforces it "
              "before spending money.",
    ),
    Capability(
        "SSH keys: add", "keys", "ssh_keys.add", True, "SKILL.md 'jl ssh-key add'",
    ),
    Capability(
        "SSH keys: remove", "keys", "ssh_keys.remove", True, "SKILL.md 'jl ssh-key'",
    ),
    Capability(
        "VPC: create/list/get/delete/ips", "keys", "vpcs.list", True,
        "SKILL.md 'jl vpc'",
        notes="Private networking. The mitigation for spike 2's finding that "
              "VMs come up with a public IP and no firewall - worth revisiting "
              "if serving is ever exposed.",
    ),
    # --- availability, pricing, billing ----------------------------------
    Capability(
        "GPU availability and pricing", "pricing", "account.gpu_availability", True,
        "SKILL.md 'jl gpus'", tested_in="spike1-4 preflight, api/provider.py",
        notes="Availability for containers and for VMs is reported separately. "
              "A GPU free for containers may not be free for VMs.",
    ),
    Capability(
        "raw resource metadata", "pricing", "account.resources", True,
        "SKILL.md 'jl gpus --json'",
        notes="The unaggregated payload gpu_availability() is derived from - "
              "where per-GPU price fields live.",
    ),
    Capability(
        "CPU VM availability and pricing", "pricing", "account.resources", True,
        "SKILL.md 'jl cpus'",
    ),
    Capability(
        "account balance", "pricing", "account.balance", True,
        "SKILL.md", tested_in="spike1-4 preflight",
    ),
    Capability(
        "account currency", "pricing", "account.currency", True,
        "SKILL.md", tested_in="spike1-4, api/provider.py",
        notes="Returns INR on this account. Reading it rather than assuming "
              "USD is the difference between a correct quote and one off by "
              "~85x.",
    ),
    Capability(
        "resource metrics", "pricing", "account.resource_metrics", True, "SKILL.md",
    ),
    Capability(
        "user info", "pricing", "account.user_info", True, "SKILL.md",
    ),
    Capability(
        "per-instance cost to date", "pricing", "instances.list", False,
        "absent from SKILL.md",
        tested_in="spike8 live probe",
        notes="CORRECTION, FOUND BY RUNNING THIS SPIKE. This row was first "
              "written as absent, on the reasoning that no CALL is named "
              "'cost'. The live probe disproved it: every Instance row carries "
              "`cost` (a float, currency per account.currency()) and `runtime` "
              "alongside it. The product computes spend from uptime x hourly "
              "rate and never reads the provider's own figure -- so it has a "
              "second, independent number available to reconcile against, "
              "which is worth having when the first one is an estimate.",
    ),
    Capability(
        "invoices / billing history", "pricing", None, False,
        "absent from SKILL.md",
        notes="balance() and the per-instance `cost` field are the only "
              "billing reads. Nothing returns a statement or a line-itemised "
              "history, so anything resembling per-user billing has to be "
              "accumulated by this product as it goes.",
        via="absent",
    ),
    # --- events -----------------------------------------------------------
    Capability(
        "webhooks or an event stream", "events", None, False,
        "absent from SKILL.md",
        notes="NOTHING PUSHES. Every state change is discovered by polling, "
              "which is why the orchestrator's event channel is the machine's "
              "own stdout over SSH (ADR-0001) rather than anything the "
              "provider offers.",
        via="absent",
    ),
    Capability(
        "instance logs via the API", "events", None, False,
        "SKILL.md: 'jl run logs' reads over SSH",
        notes="The CLI's log command SSHes in; there is no server-side log "
              "API. Confirms ADR-0001 had no alternative to reject.",
        via="absent",
    ),
    # --- serverless -------------------------------------------------------
    Capability(
        "serverless deployments", "serving", "deployments.create", True,
        "SKILL.md 'jl deploy = beta serverless model serving'",
        notes="An OpenAI-compatible endpoint with autoscaling and no instance "
              "to manage. Directly relevant to serving a trained adapter, and "
              "out of scope for this submission - recorded so the scope cut is "
              "informed rather than accidental.",
    ),
    Capability(
        "deployment logs", "serving", "deployments.logs", True, "SKILL.md 'jl deploy'",
    ),
    Capability(
        "deployment cost readout", "serving", "deployments.get", True,
        "SKILL.md: 'jl deploy list - compact table with cost'",
        notes="Deployments DO report cost, while instances do not. The one "
              "place the platform exposes spend per resource.",
    ),
]

# Assertions against the shipped documentation. Each is a substring that must
# still be present in SKILL.md for the corresponding transcription above to be
# honest. If a later SDK rewords these, the spike fails loudly rather than
# reporting a stale claim as current.
DOC_ASSERTIONS = {
    "pause stops compute billing":
        "Paused** (compute billing stopped, storage billing continues, data persists)",
    "resume is region-locked": "Resume is **region-locked**",
    "resume may return a new machine_id": "Resume may return a **new machine_id**",
    "spot is containers-only": "Spot prices are shown only for GPU containers",
    "VM disks persist whole": "VMs are full machines and keep their whole disk",
    "deployments have no pause/resume": "no SSH/exec, no pause/resume",
}


def enumerate_sdk_surface(client: Client, f: Findings) -> dict:
    """Walk the client's namespaces and list every public method it has.

    Deliberately separate from the curated table: this half cannot be out of
    date, because it is generated. Anything it lists that the table does not
    mention is a capability that was missed.
    """
    header("SDK SURFACE - generated by introspection")
    surface: dict[str, list[str]] = {}
    namespaces = [n for n in dir(client) if not n.startswith("_") and n != "close"]
    for ns in sorted(namespaces):
        obj = getattr(client, ns)
        methods = []
        for name in sorted(dir(obj)):
            if name.startswith("_"):
                continue
            attr = getattr(obj, name, None)
            if not callable(attr):
                continue
            try:
                methods.append(f"{name}{inspect.signature(attr)}")
            except (TypeError, ValueError):
                methods.append(f"{name}(...)")
        surface[ns] = methods
        print(f"  {ns}: {len(methods)} method(s)")
        for m in methods:
            print(f"      {m}")

    covered = {c.sdk_path for c in CAPABILITIES if c.sdk_path}
    missed = [
        f"{ns}.{m.split('(')[0]}"
        for ns, ms in surface.items()
        for m in ms
        if f"{ns}.{m.split('(')[0]}" not in covered
    ]
    f.record("SDK surface enumerated", True,
             f"{len(namespaces)} namespaces, "
             f"{sum(len(v) for v in surface.values())} public methods")
    if missed:
        f.note("methods not in the curated table", ", ".join(missed),
               uncovered_methods=missed)
    return surface


def check_documentation(f: Findings) -> dict:
    """Verify the shipped docs still say what the table transcribes."""
    header("DOCUMENTATION - vendor SKILL.md shipped inside the wheel")
    if not SKILL_DOC.exists():
        f.record("SKILL.md present", False, str(SKILL_DOC))
        return {"present": False}
    text = SKILL_DOC.read_text(encoding="utf-8")
    f.record("SKILL.md present", True,
             f"{len(text.splitlines())} lines at {SKILL_DOC}")
    results = {}
    for claim, needle in DOC_ASSERTIONS.items():
        found = needle in text
        results[claim] = found
        f.record(f"doc says: {claim}", found,
                 "" if found else f"NOT FOUND -- transcription may be stale: {needle!r}")
    return {"present": True, "path": str(SKILL_DOC), "assertions": results}


def resolve_capabilities(client: Client, f: Findings) -> list[dict]:
    header("CAPABILITY TABLE - documented / in-SDK / tested-here, separately")
    rows = []
    for cap in CAPABILITIES:
        cap.resolve(client)
        rows.append(cap.to_dict())
        # Three columns, three characters: documented / in-SDK / tested-here.
        # Kept adjacent on purpose -- the whole point of this table is that a
        # row can be [y--] or [yy-], and collapsing them into one tick is how
        # "documented" gets read as "works".
        marks = "".join("y" if claim else "-" for claim in
                        (cap.documented, cap.in_sdk, cap.tested_in))
        print(f"  [{marks}] {cap.name}")
        if cap.notes:
            print(f"          {cap.notes}")

    # A row only disagrees with the docs when it claims to be a CALL that is
    # documented and missing (or present and undocumented). Rows reached
    # through a create/resume parameter have no attribute to resolve, and
    # counting them as disagreements buried the one that matters.
    disagreements = [
        r for r in rows
        if (r["via"] == "method" and r["documented"] != bool(r["in_sdk"]))
        or "SILENTLY IGNORED" in r["notes"]
    ]
    for r in disagreements:
        f.note("docs/SDK disagreement", f"{r['capability']}: "
               f"documented={r['documented']} in_sdk={r['in_sdk']}")
    f.record("capabilities enumerated", True,
             f"{len(rows)} rows, {sum(1 for r in rows if r['in_sdk'])} in SDK, "
             f"{sum(1 for r in rows if r['tested_here'])} exercised here, "
             f"{len(disagreements)} disagreement(s)")
    return rows


def answer_pause(rows: list[dict], probes: dict, f: Findings) -> dict:
    """The question this spike exists to answer, answered in one place."""
    header("PAUSE - the question this spike exists for")
    pause = next(r for r in rows if r["capability"] == "PAUSE instance")
    resume = next(r for r in rows if r["capability"] == "resume instance")
    exists = bool(pause["in_sdk"] and resume["in_sdk"])

    # A paused machine seen in the account is stronger evidence than an
    # attribute existing: it means the backend really holds this state, and it
    # reports what such a machine costs.
    observed = probes.get("instances.list", {}).get("paused", [])

    answer = {
        "pause_exists": exists,
        "evidence": "measured: introspected on the installed SDK"
                    if exists else "absent from the installed SDK",
        "observed_paused_instances": observed,
        "pricing": {
            "claim": "compute billing stops; storage billing continues; "
                     "data persists",
            "source": "vendor SKILL.md shipped in the wheel",
            "kind": "DOCUMENTED, NOT MEASURED -- no paused instance has been "
                    "billed and observed by this repo",
        },
        "caveats": [
            "resume is region-locked",
            "resume may return a NEW machine_id -- any stored id must be re-read",
            "a paused VM keeps its whole disk, so a paused 400 GB machine keeps "
            "paying for 400 GB",
        ],
        "reopens": [] if not exists else [
            {
                "record": "ADR-0003 cancellation is destructive",
                "why": "the ADR's premise is that stopping a run has only one "
                       "cheap outcome. Pause is a second one: stop paying for "
                       "the GPU, keep the checkpoint on disk, let the user "
                       "resume or discard. That is a different product "
                       "behaviour, not a cheaper implementation of the same "
                       "one -- so the ADR is FLAGGED FOR REOPENING, not "
                       "silently amended.",
                "counterweight": "pause leaves storage billing running with no "
                                 "run that owns it. The reconciler exists to "
                                 "destroy machines nothing owns, and a paused "
                                 "machine is exactly that shape. Reopening the "
                                 "ADR means deciding who eventually kills a "
                                 "paused machine, and that is not obviously "
                                 "cheaper than destroying it now.",
            },
            {
                "record": "the OOM-retry cost model",
                "why": "retry is specified around paying a 2-4 minute cold "
                       "start per attempt. Pausing instead of destroying "
                       "removes that per-attempt cost -- but only if the "
                       "retry is on the same machine, and OOM retries "
                       "typically want a BIGGER machine, which resume can "
                       "provide within the same region.",
            },
        ],
    }
    print(f"  pause exists: {exists}")
    for row in observed:
        print(f"    OBSERVED paused machine: {row}")
    if observed:
        f.note("a paused machine exists in this account",
               "pause is not merely an SDK attribute -- the backend holds the "
               "state and keeps reporting the machine")
    for line in answer["caveats"]:
        print(f"    caveat: {line}")
    f.record("pause exists", exists,
             "compute billing stops, storage continues (documented, unmeasured)")
    if exists:
        f.note("ADR-0003 flagged for reopening",
               "pause is a second cheap outcome for a stopped run")
    return answer


def live_probes(client: Client, f: Findings) -> dict:
    """Read-only calls against the real account. No writes, no money."""
    header("LIVE PROBES - reads only, nothing provisioned")
    out: dict = {}

    def probe(label: str, fn, summarise) -> None:
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 - a probe records failures
            out[label] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            f.record(label, False, f"{type(e).__name__}: {e}")
            return
        summary = summarise(value)
        out[label] = {"ok": True, "summary": summary}
        f.record(label, True, summary)

    # Deliberately does NOT record the balance figure -- findings files are
    # evidence and this repository goes public.
    probe("account.currency", client.account.currency, lambda v: f"currency={v}")
    probe("account.balance", client.account.balance,
          lambda v: "balance readable (value withheld from findings)")
    probe("instances.list", client.instances.list,
          lambda v: f"{len(v)} instance(s) -- "
                    f"{[getattr(i, 'status', '?') for i in v] or 'none running'}")
    # Pull the paused rows out separately, and record ONLY what supports the
    # finding. This file is a public artifact and the account is a person's:
    # a paused machine here may be their own work and nothing to do with this
    # project, so the name and the template it runs are dropped. What is kept
    # is the billing evidence -- a machine that has been paused for days and is
    # still accruing cost is the strongest support there is for the vendor's
    # claim that storage billing continues while compute stops.
    if out.get("instances.list", {}).get("ok"):
        out["instances.list"]["paused"] = [
            {
                "_note": "an unrelated machine on the same account; identifying "
                         "fields deliberately omitted",
                "gpu_type": getattr(i, "gpu_type", None),
                "storage_gb": getattr(i, "storage_gb", None),
                "paused_for": getattr(i, "runtime", None),
                "cost_accrued": getattr(i, "cost", None),
                "paused_image_size": getattr(i, "paused_image_size", None),
            }
            for i in client.instances.list()
            if getattr(i, "status", None) == "Paused"
        ]
    probe("ssh_keys.list", client.ssh_keys.list, lambda v: f"{len(v)} key(s)")
    probe("filesystems.list", client.filesystems.list, lambda v: f"{len(v)} filesystem(s)")
    probe("scripts.list", client.scripts.list, lambda v: f"{len(v)} startup script(s)")
    probe("vpcs.list", client.vpcs.list, lambda v: f"{len(v)} vpc(s)")
    probe("account.templates", client.account.templates, lambda v: f"{len(v)} template(s)")
    probe("account.gpu_availability", client.account.gpu_availability,
          lambda v: f"{len(v)} gpu row(s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="enumerate the installed SDK only; make no network calls")
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    f = Findings()

    header("SPIKE 8 -- JarvisLabs SDK surface")
    print(f"  SDK at {Path(jarvislabs.__file__).parent}")

    docs = check_documentation(f)

    # The client is constructed even offline: it needs no network to exist, and
    # introspection needs the namespaces it builds. A placeholder key keeps the
    # constructor happy without reaching anything -- but ONLY offline. In live
    # mode a missing key must stop the run, because the alternative is a page
    # of failed probes recorded as findings, and a findings file that says
    # "pause: not tested" when the truth is "nobody was logged in" is worse
    # than no findings file.
    if not os.environ.get("JL_API_KEY"):
        if not args.offline:
            sys.exit("JL_API_KEY is not set. Put it in spike/.env, or run "
                     "--offline to enumerate the installed SDK without it.")
        os.environ["JL_API_KEY"] = "offline-introspection-only"
    client = Client()

    surface = enumerate_sdk_surface(client, f)
    rows = resolve_capabilities(client, f)
    probes = live_probes(client, f) if not args.offline else {"skipped": "--offline"}
    pause = answer_pause(rows, probes, f)

    payload = {
        "spike": "spike8-sdk-surface",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "sdk_path": str(Path(jarvislabs.__file__).parent),
        "documentation": docs,
        "generated_surface": surface,
        "capabilities": rows,
        "pause": pause,
        "live_probes": probes,
        "steps": f.steps,
    }
    FINDINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFindings written to {FINDINGS_PATH}")

    header("WHAT THIS MEANS")
    print("  Pause exists and is documented to stop compute billing while")
    print("  keeping the disk. That is a second cheap outcome for a stopped")
    print("  run, so ADR-0003 is flagged for reopening -- and the reopening")
    print("  has to answer who destroys a paused machine, because a paused")
    print("  machine still bills for storage and has no run that owns it.")
    print("  Nothing pushes: no webhooks, no server-side log API. ADR-0001's")
    print("  stdout-over-SSH channel had no provider alternative to reject.")
    print("  Spot is containers-only, so it is unavailable to a product that")
    print("  needs VMs for Docker. That closes a cost avenue rather than")
    print("  opening one, and it is better closed on paper than in a ticket.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
