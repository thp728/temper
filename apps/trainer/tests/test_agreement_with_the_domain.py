"""The resolver and the trainer agree, and the values come from one place.

The resolver lives in `temper_core.hyperparams` and defines the job spec's
`hyperparameters` block; the trainer applies exactly that block and refuses
anything else. Until #82 that table was a hand-maintained mirror across three
files; now it is data in `packages/contracts/trainer-defaults.json`, and this
module guards two things: the trainer's required set matches the resolver's
output (a default added without updating the trainer must fail here, not on a
paid machine), and no Python module redeclares the table as literals.

The guard scans rather than imports because the whole point is that nothing
else may declare these names; an import-based test could only check the two
modules that already existed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import entrypoint

from temper_core import hyperparams

# Workspace root anchored on the contract the resolver reads. Everything the
# scan walks is expressed relative to that result rather than found a second way.
ROOT = Path(hyperparams.CONTRACT_PATH).parents[2]

_WATCHED_NAMES = {"DEFAULTS", "ALLOWED_OVERRIDES", "DEFAULT_EPOCHS"}
_LITERAL_NODES = (ast.Dict, ast.Set, ast.List, ast.Constant)


def _module_level_literal_names(path: Path) -> set[str]:
    """Names bound at module level straight to a literal in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, _LITERAL_NODES):
            continue
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def test_the_trainer_requires_exactly_what_the_resolver_produces():
    """`hyperparams.effective` output is the job spec's hyperparameters block.
    If a default is added to the resolver without the trainer learning to read
    it, this fails here rather than as `spec_incomplete` on the GPU."""
    assert set(entrypoint.REQUIRED_HYPERPARAMETERS) == set(
        hyperparams.effective({})
    )


def test_no_python_module_declares_the_table_as_a_literal():
    """A reappeared copy means two places can disagree again. Reading the
    values from the contract is not a copy; binding them to literals is.

    The scan watches the names the copies historically travelled under -- a
    paste carries its names with it -- across every Python file in `apps/`
    and `packages/`. `spike/` is excluded deliberately: it is measurement
    code, not product, and its scripts configure their own runs.
    """
    offenders = []
    for root in (ROOT / "apps", ROOT / "packages"):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            # The canonical reader binds these names too, but from data --
            # a load expression is not a literal, so it never offends.
            declared = _module_level_literal_names(path) & _WATCHED_NAMES
            if declared:
                offenders.append(f"{path}: {sorted(declared)}")
    assert not offenders, "hand-copies of the trainer table reappeared:\n" + (
        "\n".join(offenders)
    )
