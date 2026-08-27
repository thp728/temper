# Temper

A fine-tuning platform: a user supplies a dataset of examples and chooses a base model, and gets back an adapter that carries the change. This glossary fixes the words for that domain, so the code, the specs and the decision records use one vocabulary rather than three.

## Work

**Job**:
A user's request to fine-tune one base model on one dataset, and the record of what became of it.
_Avoid_: run, training run, task

**Attempt**:
One execution of a job. A job has exactly one today; retry after an out-of-memory failure is what makes them plural.
_Avoid_: run, try, retry

**Job spec**:
The full set of choices a job will train with — base model, dataset, and hyperparameters — captured when the job is created and immutable from that moment, so a completed job's claim about what it did cannot be changed after the fact. The copy the machine receives at launch is derived from this one before any money is spent: the same choices with every hyperparameter resolved, because the trainer applies values rather than choosing them ([ADR-0025](docs/adr/0025-the-control-plane-resolves-and-the-trainer-applies.md)).
_Avoid_: run spec, config, settings

**Cancelled**:
A job that ended because the user asked it to stop. Distinct from failure: the user's own decision is not a defect, and a cancelled job produces no adapter.
_Avoid_: aborted, stopped, killed

**Stalled**:
A job that has stopped producing output without ending. It is not the same as a job that failed and said so, and not the same as one that is merely slow — from the control plane a wedged machine and a working one look identical, which is why the silence itself is what is measured.
_Avoid_: hung, frozen, stuck, timed out

**Duration ceiling**:
The longest a job is permitted to run, whether or not it is still making progress. A circuit breaker against a job nobody is watching rather than a limit on what a user may legitimately train — which is also why it is set from what a healthy run looks like, not from a budget.
_Avoid_: timeout, time limit, max runtime

**Failed**:
A job that ended for a reason nobody chose — including one stopped by a safety limit, which the user did not ask for even though the platform did it deliberately.
_Avoid_: errored, crashed, broken

## Data

**Dataset**:
A file of training examples supplied by the user, one example per line.
_Avoid_: upload, file, corpus, training data

**Row**:
One line of a dataset: a single conversation the model will be trained on.
_Avoid_: example, sample, record, item

**Usable row**:
A row that parsed and satisfied the schema, and so will actually be trained on. A dataset's row count and its usable row count are different numbers and are reported separately.
_Avoid_: valid row, good row

**Thinking mode**:
Whether a dataset's assistant turns contain reasoning traces. It is read from the dataset rather than chosen, and applied identically when training and when serving; a dataset that mixes both is ambiguous by construction and is refused.
_Avoid_: reasoning mode, chain of thought, CoT

**Validation report**:
What checking a dataset produces: its counts, its detected thinking mode, a preview of how the rows were understood, and any problems named against the line they came from.
_Avoid_: validation result, check output, errors

## Model and output

**Base model**:
The pre-trained open-weight model a job starts from, always pinned to a specific revision. Its weights are never modified.
_Avoid_: foundation model, source model, checkpoint

**Catalog**:
The curated set of base models offered to users, each pinned and carrying its licence. Curated is a default rather than a boundary — a model from outside it may be admitted once it passes a compatibility probe.
_Avoid_: model list, registry, supported models

**Adapter**:
The small set of trained weights a job produces, which modifies a base model's behaviour without altering it. This is the deliverable.
_Avoid_: model, fine-tuned model, tuned model, weights

**Artifact**:
What a user downloads: the adapter together with the configuration that makes it loadable. An adapter without that configuration is not usable, so the two are not shipped separately.
_Avoid_: output, result, download, bundle

## Storage

**Object key**:
The name a stored thing is addressed by — every dataset and every artifact alike. A key says *what* was stored, never *where*: only the storage seam resolves a key to a location, so moving storage is configuration rather than a rewrite.
_Avoid_: path, file path, location (for stored things)

**Storage seam**:
The one interface through which objects are put, got, and granted scoped writes. Nothing outside it knows whether the store is a directory or a bucket.
_Avoid_: storage layer, file store, uploads directory

## Compute

**Control plane**:
The component that accepts datasets and jobs, drives machines, records what happened, and holds every artifact. It is the only component that touches storage.
_Avoid_: server, backend, API, orchestrator

**Provider**:
The platform that supplies machines. Its own vocabulary is not ours: what it calls an instance is a machine here.
_Avoid_: cloud, vendor, platform

**Machine**:
A GPU host provisioned for a single job and destroyed when that job ends. It is a pure compute node — reached only by the control plane, and never a source or destination for stored data.
_Avoid_: instance, VM, box, node, server

**Trainer**:
The pinned container image that runs one job on a machine. It owns the training loop; the platform owns the contract it runs under.
_Avoid_: worker, runner, image, container

**Disk**:
The space provisioned on a machine to hold the downloaded weights, retained checkpoints, the trainer image and its working space. Computed per job from the model's facts, floored at the platform minimum and capped at the provider's measured ceiling — never a stored constant. Distinct from storage: storage addresses datasets and artifacts kept after a job ends; disk exists only for the lifetime of the machine and is destroyed with it.
_Avoid_: storage (for this), volume, drive

**Event**:
An appended record of one thing that happened during a job — a state change, a measurement, a line of output, or an error. Every state change appends one, because a job whose state moved with no event recorded is a job that cannot be explained.
_Avoid_: message, log entry, update

## Planning

**Quote**:
What a job is predicted to cost and how long it should take, shown before launch and frozen into the job spec when the job is created. A range, never a point — duration and cost are estimates from one measured run, so the quote warns and never blocks. Composed per phase (provisioning, readiness, image pull, model download, training, teardown) rather than as one blended rate, in the account's own currency as an integer in its smallest unit.
_Avoid_: estimate (as a noun), prediction, cost model

**Dataset version**:
The identity of the stored dataset a quote was computed against — its id and creation time, pinning the immutable stored object. A quote references the dataset version and model revision it was computed against, so it cannot outlive its inputs.
_Avoid_: snapshot, revision (for datasets)

**Decision**:
One choice the predictor made on the user's behalf, returned as a record carrying the decision, the value chosen, the constraint that forced it, and the alternatives with what each would have cost. Method, hardware, device count, disk, precision and sequence length each carry one. The records ride on the quote and are frozen with it, so a completed job explains itself as completely as a planned one.
_Avoid_: justification, explanation

**Override**:
A decision the user pinned instead of the predictor's, expressed in the decision's own vocabulary (the same string the plan shows as `chosen`). Changing one re-requests the plan — the rest recomputes in the one place the predictor recomputes — and an override that cannot be honoured is refused with the same arithmetic that did the refusing. Overrides are frozen into the job spec, and an overridden decision is marked on the plan.
_Avoid_: preference, tweak, edit

## Configuration surface

**Advanced surface**:
The generated set of every control the pinned trainer's configuration schema offers, classified and presented to an experienced user. Generated from the trainer's own schema rather than hand-listed, so it is exactly as wide as the trainer and no wider, and cannot drift when the pinned image moves.
_Avoid_: settings panel, options page, hyperparameter list

**Tier**:
One of three classifications every trainer field lands in. *Calculated* — the platform sets it, the common path never sees it. *Exposed with a named failure mode* — reachable, with the specific thing that goes wrong written beside it. *Known but unsupported here* — present in the trainer, not offered, with the reason stated. The classification is data (the tier file), not code, so a change to a judgement is a reviewable diff.
_Avoid_: level, category, bucket

**Schema snapshot**:
The checked-in configuration schema of the pinned trainer image, introspected from inside it. The universe the advanced surface is generated from: a key not in it is unknown to the trainer, and is refused loudly and echoed back at every level.
_Avoid_: schema dump, config model, field list
