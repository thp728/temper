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

**Event**:
An appended record of one thing that happened during a job — a state change, a measurement, a line of output, or an error. Every state change appends one, because a job whose state moved with no event recorded is a job that cannot be explained.
_Avoid_: message, log entry, update
