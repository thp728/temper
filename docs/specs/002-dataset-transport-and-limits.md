# Spec 002 — Dataset transport, size limits, and upload responsiveness

**Status:** ready for tickets
**Phase:** A (target: Fri 2026-08-21 morning)
**Depends on:** nothing (independent of Spec 001)
**Produces:** ADR-0004 (the VM is a pure compute node)

## Problem Statement

Three defects share one root: nobody has ever uploaded a dataset larger than 7.5 KB, so every part of the path from upload to machine was written as though size did not exist.

- **The dataset is hex-encoded into a shell script.** It is expanded to twice its size as text, embedded in the script that runs on the machine, and held in memory several times over. Forty lines earlier in the same module, the trainer's own sources are shipped as a binary archive on standard input — the correct mechanism was already present and was not used for the larger payload.
- **There is no upper bound on a dataset.** Validation enforces a floor and nothing else. A large upload does not fail with a typed error naming the limit; it fails as an out-of-memory crash, or does not fail at all and instead produces a job that cannot finish.
- **Validation blocks the entire server.** The upload handler is asynchronous but performs synchronous, CPU-bound validation inside it, which blocks the event loop rather than one request. Measured at roughly 80 ms per megabyte, a 200 MB upload freezes every other request for 16 seconds; a 1 GB upload would freeze it for over a minute.

There is also a mismatch a user can walk straight into: a dataset can be large enough to accept and still be far too large to train within the maximum job duration. Nothing warns them, so the failure arrives hours later as a duration-limit kill they paid for.

## Solution

Use the transport that already exists, put a measured bound on what may be uploaded, stop validation from blocking everything else, and warn when a dataset is plainly too large to finish.

From the user's perspective:

- Large datasets upload and train correctly rather than inflating into a shell script.
- A dataset that exceeds the limit is refused immediately, with the limit and the actual size both named, and with the reason described as an implementation limit rather than a product rule.
- Uploading a large dataset no longer makes the rest of the application unresponsive.
- A dataset large enough that it plainly cannot finish inside the maximum job duration produces a warning at job creation — a warning, not a block, because the estimate is rough and a wrong block is worse than a wrong warning.

## User Stories

1. As a user with a realistic dataset, I want it delivered to the machine without being doubled in size, so that launching is not slowed by an avoidable encoding.
2. As a user with a realistic dataset, I want the upload path not to hold several copies of my file in memory, so that a moderately large dataset does not exhaust the server.
3. As a user uploading a dataset that is too large, I want it refused immediately with a clear error, so that I do not wait for a crash.
4. As a user whose dataset is refused for size, I want the message to state both the limit and my file's actual size, so that I know how much to trim.
5. As a user whose dataset is refused for size, I want to understand that this is a current implementation limit with a known removal path, so that I do not read it as a permanent product boundary.
6. As a user uploading a large dataset, I want the rest of the application to stay responsive, so that other work is not blocked while my file is validated.
7. As a user uploading a large dataset, I want validation to complete in a predictable time, so that a slow response is explicable.
8. As a user, I want validation to reject a file that is not valid UTF-8 with a byte position, so that I can find the problem.
9. As a user, I want line-numbered validation errors regardless of dataset size, so that large files are as debuggable as small ones.
10. As a user creating a job from a very large dataset, I want a warning that it may not finish inside the maximum duration, so that I can reconsider before paying for a run that gets killed.
11. As a user who accepts that warning, I want the job to launch anyway, so that a rough estimate does not veto my judgement.
12. As a user, I want the dataset limit to be the same whether I am told about it in the documentation or by the API, so that the two never disagree.
13. As an operator, I want the size limit configurable without a code change, so that a deployment with more memory can raise it.
14. As an operator, I want the limit's derivation recorded next to its value, so that a future change is an informed one rather than a guess.
15. As a developer, I want dataset bytes delivered to the machine without line-ending rewriting, so that user data is never silently modified in transit.
16. As a developer, I want one transport mechanism for everything sent to a machine, so that there is a single path to reason about and test.
17. As a reviewer, I want the memory behaviour of validation to be a measured figure rather than an assumption, so that the limit can be defended.

## Implementation Decisions

### Transport

The dataset moves onto the same binary-archive-on-standard-input mechanism already used for the trainer's sources. The hex encoding and its embedding in the remote script are removed entirely.

**Dataset bytes are added to the archive as raw bytes.** The existing path reads its files as text and normalises line endings — correct for a shell script and a Dockerfile, where a stray carriage return breaks the container in ways that read as anything but a line-ending bug, and wrong for user data, which must arrive byte-identical.

The machine remains reachable only over SSH and never contacts object storage; the control plane is the only component that touches storage. This is what makes the transport question a control-plane concern rather than a networking one.

### The size limit

**1 GB**, configurable, enforced at upload.

The number is derived rather than chosen. Peak resident memory for validation was measured across five dataset sizes and converges to **4.8× the file size** (10 MB → 67.6 MB; 25 MB → 139.2 MB; 50 MB → 256.9 MB; 100 MB → 492.0 MB; 200 MB → 959.4 MB; the higher ratios at small sizes are fixed interpreter overhead). At 4.8×, a 1 GB dataset peaks around 4.8 GB — roughly 15% of the development machine's memory, with headroom for concurrent work. One gigabyte is approximately 577,000 conversational rows.

Rejection uses a stable code and names the limit and the actual size. The message and the documentation describe it as a limit of the current in-memory validation path, with streaming named as what removes it.

This is **deliberately below the named baseline**, which documents 25 GB. That is stated openly rather than papered over: 25 GB is a property of a multi-node fleet, and on a single-GPU job with a 24-hour ceiling a dataset that size cannot finish. Advertising a limit the system cannot honour is worse than being visibly below it.

### Upload responsiveness

Validation stops running on the event loop. The handler becomes one that the framework dispatches to its worker pool, so a long validation blocks only its own request.

This is a one-keyword change with a measured 16-second blast radius at 200 MB, and it is recorded as a correction rather than quietly fixed.

### The feasibility warning

At job creation, a dataset large enough that it plainly cannot complete within the maximum job duration produces a warning attached to the job, not a refusal.

The estimate is deliberately crude — derived from measured throughput on the one real run — and is labelled as an estimate wherever it is shown. A precise predictor belongs with Phase B's quoting work, which needs a memory and throughput model anyway.

**The size limit and the duration limit answer different questions and do not reconcile.** One gigabyte is within memory but far beyond what finishes in 24 hours. The warning is the interim acknowledgement of that gap; closing it is Phase B's feasibility check.

### Decision records

This spec is not complete until its decision record exists, written as the change lands rather than afterwards.

- **ADR-0004 — The machine is a pure compute node.** That machines are reachable only over the control plane's own connection and never contact object storage; that all storage is control-plane-side, in both phases; why this makes transport a control-plane concern rather than a networking one; and what it forecloses — chiefly that the machine cannot fetch its own dataset from storage, which is the alternative rejected.

The measured memory multiplier belongs in this record, with its method, because the size limit is derived from it and a future change to the limit needs to know how the number was obtained.

## Testing Decisions

**What makes a good test here.** Tests assert what the user sees: the HTTP status and body of an upload, the error code and the numbers in its message, the presence of a warning on a created job, and the bytes that arrive at the far end of the transport. They do not assert the internal shape of the archive or which encoding helper was used.

**Prior art.** The existing control-plane suite tests uploads through the HTTP seam with a temporary upload directory, including the line-numbered validation error cases; those are the direct model. The provider fake introduced in Spec 001 is the model for asserting what the transport delivers, if the two land in either order.

**Modules under test.**

- *The upload endpoint*, through the HTTP seam: acceptance under the limit, refusal over it, and the content of the refusal.
- *The transport*, through the provider seam: that the bytes captured at the far end are byte-identical to the uploaded file, including a file containing carriage returns, which is the regression test for the line-ending hazard.
- *Job creation*, through the HTTP seam: that a large dataset attaches a warning and still launches.

**Cases that must exist.**

- A dataset just under the limit is accepted; one just over is refused with the stable code, and the message contains both the limit and the actual size.
- A dataset containing carriage returns arrives unmodified.
- A dataset containing non-ASCII text arrives unmodified.
- Job creation on a large dataset yields a warning and a launched job, not a refusal.
- Existing validation behaviour — line-numbered errors, mixed thinking-mode blocking, the row floor — is unchanged by the transport work.

**Size fixtures are generated, never committed.** Tests exercise the limit near its boundary using generated content, so no large file enters the repository.

## Out of Scope

- **Streaming validation.** Line-by-line parsing that does not hold the dataset in memory is what removes the 1 GB limit, and it belongs with Phase B's storage work rather than the Friday before the checkpoint.
- **Object storage.** Datasets and artifacts move to object storage in Phase B. The machine still will not reach it — the control plane remains the only component that does.
- **A real feasibility predictor.** Phase B, with quoting.
- **Chunked or resumable upload.** A user with a 1 GB dataset and an unreliable connection is a real case and not this one.
- **Raising the limit to match the baseline.** Only meaningful once streaming exists at every layer; recorded so the gap is deliberate rather than forgotten.

## Further Notes

The memory multiplier is measured, not estimated: five sizes, one fresh process each, peak working set read from the operating system. The earlier working assumption in discussion was 5–8×, which was high at the top end. The measured 4.8× is still declining slightly as sample size grows, so the true asymptote is a little below it; 4.8× is used as the conservative figure.

The transport defect is worth recording as more than a bug. The correct mechanism existed in the same module, forty lines from the incorrect one, and was written first. This is what an untested path looks like from the inside — it was never wrong enough to notice at 7.5 KB.
