# Spec 006 — Streaming transport, progress, and artifacts

**Status:** ready for tickets
**Phase:** B, band 1 (the demo path)
**Depends on:** Spec 004 spikes 5 and 9 — the disk ceiling, the download rate, and the streaming-validation throughput
**Produces:** [ADR-0009](../adr/0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md) (superseding ADR-0004), an ADR that progress is measured from the output that was going to be discarded, and one that the artifact is the deliverable and the adapter is one kind of artifact
**Assumes:** ADR-0001 (the event channel is the machine's stdout, pulled over SSH), ADR-0005 (the dataset size limit is derived from measured memory)

## Problem Statement

Three problems that look separate and are one problem: **every byte in this
system is held in memory, and the user is told nothing while it moves.**

**Payloads are buffers.** The transport carries datasets to the machine and
artifacts back as byte strings, held whole. That was correct when the largest
payload was a small adapter and the dataset ceiling was derived from a
validation path that holds several times the file size. It does not survive a
catalog with no size limit: a full fine-tune of a large model produces an
artifact thousands of times larger than anything this path has carried, and the
control plane would need to hold all of it to hand it over.

**The machine may not touch storage, and the largest payloads make that
untenable.** The transport rules say every byte travels over the control
plane's own connection and the machine never contacts object storage — in
either phase, explicitly. Those rules bought something real: no credentials on
the machine, no bucket policy to scope, no egress path to audit, and a
transport question that reduces to writing bytes down a connection the control
plane already owns. But they were written when the artifact was small, and at
the sizes now in scope every byte of a very large model would cross a laptop on
its way from the machine that produced it to the store that keeps it.

**The run is loud and says nothing.** One real run wrote several hundred
events, the overwhelming majority of them container-layer pull progress and a
chat template echoed line by line. The event store returns oldest-first with a
cap, so a finished job's page stops partway through the image pull and never
shows that the artifact was verified, the machine destroyed, or the job
finished. Every user who lands on a completed job sees a truncated render of
its own history. The standing plan was to filter that noise at classification
so it never became an event at all — a plan that discards evidence and was
correctly flagged as a judgment call needing a defence.

Those noisy lines carry bytes-so-far and bytes-total. **The noise is the
progress data.** The phase that will dominate a large job — pulling a hundred
gigabytes of weights — currently produces no progress signal at all, while the
phase that produces hundreds of progress lines is the one slated for deletion.

## Solution

**Bytes stream, and the machine writes its own artifact.**

The transport carries datasets up and artifacts down as streams rather than
buffers, so memory stays flat as payload size grows. A new storage seam holds
every stored object behind one interface. And for artifacts — the payloads that
grow without bound — the machine writes directly to a URL scoped to one object
key for the length of one job, rather than pushing bytes through the control
plane.

That last part supersedes an accepted decision, and the argument for
superseding it is that its stated invariant is already wider than what it
protects. The machine *already* initiates outbound traffic: it pulls a pinned
container image from a registry, and it downloads model weights from a public
hub. It is not, and never was, a component that receives everything and
initiates nothing. **What the rule actually protects is that the machine holds
no long-lived credential and cannot reach anything but its own job.** A URL
that grants one write to one key, expiring with the job, preserves that
completely. Datasets keep travelling over the control plane's connection: they
are small, the path is proven, and normalising line endings on user data is a
corruption bug the current transport already avoids deliberately.

**Progress is promoted, not filtered.**

Classification gains a fourth kind of event. Layer-pull lines and model-download
lines stop being log lines and become progress records carrying the phase, the
bytes done, the bytes total, and a measured rate. Hundreds of log events
collapse into a handful of progress updates that supersede one another, which
means:

- The finished-job page stops truncating, because the events that made it
  overflow no longer exist as events.
- The noise problem is solved by promotion rather than deletion, so *"we keep
  the whole log"* stays true and no evidence is discarded.
- The phase that matters most on a large job gains the progress signal it
  currently lacks entirely.
- The rate is measured live, so a download predicted at one speed and running
  at another shows a corrected estimate instead of a stale one.

**Validation streams**, so the upload ceiling stops being a property of how
validation is written. Rows are validated as they arrive, memory stays flat,
line numbers survive, and token counts are produced in the same pass — which
the quote depends on, because cost is per training token.

**And the deliverable gets its proper name.** The glossary currently makes the
adapter the deliverable. Full fine-tuning produces no adapter, so the artifact
becomes the canonical deliverable and the adapter becomes one kind of it.

## User Stories

1. As a user with a large dataset, I want to upload it without hitting a ceiling that exists because of how validation was written, so that the limit reflects a product decision rather than an implementation detail.
2. As a user uploading a large dataset, I want validation to report errors against their line numbers exactly as it does for a small one, so that size does not cost me the most useful thing validation produces.
3. As a user, I want to see the token count of my dataset after validation, so that I understand the size of what I am about to train on.
4. As a user, I want validation to show progress while it runs, so that a large file does not look like a frozen page.
5. As a user, I want the product to tell me it is provisioning a machine, so that the first silent minute is explained.
6. As a user, I want to see the container image downloading with a proportion complete, so that I can tell the difference between slow and stuck.
7. As a user, I want to see the model weights downloading with a proportion complete, so that the longest phase of a large job is not a blank screen.
8. As a user, I want each phase to show a rate measured from what is actually happening, so that the estimate corrects itself instead of repeating a guess.
9. As a user, I want an estimated time remaining for the phase in progress, so that I can decide whether to wait.
10. As a user, I want the estimate to change when the real rate differs from the predicted one, so that the product is honest about what it is observing.
11. As a user, I want to know which phase my job is in, so that a failure during download is distinguishable from a failure during training.
12. As a user watching a finished job, I want to see how it ended, so that the page tells me the outcome rather than stopping partway through its own history.
13. As a user, I want the full log still available, so that promoting progress does not mean losing detail I might need.
14. As a user, I want the detailed output collapsed by default, so that the interesting lines are not buried under machine chatter.
15. As a user, I want to download what my job produced regardless of which method I chose, so that the product works the same way for an adapter and for a fully trained model.
16. As a user, I want a very large result to download without failing, so that method choice does not silently limit what I can retrieve.
17. As a user, I want to know the size of what I am about to download before I start, so that I am not surprised.
18. As a user, I want the checksum of what I download to match what the machine computed, so that a truncated transfer is caught rather than handed to me looking fine.
19. As a user, I want the configuration needed to load my result shipped alongside it, so that what I download is usable rather than merely present.
20. As a user, I want a failed job to still tell me what happened, so that a failure produces an explanation rather than silence.
21. As an operator, I want the control plane's memory to stay flat regardless of dataset or artifact size, so that one large job cannot exhaust the process.
22. As an operator, I want the machine to hold no credential that outlives its job, so that a compromised machine cannot reach anything else.
23. As an operator, I want the machine unable to enumerate or read stored objects, so that write access to its own result is the entire grant.
24. As an operator, I want stored objects behind one interface, so that moving from local storage to a hosted store is configuration rather than a rewrite.
25. As a reviewer, I want to see why the earlier transport decision stopped holding, so that the change reads as a considered reversal rather than an inconsistency.

## Implementation Decisions

**The provider's transfer methods take and return streams.** Push accepts an
iterator of chunks and a destination; fetch returns an iterator of chunks. The
streaming shape the output channel already uses is extended to the payload
channel, for the same reason it was chosen there: it holds whether the transport
is a shell connection today or something else later, so the transport can change
without the orchestration changing.

**A new `storage` seam** exposes putting an object, getting one, and minting a
scoped write URL. One seam, with a filesystem implementation for tests and local
runs and an object-store implementation behind configuration. Nothing outside
this seam knows where objects live.

**The scoped write URL grants one operation on one key and expires with the
job.** It is minted by the control plane, passed to the machine as part of its
job description, and it confers no ability to list, read, or write anywhere
else. The machine reports the checksum it computed; the control plane verifies
against the stored object before the job is allowed to report success.

**Classification gains a progress kind.** A progress event carries the phase it
belongs to, bytes completed, bytes expected, and an observed rate. Unlike log
events, progress events supersede rather than accumulate — the interface renders
the latest per phase. This is why progress cannot be modelled as a log line or
as a training metric: it has no step, and it replaces rather than appends.

**Bootstrap emits parseable download progress.** Image pull already does; model
download must be invoked in a mode that reports it. Without that change the
phase that dominates a large job stays invisible, which is the whole point of
this spec.

**Raw lines that were promoted are retained but not emitted as events.** They
are written to the job's own output record so nothing is lost, and the interface
offers them as detail. *We keep everything* stays true, and the log stops being
unreadable — solved by promotion rather than by a judgment call about what a
user is allowed to see.

**Validation becomes an iterator over rows rather than a function over a list.**
The public contract — a validation report with counts, detected thinking mode,
a preview and line-numbered problems — is unchanged. Token counting happens in
the same pass, because tokenising is the expensive half and doing it twice
doubles the cost of the thing users wait for. If spike 9 shows tokenising
dominates beyond a wait a user will tolerate, counting splits into a phase of
its own and the quote gains a state for it.

**The upload ceiling becomes a configured product limit** rather than a number
derived from a memory multiplier. The derivation that produced the current limit
is superseded by measurement, and the new figure comes from spike 9.

**The artifact model gains kinds.** An artifact is what a user downloads; an
adapter is one kind of it, a fully trained model is another, and a merged model
is a third. The glossary changes with this spec — the deliverable is the
artifact, and the adapter's entry stops claiming that title. Downstream, the
manifest that travels with an artifact records which kind it is, because loading
one differs from loading another.

## Testing Decisions

**A good test here asserts that bytes arrive unchanged and that memory does not
grow.** Not that a particular chunk size was used, not how many reads happened.
The external behaviours are: what went in comes out identical; peak memory is
flat across payload sizes; progress events describe the phase correctly; and a
truncated transfer is refused.

**The transport tier is new and it is the most important test in this spec.**
The defect that cost this project a run — line endings translated on the way to
a remote shell — lived exactly between the fake provider and the real one. The
fake provider never crosses a pipe and is structurally blind to it. A real
connection endpoint runs alongside the other services, and the *real* provider
implementation is driven against it: push a payload, stream a script's output,
fetch a file, assert bytes round-trip byte-identical including line endings and
binary content. No GPU, runs on every push, and it closes the gap that the
existing double cannot cover by construction.

**Memory is asserted, not assumed.** Streaming that quietly accumulates is worse
than an honest ceiling, so tests measure peak process memory across payloads of
increasing size and assert it does not scale with them.

**Classification is tested against real captured output**, extending the
existing practice. The loss parser was fixed only because real output was
checked line by line against it; progress parsing inherits that discipline —
promotion is asserted against genuine pull and download output, including
partial lines and interleaved streams, not against invented samples.

**Prior art:** the provider tests for the seam-and-double pattern; the event
tests for classification against captured output; the upload-limit tests for
refusing oversized input before reading it.

**The verification clause.** Done means: a real job on real hardware, watched
through the interface, showing image-pull and model-download progress advancing
with a measured rate, ending on a completed page that displays its own final
events, with an artifact downloaded and its checksum verified. This is the
healthy-run cluster and it shares a run with Spec 005's calibration.

## Out of Scope

- **Fan-out of events to many watchers**, and replay after a dropped connection.
  This spec changes what events *are*; distributing them is Spec 008.
- **Converting artifacts to other formats**, and the manifest's evaluation and
  lineage content. Spec 011.
- **Importing datasets from outside an upload.** Streaming validation is written
  so that an imported dataset can go through the identical path, but the import
  itself is Spec 009.
- **Retention and expiry of stored artifacts.** Objects are written and kept;
  lifecycle is Spec 012.
- **Encryption at rest** beyond whatever the configured store provides.

## Further Notes

The superseding decision record matters more than the code change it licenses.
The earlier decision is not being quietly edited — it stated a rule in words
broader than the property it defended, and the honest record is a new entry
saying where it stopped holding and why. That is the format the decision
directory exists for, and an entry that says *"this is where my earlier
reasoning ran out"* is stronger evidence of judgment than one that was right the
first time.

One consequence worth flagging early: promoting progress out of the log removes
the pressure that produced the event cap in the first place, but the cap itself
should stay. A job that genuinely produces very many events should still be
paginated rather than truncated silently, and the interface should say what it
is showing and of how many. Fixing the cause does not make the symptom's
guardrail unnecessary.
