# Decision records

Every decision that shapes this platform, with the alternatives that were
rejected and why. Written **as each change lands, not afterwards** — an entry
written during the build records reasoning, an entry written after it
reconstructs reasoning, and reconstruction is what fails under questioning.

## Why the numbering starts at 0001 partway through the project

It does not start partway through. This directory was created on 2026-08-21,
about three-quarters of the way into the build, and that is the honest reason
the first file here is dated so late — but it is not where the record begins.

The **first thirteen decisions were logged in a private working vault** as they
were made, between 2026-08-14 and 2026-08-19: the choice of Axolotl over
calling TRL and PEFT directly, pinning the trainer image by digest, locking the
correctness settings, detecting thinking mode from the dataset, and the rest.
That log is closed to new entries and stays as the record of the build to that
date. **Its entries are copied into this directory before the repository
becomes public**, in their original order, so the record does not appear to
begin three-quarters of the way through.

Until that copy happens, this index lists only the decisions made after the
directory existed. Anything referenced here that is not yet a file is in the
vault.

Records are written for a public audience from the first entry, because the
repository becomes public at submission.

## Index

| ADR | Decision | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-event-channel-over-ssh-stdout.md) | The event channel is the machine's stdout, pulled over SSH | accepted | 2026-08-21 |
| [0002](0002-stall-detection-and-a-duration-ceiling.md) | A stalled job and an over-long job are stopped separately, and named separately | accepted | 2026-08-21 |
| [0003](0003-cancellation-is-destructive.md) | Cancelling destroys the machine, produces no adapter, and is not a failure | accepted | 2026-08-21 |

## Format

Each record carries: the context that forced the decision, the decision itself,
**the alternatives considered and why each was rejected**, the consequences
accepted, and how to roll it back. The alternatives section is the point of the
document — a decision recorded without its rejected options is an assertion, not
a record.

A decision is not edited once accepted. Changing one means writing a new record
that supersedes it, so that the reasoning at the time stays legible.
