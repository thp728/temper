# Spec 012 — Clone and run

**Status:** ready for tickets
**Phase:** B, band 1 for the parts that gate evaluation; band 2 for the rest
**Depends on:** Spec 007 (the shell that starts), Spec 008 (the services that start with it)
**Produces:** ADR-0026 (the zero-cost path is a labelled demonstration and is structurally blind to the transport)
**Assumes:** the repository becomes public at submission, and the private working vault does not

## Problem Statement

Someone who did not write this is going to clone it and try to run it, and the
first ten minutes decide what they think of everything after.

Every fact about the current state points the wrong way:

- **The system has never started on a machine that is not the author's.** It has
  never run on anything but Windows. The environment documentation is entirely
  about Windows-specific traps — which shell can see the key agent, which
  encoding the interpreter defaults to, which line endings a remote shell
  receives. A reviewer will be on Linux or macOS, and **the defect that cost this
  project a real run was a line-ending bug on exactly that boundary.** There is
  no evidence the product works there, and the one class of bug already proven to
  live at that seam is the class most likely to be waiting.
- **Running it requires an account with the compute provider**, a key, a
  registered key pair, and a willingness to spend. Any of those failing quietly
  and the evaluation ends at the setup step, and what the reviewer remembers is
  that it did not run.
- **There is no licence.** A public repository without one grants nothing, and
  a fine-tuning platform whose own artifacts carry licence obligations that it
  surfaces to users, while carrying none itself, is an awkward thing to be asked
  about.
- **The probe directory has four near-duplicate scripts** left from the
  investigation that produced them. They were the right artefact then; they read
  as clutter now.
- **The decision record starts three-quarters of the way through the project.**
  Thirteen decisions made during the first week live in a private vault, and the
  public directory says in writing that they are copied in before publication.
  Until they are, the record appears to begin late.
- **The interface's front page assumes you already have hardware.**

## Solution

**Two front doors, both honest about what they are.**

The first is the real one: the whole system starts with one command, reads
everything it needs from the environment, and runs real jobs on real hardware
for someone with an account. The setup that requires a human — obtaining a key,
registering a key pair, knowing which shell can reach the agent — is documented
as the sequence it actually is, including the trap that cost an evening here,
because a reviewer hitting it will not have an evening to lose.

The second is the zero-cost one. The existing double already implements the whole
compute interface — scripted output, staged failures, controlled timing, held
pauses — and is a configuration switch away from being a front door. Someone with
no account, or no wish to spend, can start the system, upload a dataset, watch a
job progress, and download a result, in minutes.

**And it says what it is, in the interface, not only in a file nobody reads.**
A demonstration mode that looks like a real run is the single most dishonest
thing this repository could ship. Every page in that mode is marked, and the
documentation states the specific limitation plainly: **the double never crosses
a connection, so it is structurally blind to transport defects — and a transport
defect is exactly what broke a real run here.** Volunteering that is worth more
than the mode itself, and it is the direct answer to a question a reviewer will
ask.

Around those two doors, the things that make a repository readable by someone who
was not in the room: a licence, a README that says what the product is and what
is deliberately absent, an architecture diagram from dataset to served model, the
decision record complete from its first entry, a probe directory reduced to what
it proves, and a security review before any of it is public.

## User Stories

1. As a reviewer, I want one command to start the whole system, so that evaluating it does not begin with a debugging session.
2. As a reviewer, I want it to start with no configuration, so that I can see it work before I decide whether to invest in setting it up.
3. As a reviewer without a compute account, I want to walk the entire journey at no cost, so that I can evaluate the product rather than the provider.
4. As a reviewer in the zero-cost mode, I want every page to tell me it is a demonstration, so that I never mistake it for a real run.
5. As a reviewer, I want to be told exactly what the demonstration cannot prove, so that I can weigh what I have seen.
6. As a reviewer with an account, I want the real setup documented as a sequence, so that I can get to a real job without guessing.
7. As a reviewer with an account, I want the known environment traps named, so that I do not lose an evening to one that is already known.
8. As a reviewer, I want to know what a real run will cost before I start one, so that spending is a decision rather than a surprise.
9. As a reviewer on Linux, I want it to work, so that the product is not accidentally tied to the author's machine.
10. As a reviewer on macOS, I want it to work, for the same reason.
11. As a reviewer, I want a sample dataset included, so that I do not have to invent one to try it.
12. As a reviewer, I want to know what is deliberately not built and why, so that a gap reads as a decision rather than an oversight.
13. As a reviewer, I want the remaining work visible as tracked items, so that "not finished" is specific rather than vague.
14. As a reviewer, I want the decision records complete from the first decision, so that the reasoning does not appear to start late.
15. As a reviewer, I want each decision's rejected alternatives, so that I can tell a decision from an assertion.
16. As a reviewer, I want a diagram of the whole path, so that I can understand the shape before reading code.
17. As a reviewer, I want to know which numbers were measured and which were derived, so that I can tell proven from calculated.
18. As a reviewer, I want the probe directory to contain what it proves and nothing else, so that reading it is fast.
19. As a reviewer, I want a licence, so that I know what I may do with this.
20. As a reviewer, I want no secret anywhere in the history, so that publishing was safe.
21. As an operator, I want stored data to have a stated retention, so that objects do not accumulate forever.
22. As a user, I want the product to tell me when it is not connected to real compute, so that I am never confused about what I am looking at.

## Implementation Decisions

**One command starts everything**, with local defaults that require nothing to be
set. Whether real compute is used is one setting; nothing else changes between
the two modes.

**The zero-cost mode is the existing double promoted, not a second
implementation.** It already satisfies the compute interface. Promoting it means
selecting it by configuration, seeding a dataset and a plausible run so the
journey has something to show, and marking every page it serves.

**The marking is in the interface and cannot be turned off in that mode.** A
banner that a screenshot would not include is not a marking.

**The documented limitation is specific, not general.** Naming the transport
defect that the double could not have caught is more useful than a caution about
simulations, and it is the honest version of the claim.

**A cold clone on a fresh machine that is not the author's is a gate, not a
document.** Provision a plain machine, clone from the published repository, start
it, run one real job, and record every step the documentation failed to mention.
This is the same discipline that found every defect worth finding in this project
— run the assembled thing — applied to the part of it a reviewer meets first.

**The probe directory keeps what it proves.** The near-duplicate bootstrap
scripts collapse to the one that represents the final approach; the findings
files stay, because they are the measured evidence and the reason the numbers in
this repository can be called measured at all.

**The decision records are completed before publication.** The thirteen earlier
entries are copied into the public directory in their original order, as the
directory's own index already promises. They are edited for a public audience but
not rewritten — an entry that reconstructs its reasoning after the fact is the
thing this project has consistently said fails under questioning.

**The remaining work becomes tracked items, written for a stranger.** Not a list
in a document — items with the same care as the ones already completed, so that
*"the rest is in the tracker"* is an answer rather than a deflection. Those items
are also the first draft of what another month would buy.

**A security review runs before anything is public**, covering the history as
well as the working tree.

**Retention is stated and enforced** for datasets, checkpoints and artifacts, so
that storage has a policy rather than a habit.

## Testing Decisions

**A good test here is somebody else's machine.** Most of this spec cannot be
proven by a test in this repository, and pretending otherwise would repeat the
mistake the whole project has been correcting for.

**What automation covers:** the browser journeys run against the zero-cost mode
on every push, which means the demonstration path is continuously verified and
cannot rot; the pipeline builds the images the documentation tells a reader to
build; a check asserts that no secret pattern appears in the tree.

**What only a cold clone covers:** that the documented sequence is complete, that
the platform-specific traps are the only ones, that one command genuinely starts
everything from nothing, and that a real job runs from a fresh checkout on an
operating system the author does not use. **This is the verification clause and
it is scheduled, not aspirational** — before the final two days, so that what it
finds has somewhere to be fixed.

**Prior art:** the interface tests already run against the double and are the
foundation of the demonstration mode's coverage. The probe findings files are the
model for how a cold-clone run should be recorded — measured, dated, and specific
about what was and was not proven.

## Out of Scope

- **Deploying anywhere.** The stack is written to be deployable and is
  deliberately not deployed; how it would be done is answered in writing rather
  than in infrastructure.
- **Supporting operating systems beyond the three a reviewer plausibly uses.**
- **A hosted demonstration.** The zero-cost mode runs on the reviewer's machine.
- **Publishing the working vault.** The research material and the
  project-management record stay private. What becomes public is the decision
  records, the specifications, and the measured findings.
- **Packaging for distribution.** Cloning and running is the interface.

## Further Notes

The uncomfortable observation worth writing down: this spec's most valuable
outcome is a list of things that were wrong, produced by a single cold clone on a
machine that is not the author's. Every significant defect in this project was
found by running the assembled thing rather than by reasoning about it, and the
assembled thing has never once been run by anyone else, anywhere else. Budgeting
a day for that is not contingency — it is the highest-yield hour-for-hour
activity in the entire phase, and it is the one most likely to be sacrificed if
the schedule slips.

The second note concerns the demonstration mode, because there is a real argument
against building it: it shows a reviewer something that is not real, and a
reviewer who only ever runs that mode has evaluated a puppet show. That argument
is answered by the marking and by the stated limitation, not dismissed. The
reason to build it anyway is that the alternative — a reviewer whose key is not
registered at eleven at night — ends with nothing seen at all.
