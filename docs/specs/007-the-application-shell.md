# Spec 007 — The application shell

**Status:** ready for tickets
**Phase:** B, band 1 (the demo path — and a prerequisite for verifying everything after it)
**Depends on:** Spec 003 (the pages this ports), Specs 005 and 006 (the contracts it renders)
**Produces:** an ADR that the interface consumes a client generated from the API contract, and that the two interfaces never coexist
**Assumes:** the Phase A interface was deliberately disposable — Spec 003 says so in those words

## Problem Statement

Phase B's acceptance rule is that a flow is not done until it has been exercised
**through the interface, against real compute, immediately after it is built.**
That rule is the direct lesson of Phase A, where a large green test suite
coexisted with a product that could not train, and where every defect was found
by running the assembled thing.

The rule has a consequence nobody has scheduled around: **the interface is now a
gate in front of every remaining flow.** If the interface a flow is verified
through is the one being replaced, each flow gets built twice — once against the
old surface and once against the new — and the second build is the one that
never meets real hardware. That is precisely the failure mode the rule exists to
prevent, arrived at by a different road.

The current interface is five server-rendered pages with no build step, written
deliberately as throwaway. They work: a full journey was walked through them on
real hardware. They are also the wrong foundation for what follows, because
every remaining flow adds surface — a plan with reasons and editable decisions,
live per-phase progress with corrected estimates, a compatibility report, an
advanced configuration form derived from a schema, an evaluation comparison —
and none of that is a form post and a page reload.

## Solution

**A thin shell, ported plainly and early, that every later spec extends.**

Not a redesign. The first delivery is the existing five screens moved across
with their behaviour intact and nothing added: upload and validation report,
model choice and launch, the job list, the live job view, the finished job. It
exists to be the surface that the rest of Phase B is verified through, and it
earns its place by existing before the flows do, not by being better than what
it replaces.

Three properties make it a foundation rather than a rewrite:

- **The client is generated from the API contract**, so the interface cannot
  drift from the server silently. A field that changes shape breaks a build
  rather than a page.
- **The two interfaces never coexist.** The old pages are deleted in the same
  change that ports the last screen. A half-ported interface leaves a reader
  asking which one is real, and leaves the team maintaining both.
- **One surface with progressive disclosure**, not a beginner mode and an expert
  mode. The plan a first-time user reads as an explanation is the same plan an
  experienced user edits. Two products cannot be built in the time available and
  should not be built at all — the explanation *is* the product, and hiding it
  from experts or hiding the controls from beginners breaks it in both
  directions.

## User Stories

1. As a user, I want the whole journey available in one application, so that I do not move between differently built pages to complete one task.
2. As a user, I want to upload a dataset and see its validation report, so that I know whether I can proceed.
3. As a user whose dataset was rejected, I want each problem shown against its line number, so that I can fix the file without guessing.
4. As a user, I want to see a preview of how my rows were understood, so that I can confirm the schema was read correctly.
5. As a user, I want to choose a base model and see its licence and pinned revision, so that I know what I am training and under what terms.
6. As a user, I want to review the plan before launching, so that there are no surprises after I commit to spending.
7. As a user, I want a single obvious action to launch, so that starting a job is unambiguous.
8. As a user, I want the job to appear immediately after launching, so that I am not left wondering whether it started.
9. As a user watching a job, I want its state shown prominently, so that I can tell at a glance where it is.
10. As a user watching a job, I want live output without refreshing the page, so that watching is watching rather than polling by hand.
11. As a user watching a job, I want a cancel control that is clearly destructive, so that I do not stop a run by accident.
12. As a user, I want to leave the page and come back to a running job, so that watching is not a commitment to stay.
13. As a user returning to a running job, I want to see what I missed while I was away, so that the history is continuous rather than starting where I rejoined.
14. As a user, I want to see all my jobs in one list with their outcomes, so that I can find a run from yesterday.
15. As a user, I want a finished job's page to show how it ended, so that the record is complete.
16. As a user, I want to download what a finished job produced, so that I can use it.
17. As a user whose job failed, I want to read why in plain language, so that I know what to change.
18. As a user on a small screen, I want the application to remain usable, so that I can check a running job away from my desk.
19. As a user navigating by keyboard, I want every control reachable and labelled, so that the product does not require a mouse.
20. As a user, I want detail available but collapsed, so that the interesting information is not buried under machine output.
21. As an experienced user, I want the controls for a decision beside the explanation of it, so that I do not have to leave the plan to change it.
22. As a developer of this product, I want the client generated from the server's contract, so that a change to a response shape fails a build rather than a page.
23. As a developer of this product, I want one interface rather than two, so that a fix does not have to be made twice.
24. As a reviewer, I want the application to start from a cold clone with one command, so that evaluating it does not begin with a debugging session.

## Implementation Decisions

**Ported before extended.** The first change set moves the five existing screens
across with equivalent behaviour and no new capability. Subsequent specs add
their own surfaces to this shell, and each of those surfaces ships in the same
change as the flow it exposes — a flow and its interface are one deliverable,
because the acceptance rule cannot be met by either alone.

**The old interface is deleted in the change that ports the last screen.** Not
before, so there is never a gap in the journey; not after, so there is never a
window where both exist. Two interfaces in a repository is a question a reviewer
will ask and there is no good answer to it.

**The client is generated from the API's published contract**, not hand-written.
This is the mechanism that keeps the two halves honest, and it is why the API's
response models matter more than they did when the only consumer was a template
rendered in the same process.

**Live updates arrive over a server-pushed stream**, replacing the poll the
current watch page performs. The stream's fan-out, persistence and replay after
a dropped connection are Spec 008's; this spec consumes the endpoint and renders
what arrives, including reconnecting when the connection drops.

**The plan page is the centre of the application**, not the form. The existing
flow asks the user to choose and then launches. After Spec 005 the user asks for
an outcome and the product proposes a plan; this shell renders that plan with
each decision, its reason, its rejected alternatives, and an inline control to
change it. Editing one decision re-requests the plan rather than mutating it
locally, so the recomputation rules live in one place.

**Progressive disclosure is a rendering rule, not a mode.** Every decision shows
its value and reason by default; its alternatives and its control are one
interaction away. There is no toggle that changes what the product is, because a
toggle means two surfaces to build, test and verify.

**Accessibility is part of the port, not a later pass.** Labelled controls,
keyboard reachability, focus management on navigation, and status changes
announced. Retrofitting is more expensive than doing it once, and this shell is
small enough that doing it once is cheap.

**No cosmetic redesign in this spec.** Layout, spacing and components come from a
conventional component library used plainly. The time budget is spent on the
surfaces later specs add, not on visual identity.

## Testing Decisions

**A good test here drives the application the way a person does** — find a
control by its accessible name, act on it, assert on what the user can then see.
Tests that reach for internal structure will break on every layout change and
protect nothing.

**Browser tests cover the journeys, against the zero-cost provider.** Upload and
proceed; upload and be rejected with line numbers; launch and watch a job to
completion; cancel a running job; land on a finished job and download; land on a
failed job and read the reason. These run without hardware and without money,
which is what makes them runnable on every push.

**The generated client is verified by building against the real contract**, so a
server change that breaks the interface fails in the pipeline rather than in
front of a user.

**What these tests explicitly do not prove:** anything below the transport seam.
The zero-cost provider does not cross a connection, and the class of defect that
cost this project a run lives exactly there. Spec 006's transport tier covers
that gap, and this spec's tests should not be described as covering it.

**Prior art:** the existing interface tests are the closest model for what to
assert — the rejected-dataset report naming its lines, the frozen job spec being
visible before launch, the terminal state rendering. Those assertions carry over
almost unchanged, which is a good sign that the port is behaviour-preserving.

**The verification clause.** Done means the five journeys walk end to end in a
browser against real hardware for at least one job, and the old pages are gone
from the repository. Every subsequent spec's verification runs through this
shell, which is why it is early rather than late.

## Out of Scope

- **Event fan-out, persistence and replay.** Spec 008.
- **Every surface that a later flow introduces** — the compatibility report, the
  advanced configuration form, the evaluation comparison. Each ships with its
  flow.
- **Visual design and brand.** Conventional components, used plainly.
- **Internationalisation.** Single language.
- **Authentication and anything that follows from it.** The brief sanctions the
  cut and the product is single-tenant by design.

## Further Notes

The strongest argument for doing this early is also the least comfortable one:
this spec produces no new capability, and it is on the critical path anyway. It
is worth naming that plainly rather than justifying it as progress — the reason
it goes early is that the acceptance rule for every other spec depends on it,
and a rule that cannot be met is a rule that will quietly be abandoned around the
twenty-ninth.

The counter-consideration, recorded because it was a real disagreement: the
existing pages work, were walked through a full journey on hardware, and cost
nothing to keep. An argument exists that a build step and a second toolchain
make a cold clone worse for the reviewer who is going to run this, and that the
days would buy more as flows. That argument was heard and rejected: a
production-grade platform is not server-rendered templates emitted by its own
API, and the shell is where every remaining flow's surface has to live. The
mitigation for the cold-clone cost is that starting the whole stack must remain
one command, which Spec 012 owns and treats as a gate.
