# ADR-0025 — Phase A closes from a browser, and the test double's blind spot

- **Status:** accepted
- **Date:** 2026-08-21

> **A note on the number and the date.** This decision was recorded on
> 2026-08-21 in the private working vault where the first thirteen decisions
> were logged, and copied into this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made. The wording is
> the original: private-vault links are replaced by descriptions of what they
> pointed at, and module names are given as the record wrote them — where a
> file has since moved (for example the event parser now lives in
> `packages/core`), the record below describes the state that produced the
> decision.

## Context

**The checkpoint question is answered from the product rather than from the suite.**
`job_05300098085f` went upload → validation report → model choice → live watch →
adapter download entirely through server-rendered pages: L4, **5m 38s** wall clock, train
165.2s, three checkpoints, adapter **132.19 MB** with the sha256 matching the one the
container computed. VM destroyed, and the account confirmed afterwards to hold no Temper
instance.

**It took two attempts, and the failure is the more useful half.** Both defects were
invisible to a green 195-test suite, and neither was findable without real hardware.

### The transport defect, and why the double could not see it

The first launch (`job_22f7df957c00`) died 86s in with `bash: line 1: $'\r': command not
found`. `provider.py` opened the SSH pipe with `text=True`, so Python's `TextIOWrapper`
translated every `\n` into `\r\n` on the way to the remote `bash -s`. A **regression from
the streaming work itself** — the old blocking path fed bytes; `stream()` introduced
text-mode stdin, and the suite went on passing.

**The generalisable point, and the one to volunteer under grilling:** `fake_provider`
implements the `Provider` protocol faithfully and **never crosses a pipe**, so the
translation that breaks every real run cannot happen to it. *A double proves the logic above
the seam and is structurally blind to the seam itself.* The remedy adopted is not "trust the
double less" but a test that drives a real subprocess and asserts the bytes arriving on the
far side are the bytes that went in — cheap, and it fails without the fix.

### The parser defect: the format was assumed from the wrong library

The event parser was written against plain transformers, which logs live values
(`{'loss': 1.9042}`). **Axolotl formats every value to fixed precision first and logs the
strings** (`{'loss': '0.7157'}`), so the run produced 24 loss lines and **zero** metric
events. Fixed by accepting a quoted number, with the closing quote required so a value that
merely starts with digits cannot be truncated into a measurement. Of the 24 lines, 23 promote
and the 24th is the `train_loss` summary, excluded exactly as designed — **the design held;
only the assumption about who formats the line was wrong.**

**Decision within the decision:** no third run was bought to watch the curve render. The
parser is verified against the run's own captured output, which is the same evidence a live
render would produce, for ₹5 less. ⚠️ **The honesty constraint this creates:** say *verified
against real output*, never *proven live*. If asked directly whether the chart has been seen
working end to end, the answer is no.

## Decision

Called the checkpoint met on evidence from the product rather than from the suite.

## Alternatives considered

Run a third job to prove the curve live (rejected — ₹5 and six minutes for evidence already held in the captured lines); fix the log truncation first and prove both in one run (rejected with it, same reasoning); ship the log fixes today (rejected — the noise question is a product judgment about what a user is allowed to see, and it deserves a decision rather than a reflex).

## Consequences

The loss chart is the one Phase A claim resting on inference rather than observation. Two known log defects ship into Phase B: the finished-job page truncates at 500 events and so never shows its own outcome, and 718 of 725 events on a real run are BuildKit noise.

## Rollback

Both log items are tracked Phase B work with the fix for the first one sketched; neither is load-bearing for the loop. If the curve turns out not to render, one ₹5 run settles it and the parser fix is a single regex.
