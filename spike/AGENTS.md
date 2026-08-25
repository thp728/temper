# Probes

Infrastructure probes against the live JarvisLabs account. `README.md` records what each spike
proved and what it cost.

**This is a one-time-use directory, kept for reference.** The findings files are the measured
evidence that lets numbers elsewhere in this repository be called measured, and they stay. The
scripts are the investigation that produced them, not product code, and issue #26 collapses the four
near-duplicate bootstraps to the one representing the final approach.

**Code that graduates into the product takes its tests with it.** Streaming logic and
`test_streaming.py` go to `packages/core/`; availability and price filtering and `test_spike5.py` go
to `apps/worker/`. Whatever stays here has no tests, because none of it is product code.

## Running one

**PowerShell, never Bash.** Git Bash cannot see the Windows `ssh-agent`, so the passphrase-protected
key fails and surfaces as `"ssh ready — no answer within 240s"`, which looks exactly like a dead VM.

```powershell
& "d:\Dev\life-os\.venv\Scripts\python.exe" -u spike/spike4.py
```

`ssh-add -l` must list one ED25519 key first.

## Rules

**Every path that creates a VM destroys it in a `finally` block and then confirms by listing
instances.** A destroy call's return value is not evidence. `teardown.py` is the manual sweep.

**Readiness distinguishes *unreachable* from *authentication failed*.** Opposite remedies.
Collapsing them into "no answer" is how an evening was lost and a platform bug wrongly filed.

**Read `account.currency()`.** The account bills in INR, not USD.

**`.env` holds `JL_API_KEY`** and is git-ignored at two levels. Never print it, never commit it.

**Every finding lands in a `findings-spikeN.json`**, recorded as it is measured. A probe whose
result lives only in a terminal scrollback proved nothing.
