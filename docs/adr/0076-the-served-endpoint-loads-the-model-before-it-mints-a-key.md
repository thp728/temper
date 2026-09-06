# ADR-0076 - The served endpoint loads the model before it mints a key

- **Status:** accepted
- **Date:** 2026-09-06
- **Supersedes:** nothing. Completes [0065](0065-a-served-endpoint-stops-itself.md), which specified the endpoint's lifecycle and left the generation unimplemented.
- **Spec:** Spec 011 (try your model)
- **Issue:** none - found during the pre-submission hardware pass and recorded in `docs/final-pass-findings.md`.

## Context

Starting an endpoint provisioned a real L4 at 41.31 INR/hr, verified its
inference port was not reachable from outside, minted a key, and then
answered every prompt with:

```
[qwen3-4b] tuned response to: How do I connect my weather station to Wi-Fi?
```

The prompt, echoed inside a template. No model was ever loaded. The machine
existed, billed, and held the string together. The comment beside that line
described "the real inference path on real hardware", and there was no
branch, so anyone reading the source to check was told the opposite of what
the code did.

Spending a user's money to return their own prompt is the one outcome that
cannot be defended. The comment claiming otherwise is what turned a missing
feature into a misrepresentation.

Closing it took five decisions, and none of them is obvious.

## Decision

**The model server ships to the machine at endpoint start, not inside the
image.** `apps/trainer/serve.py` is pushed with `provider.push_stream`, the
same way a dataset travels to a training machine, and run under the
published trainer image, which already carries torch, transformers and peft.
Baking it in would be tidier and is where this belongs eventually
(`TRAINER_SOURCES`), but it would mean no endpoint could serve until the
image is rebuilt, republished and its digest merged. The push works against
the digest already published.

**The inference port is reached over the provider's own channel, never
published.** `serve.py` binds 127.0.0.1, and both the readiness probe and
the generation run as commands on the machine. The endpoint's own
`verify_not_reachable` refuses to issue a key if that port answers from
outside, so reaching it any other way would contradict the property that
check enforces. This costs an SSH round trip per prompt, which against a
generation measured in seconds is not the expensive part.

**The key is minted only after the model answers.** `start_endpoint` waits
for the server's `/health` to report ready, and a machine that never gets
there is destroyed through the confirmed teardown path with the start
refused as `endpoint_model_not_ready`. Readiness is asked of the server
rather than timed: how long weights take to download and load depends on
the base model and the network, so a sleep would either waste the user's
money or hand out a key the endpoint cannot honour.

**The endpoint stores its machine's SSH handle** (migration
`0008_endpoint_machine_handle`). A training run holds its `Machine` in
memory for the whole run, so the handle never needed a home. An endpoint is
started by one request and asked for a completion by another, and
`list_machines` reports ids and names but not handles, so a handle that is
not written down cannot be recovered. Without it, inference has a machine id
and no way to reach the machine.

**The endpoint renders under the run's own thinking mode and decoding.**
`build_config` says it beside the value it sets: Qwen3 emits thinking blocks
through its own template by default, the trainer detects from the dataset
whether the training rows had them, and the same value must be applied at
serving. The control plane reads it from the run's record and passes it in.
Serving under the other one is a train/infer mismatch that produces
plausible output, so nothing downstream would flag it.

**The endpoint decodes the way the comparison did.** `serve.py` restates
`COMPARISON_DECODING` as a literal because it is pushed rather than imported
with the package, and a control-plane test asserts the two are equal. An
endpoint answering at a temperature the prediction-versus-actual panel never
used would make "base answered X, tuned answered Y" mean something different
on each screen.

## Alternatives considered

- **Say on the screen that serving is not built, and stop provisioning.**
  Defensible, and it was the other honest option. Rejected because the
  endpoint is Spec 011's deliverable and the machine it holds is exactly
  what makes a tuned model touchable; documenting the gap would have left
  the product's most persuasive screen as a placeholder.
- **Bake `serve.py` into the trainer image now.** Rejected for this change
  only, and on timing rather than principle: it blocks every endpoint behind
  a republish. The file carries a note saying where it belongs.
- **Publish the inference port and call it directly from the control
  plane.** Rejected. The platform's firewall does not filter published
  container ports the way it appears to, which is the finding the
  `verify_not_reachable` check exists for. A published port would be an open
  door with a key check in front of it.
- **A fixed sleep before minting the key.** Rejected. It is wrong in both
  directions, and the direction that costs money is the one that also hands
  out a key to a server that cannot answer.
- **Load the model per request instead of holding it resident.** Rejected.
  Loading a 4B base takes about a minute of the user's money, per prompt,
  and holding it resident is the whole reason the endpoint holds a machine.
- **vLLM or TGI instead of a `BaseHTTPRequestHandler`.** Rejected for now.
  Either is the right answer for throughput, and neither is in the published
  image; a stdlib server that loads the same way the comparison loads adds
  no dependency and matches what the run already reported. Throughput is not
  what a single-user "try your model" endpoint is short of.

## Consequences

- `start_endpoint` now takes minutes rather than seconds against real
  hardware: it pulls the image and downloads the base weights before the
  key exists. The preview's cost and stop-time figures already describe a
  billed machine, so this is time the user was told about, but the interface
  shows no progress during it.
- The warm-up runs only when `provider.is_remote`. The simulated tier has no
  machine to load anything on and keeps the canned completion, so the
  hardware-free journeys are unchanged.
- A generation that fails answers 502 with `endpoint_generation_failed` or
  `endpoint_machine_unreachable`, not 400: the request was well formed and
  authorised, and what failed was the machine behind it.
- Endpoint rows created before migration 0008 have no handle and refuse to
  serve rather than shelling out to `ssh` with no destination. Stopping and
  restarting the endpoint is the remedy, and it is what the message says.
- `serve.py` is not in `TRAINER_SOURCES`, so it is the one piece of trainer
  code not pinned by the image digest. What runs on a serving machine is the
  file in the working tree of the control plane that started it.
- Nine tests cover this, including the one that would have caught the
  original defect: a completion from a remote provider must not contain the
  prompt. None of it has run against real hardware yet, and until it has,
  "implemented" means the tests pass and nothing more.
