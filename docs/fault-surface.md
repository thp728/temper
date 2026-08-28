# The fault surface

The part of the product that makes a failure happen on demand. It ships
*before* the recoveries, deliberately: every recovery path is verified by
something going wrong, and nothing could make something go wrong on demand. A
recovery proven only by reasoning is the same class of artifact as a large
green suite over a product that could not train. This surface is the answer to
*"how do you know your recovery works?"* — a reviewer turns on a fault and
watches what the product does.

The fault surface is not test scaffolding that leaked into the product. It is
the mechanism by which a claim about reliability becomes checkable by someone
who did not write it. It is part of the product, off by default, and its
vocabulary lives in one data file, `packages/contracts/fault-surface.json`,
read by the control plane and the trainer image alike so the two sides cannot
drift about what a fault is called or which code a broken run carries.

## The six faults

| name | side | what it makes happen | code on the run |
| --- | --- | --- | --- |
| `oom` | trainer | exhaust device memory during training | `simulated_oom` |
| `divergence` | trainer | drive the loss to a meaningless value | `simulated_divergence` |
| `worker_kill` | trainer | kill the worker mid-run | (no result document — the run fails as an interruption) |
| `machine_silent` | provider | make the machine go silent without ending | (the stall detector names `gpu_stalled`) |
| `orphan` | provider | leave a machine running with no job that owns it | (the job completes; an unowned machine is left listed) |
| `destroy_refused` | provider | make the provider refuse a destroy | (the job completes; teardown reports the stray machine) |

**`side` is where the fault lives, and it is the spec's constraint taken
literally.** Provider-side faults are behaviour of the existing provider seam —
only the fake provider (`SimulatedMachine`) honours them, so they can never
fire against a real machine. Trainer-side faults are an environment switch the
trainer reads: the control plane writes the job's fault spec into the
trainer's environment on the machine, and the trainer makes the fault real.
No new module, no new boundary — a fault-injection framework would be a seam
introduced to test seams.

## Off by default, and enforced, not hoped for

The surface is off by default, and *off* is a safety property, not a
convenience: a fault surface that can be switched on by accident in front of a
user is worse than none. Two things make accidental use impossible:

1. **Creation refuses a fault spec unless the surface is on.** A job whose
   `simulated_failure_code` is a dict is refused at the single creation path
   (`jobs.create`) with the stable code `fault_surface_refused` unless either
   `TEMPER_FAKE_PROVIDER` (the zero-cost tier) or `TEMPER_FAULT_SURFACE` (the
   deliberate real-hardware tier) is set.
2. **The orchestrator refuses too.** `run_job` refuses a fault-spec job the
   same way if a row slipped past creation — a guard that lives on only one
   side of a money path is a hope.

And the trainer-side switch is gated a third time: the fault spec only reaches
the trainer's environment when the surface is on, so even a fault-spec job
that somehow reached a real machine would carry no instruction to the trainer.

## How to switch a fault on

The surface is configured per job, through the same hyperparameter the
product has always used for its internal keys. The **string** form —
`simulated_failure_code: "gpu_stalled"` — is the reserved early-exit
affordance from before the surface existed, and still works. The **dict**
form is the surface:

```json
{
  "simulated_failure_code": { "name": "oom" }
}
```

Parameters pick where a fault fires: `machine_silent` takes `after_line`,
`destroy_refused` takes `times`, and the trainer-side faults take `delay_s`
(the chosen point in the run). For example, to make a machine go silent after
three lines of output:

```json
{
  "simulated_failure_code": { "name": "machine_silent", "after_line": 3 }
}
```

The two tiers:

- **Zero-cost tier.** `TEMPER_FAKE_PROVIDER=1` swaps the fake machine in; the
  fault fires against the fake, and every test and browser journey in this
  repository runs this way. It proves the orchestration logic — that a fault
  is refused when off, named in the history, and drives the job to the
  documented outcome. It cannot prove the recovery *works*, because a
  simulated failure is not a failure; saying so unprompted is the point.
- **Deliberate real-hardware tier.** Set `TEMPER_FAULT_SURFACE=1` (without
  the fake) and a fault spec reaches a real machine. The trainer-side faults
  genuinely fire — the trainer exhausts device memory, or drives the loss to
  a meaningless value, or kills itself. Provider-side faults cannot fire
  here, by construction: the real provider has no fault-reading code, which
  is the asymmetry the spec intends. This tier costs money and is the
  second tier of Spec 010's testing decision.

## Every injected fault is named in the run's history

A deliberately broken run can never be mistaken for a real one, three ways:

- The orchestrator writes a launch event naming the fault and why it is
  deliberate: *"Simulated fault injected: oom - exhaust device memory during
  training. This run is deliberately broken and cannot be mistaken for a real
  one."*
- The machine's own stream (or the trainer's output) narrates the fault at
  the moment it fires, so it is stamped in the history where it happened.
- A run that produces a result carries the fault's `simulated_` code as its
  `error_code`, and the frozen job record retains the fault spec itself in
  its hyperparameters.

## How to use it to verify a recovery

Turn on the fault whose failure the recovery is meant to handle, launch, and
watch the recovery happen through the job's own history. Concretely:

- **Out-of-memory recovery (#35):** launch with
  `{"name": "oom"}` and watch the run retry with the effective batch
  preserved instead of failing.
- **Divergence (#36):** launch with `{"name": "divergence"}` and watch the
  run stop early with a plain cause.
- **Interrupted-job resumption (#60):** launch with `{"name": "worker_kill"}`
  and watch the run resume from its last checkpoint on a fresh machine.
- **Silent machine (#10 / stall detection):** launch with
  `{"name": "machine_silent"}` and watch the stall detector stop the job.
- **Orphaned machine (#61):** launch with `{"name": "orphan"}` and watch the
  reconciler find and destroy the unowned machine.
- **Refused destroy (#34):** launch with `{"name": "destroy_refused"}` and
  watch teardown retry, escalate and confirm across observations.

A recovery path proven only against the fake is explicitly not done; the
verification clause in Spec 010 requires each failure to have been caused on
real hardware and the recovery observed through the interface. The faults
above are how that clause becomes executable.
