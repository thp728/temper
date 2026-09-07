"""Serve one tuned model for the temporary endpoint (ADR-0076).

The endpoint used to provision a machine, verify its inference port was not
reachable, and then answer every prompt with `f"[{model_id}] tuned response
to: {prompt}"`. The machine billed at the GPU's hourly rate and never loaded
a model. This is the generation that was missing.

**It ships to the machine at endpoint start, not inside the image.** The
trainer image already carries torch, transformers and peft, so this needs
nothing installed; pushing the script the same way the dataset travels
(`provider.push_stream`) means serving works against the currently published
digest instead of waiting on a republish. It belongs in `TRAINER_SOURCES`
eventually, and then this file is baked in and the push goes away.

**Bound to localhost.** The port is never published off the machine: the
control plane reaches it by running a command over its existing SSH channel,
and `serving.verify_not_reachable` already refuses to hand out a key if the
port answers from outside. Binding to 127.0.0.1 means that check is enforcing
a property this process also asserts, rather than resting on the firewall
alone.

The model loads once, at startup, and stays resident. That is the whole
reason the endpoint holds a machine: a load per request would cost about a
minute of the user's money for every prompt.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# The decoding settings the run's own comparison used, so a prompt answered
# here and a prompt answered in the evaluation are answered the same way.
# Restated as a literal only because this file is pushed to the machine
# rather than imported with the package; `comparison.COMPARISON_DECODING` is
# the definition, and the two are asserted equal by the control plane's tests.
DECODING = {"do_sample": True, "temperature": 0.7, "max_new_tokens": 128}

# The control plane always passes the port, and `serving.INFERENCE_PORT` is
# where that number is defined -- it is the one the isolation check probes
# from outside. The literal here is only the fallback for running this file
# by hand, and a control-plane test asserts the two agree.
# The thinking mode the run trained under, passed in rather than guessed.
# Absent means False, which is `entrypoint.build_config`'s own default for
# the same unknown.
#
# Spread into `apply_chat_template` as a keyword rather than handed to it as
# `chat_template_kwargs=`. Qwen3's template reads a top-level variable:
#
#     {%- if enable_thinking is defined and enable_thinking is false %}
#
# and `is defined` is the whole difficulty. A keyword lands in the render
# context on every version of transformers; `chat_template_kwargs` is a
# newer parameter, and where it is not recognised it becomes a template
# variable of that name while `enable_thinking` stays undefined -- the test
# above then fails open and thinking stays on, silently.
#
# Measured, not reasoned: the first real endpoint was told
# TEMPER_ENABLE_THINKING=0 and its answer opened with a thinking block all
# the same. See docs/final-pass-findings.md.
ENABLE_THINKING = {
    "enable_thinking": os.environ.get("TEMPER_ENABLE_THINKING") == "1"
}

HOST = "127.0.0.1"
PORT = int(os.environ.get("TEMPER_SERVE_PORT", "8000"))


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load():
    """Load the base model with the adapter applied, ready to generate.

    Imports are inside the function because they come from the base image,
    never from this repo (ADR-0010). The adapter was trained against a 4-bit
    base; applying it to a bf16 base is the ordinary inference setup and the
    same one `load_comparison_generators` uses, so what the endpoint answers
    matches what the comparison reported.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = os.environ["TEMPER_BASE_MODEL"]
    revision = os.environ.get("TEMPER_BASE_REVISION") or None
    adapter_dir = os.environ.get("TEMPER_ADAPTER_DIR", "/adapter")

    _log(f"loading base {repo} at {revision or 'default'}")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=torch.bfloat16, device_map="auto"
    )
    _log(f"applying adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    _log("model ready")
    return model, tokenizer


class _Handler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        # Readiness, so the control plane can wait for the weights to land
        # rather than guessing at a sleep.
        if self.path == "/health":
            self._reply(200, {"ready": self.model is not None})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path != "/generate":
            self._reply(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._reply(400, {"error": "body is not JSON"})
            return
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._reply(400, {"error": "prompt must be a non-empty string"})
            return
        try:
            self._reply(200, {"completion": self._generate(prompt)})
        except Exception as e:  # noqa: BLE001 - the reason must reach the user
            self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    def _generate(self, prompt: str) -> str:
        import torch

        # `enable_thinking` is the run's own, passed in by the control plane
        # from the record the trainer wrote. `build_config` says it beside
        # the value it sets: the SAME value must be applied at serving.
        # Rendering under the other one is a train/infer mismatch that
        # produces plausible output, so nothing would flag it.
        #
        # `add_generation_prompt=True` is the one deliberate difference from
        # the comparison's `_ModelGenerator`, which omits it. Here the whole
        # conversation is one user turn and the model must answer it; without
        # the generation prompt the model emits the assistant header itself
        # and the decoded text begins with a stray "assistant".
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **ENABLE_THINKING,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            output = self.model.generate(**inputs, **DECODING)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def log_message(self, fmt: str, *args) -> None:
        # The default handler writes to stderr per request; the control plane
        # reads that channel, so keep it to something it can classify.
        _log("serve " + (fmt % args))


def main() -> int:
    model, tokenizer = _load()
    _Handler.model = model
    _Handler.tokenizer = tokenizer
    server = HTTPServer((HOST, PORT), _Handler)
    _log(f"listening on {HOST}:{PORT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
