"""Runs INSIDE the pinned trainer image. Prints the config schema as JSON.

Kept as its own file rather than a heredoc inside spike7.py so it can be read,
diffed and reasoned about. It is piped to `python -` in the container, so it
must depend on nothing but the standard library and whatever Axolotl already
brings.

It never raises: an import failure is a finding ("the schema is not reachable
from the image"), and a traceback on stderr would be a worse way to record it.
"""

from __future__ import annotations

import json
import sys
import time
import typing

MARKER = "---SPIKE7-JSON---"

# Axolotl has moved this model between releases. Try the known locations in
# newest-first order and record which one answered -- the location is itself a
# fact the Advanced-mode code will have to pin.
CANDIDATE_PATHS = [
    ("axolotl.utils.schemas.config", "AxolotlInputConfig"),
    ("axolotl.utils.config.models.input.v0_4_1", "AxolotlInputConfig"),
    ("axolotl.utils.config.models.input.v0_5_0", "AxolotlInputConfig"),
    ("axolotl.utils.schemas.config", "AxolotlConfigWCapabilities"),
]

# Fields that configure the machine rather than the training. Excluded from the
# "training relevant" count because exposing them in Advanced mode would be
# exposing the platform's own plumbing to the user -- the output directory is
# the platform's business, not a hyperparameter.
INFRASTRUCTURE_PREFIXES = (
    "output_", "dataset_prepared_", "hub_", "wandb_", "mlflow_", "comet_",
    "use_wandb", "use_mlflow", "use_comet", "use_tensorboard", "hf_",
)
INFRASTRUCTURE_EXACT = {
    "output_dir", "dataset_prepared_path", "resume_from_checkpoint",
    "logging_steps", "save_steps", "eval_steps", "save_total_limit",
    "save_strategy", "eval_strategy", "evaluation_strategy", "hub_model_id",
    "hub_strategy", "push_dataset_to_hub", "debug", "seed_everything",
    "local_rank", "world_size", "is_preprocess", "strict",
    "tokenizer_legacy", "trust_remote_code", "base_model_config",
}

# The correctness settings AGENTS.md names as default-locked. Each is looked up
# under every name Axolotl might carry it as; finding NONE of them is a real
# finding, because it means the setting is not expressible in the config at all
# and Advanced mode cannot expose it however wide the form gets.
CORRECTNESS_SETTINGS = {
    "chat template resolution": ["chat_template", "chat_template_jinja"],
    "EOS handling": ["eos_token", "special_tokens", "added_tokens_overrides"],
    "loss masking (train on assistant turns only)": [
        "train_on_inputs", "roles_to_train", "train_on_eos", "message_property_mappings"],
    "NF4 double quantisation": [
        "bnb_config_kwargs", "load_in_4bit", "adapter", "bnb_4bit_quant_type"],
    "all-linear LoRA targets": [
        "lora_target_linear", "lora_target_modules", "peft_layers_to_transform"],
    "bf16": ["bf16", "bfloat16", "fp16", "tf32"],
    "seed": ["seed"],
    "LoRA rank": ["lora_r"],
    "LoRA alpha": ["lora_alpha"],
    "rsLoRA": ["peft_use_rslora", "lora_use_rslora"],
    "FSDP / sharding": ["fsdp", "fsdp_config", "fsdp_version", "deepspeed"],
    "sequence length": ["sequence_len", "max_seq_length"],
}


def is_infrastructure(name: str) -> bool:
    return (name in INFRASTRUCTURE_EXACT
            or name.startswith(INFRASTRUCTURE_PREFIXES))


def type_name(annotation) -> str:
    try:
        return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")
    except Exception:
        return "?"


def is_free_form(annotation) -> bool:
    """Free-form means a generated form has nothing to render but a text box."""
    text = type_name(annotation)
    if text in ("Any", "?", "NoneType"):
        return True
    # A bare dict or a dict of Any is a bag of keys the schema says nothing
    # about, which is exactly the case a generated form cannot help with.
    return bool(("dict" in text.lower() and "Any" in text)
                or text.lower().startswith("dict")
                or text.startswith("Optional[Any"))


def enum_values(annotation) -> list[str] | None:
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return [str(a) for a in typing.get_args(annotation)]
    for arg in typing.get_args(annotation) or ():
        vals = enum_values(arg)
        if vals:
            return vals
    if isinstance(annotation, type) and issubclass_safe(annotation):
        return [str(m.value) for m in annotation]
    return None


def issubclass_safe(annotation) -> bool:
    try:
        import enum
        return issubclass(annotation, enum.Enum)
    except Exception:
        return False


def field_constraints(field_info) -> list[str]:
    """Bounds a schema reader can see: ge, le, gt, lt, patterns, lengths."""
    out = []
    for meta in getattr(field_info, "metadata", None) or []:
        for attr in ("ge", "le", "gt", "lt", "min_length", "max_length",
                     "multiple_of", "pattern"):
            if hasattr(meta, attr):
                out.append(f"{attr}={getattr(meta, attr)}")
    return out


def collect() -> dict:
    t0 = time.perf_counter()

    model = None
    source = None
    import_error = None
    for module_path, cls_name in CANDIDATE_PATHS:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            model = getattr(module, cls_name)
            source = f"{module_path}.{cls_name}"
            break
        except Exception as e:  # noqa: BLE001 - any failure is the same finding
            import_error = f"{module_path}.{cls_name}: {type(e).__name__}: {e}"

    if model is None:
        return {"error": f"no config model importable. Last: {import_error}",
                "tried": [f"{m}.{c}" for m, c in CANDIDATE_PATHS]}

    try:
        import axolotl
        version = getattr(axolotl, "__version__", "unknown")
    except Exception:
        version = "unknown"

    fields = []
    for name, info in model.model_fields.items():
        ann = info.annotation
        required = info.is_required()
        default = None if required else info.default
        try:
            json.dumps(default)
        except (TypeError, ValueError):
            default = repr(default)
        fields.append({
            "name": name,
            "type": type_name(ann),
            "required": required,
            "has_default": not required,
            "default": default,
            "free_form": is_free_form(ann),
            "enum_values": enum_values(ann),
            "constraints": field_constraints(info),
            "infrastructure": is_infrastructure(name),
            "description": (info.description or "")[:300],
        })

    # Cross-field rules. These are the ones that decide whether generation is
    # safe: a model_validator is arbitrary Python, so a form generated from the
    # schema cannot know what it will refuse until the job is already running.
    validators = getattr(model, "__pydantic_decorators__", None)
    model_validators = list(getattr(validators, "model_validators", {}) or {})
    field_validators = list(getattr(validators, "field_validators", {}) or {})

    with_constraints = sum(1 for f in fields if f["constraints"])
    enum_typed = sum(1 for f in fields if f["enum_values"])
    # A crude but stated ratio: schema-expressed constraints against
    # schema-expressed plus runtime-only ones. Crude because one
    # model_validator can encode ten rules -- so this UNDERSTATES the runtime
    # half, and is recorded as a floor rather than an estimate.
    expressed = with_constraints + enum_typed
    total_constraints = expressed + len(model_validators) + len(field_validators)

    infra = sum(1 for f in fields if f["infrastructure"])
    training = len(fields) - infra

    correctness = {}
    names = {f["name"] for f in fields}
    for label, candidates in CORRECTNESS_SETTINGS.items():
        correctness[label] = [c for c in candidates if c in names]

    return {
        "axolotl_version": version,
        "config_model": source,
        "python": sys.version.split()[0],
        "introspection_seconds": round(time.perf_counter() - t0, 2),
        "counts": {
            "total": len(fields),
            "infrastructure": infra,
            "training_relevant": training,
            "required": sum(1 for f in fields if f["required"]),
            "with_default": sum(1 for f in fields if f["has_default"]),
            "free_form": sum(1 for f in fields if f["free_form"]),
        },
        "constraints": {
            "with_field_constraints": with_constraints,
            "enum_typed": enum_typed,
            "model_validators": len(model_validators),
            "field_validators": len(field_validators),
            "model_validator_names": model_validators,
            "expressed_in_schema_fraction": (
                round(expressed / total_constraints, 3) if total_constraints else 0.0),
            "note": "expressed_in_schema_fraction counts each model_validator "
                    "as ONE runtime constraint. One validator commonly encodes "
                    "several rules, so this OVERSTATES how much the schema "
                    "expresses. It is a ceiling, not an estimate.",
        },
        "correctness_settings": correctness,
        "fields": fields,
    }


if __name__ == "__main__":
    try:
        result = collect()
    except Exception as e:  # noqa: BLE001
        import traceback
        result = {"error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2000:]}
    print(MARKER)
    print(json.dumps(result))
