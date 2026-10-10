from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import cached_model_snapshot
from device_agent_arena.artifacts.run_io import git_provenance, source_tree_sha256


@dataclass(frozen=True)
class QuantizationSpec:
    load_in_4bit: bool
    quant_type: str
    double_quant: bool
    compute_dtype: str


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: int
    dropout: float
    target_modules: str
    bias: str


@dataclass(frozen=True)
class TrainingConfig:
    format_version: str
    model_id: str
    train_data: str
    train_manifest: str
    output_dir: str
    seed: int
    max_length: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    gradient_checkpointing: bool
    quantization: QuantizationSpec
    lora: LoraSpec | None = None
    model_revision: str | None = None
    initial_adapter: str | None = None
    training_mode: str = "lora"
    max_steps: int = -1
    lr_scheduler_type: str = "linear"
    attn_implementation: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def load_training_config(path: str | Path) -> TrainingConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["quantization"] = QuantizationSpec(**raw["quantization"])
    raw["lora"] = LoraSpec(**raw["lora"]) if raw.get("lora") is not None else None
    config = TrainingConfig(**raw)
    _validate_config(config)
    return config


def _validate_config(config: TrainingConfig) -> None:
    if config.format_version not in {"0.1", "0.2", "0.3"}:
        raise ValueError("unsupported training config format_version")
    if config.training_mode not in {"lora", "full"}:
        raise ValueError("training_mode must be lora or full")
    if config.training_mode == "lora" and config.lora is None:
        raise ValueError("lora config is required for training_mode=lora")
    if config.training_mode == "full" and config.initial_adapter is not None:
        raise ValueError("initial_adapter is not supported for full-parameter SFT")
    if config.training_mode == "full" and config.quantization.load_in_4bit:
        raise ValueError("full-parameter SFT cannot train quantized weights")
    if not config.model_id.strip():
        raise ValueError("model_id is required")
    if config.max_length < 128:
        raise ValueError("max_length must be >= 128")
    if config.per_device_train_batch_size < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("batch sizes must be positive")
    if config.learning_rate <= 0 or config.num_train_epochs <= 0:
        raise ValueError("learning_rate and num_train_epochs must be positive")
    if config.lora is not None:
        if config.lora.rank < 1 or config.lora.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= config.lora.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
    if config.quantization.load_in_4bit and config.quantization.quant_type != "nf4":
        raise ValueError("the checked-in QLoRA recipe requires NF4")
    if config.quantization.compute_dtype not in {"bfloat16", "float16"}:
        raise ValueError("quantization.compute_dtype must be bfloat16 or float16")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if config.lr_scheduler_type not in {"linear", "cosine"}:
        raise ValueError("lr_scheduler_type must be linear or cosine")


def build_sft_config_kwargs(config: TrainingConfig, *, use_bf16: bool) -> dict[str, Any]:
    return {
        "output_dir": config.output_dir,
        "seed": config.seed,
        "max_length": config.max_length,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        # In the locked Transformers API, a float < 1 in warmup_steps is a fraction.
        # TrainingArguments.get_warmup_steps computes ceil(total_steps * value).
        "warmup_steps": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "report_to": "none",
        "completion_only_loss": True,
        "model_init_kwargs": (
            {"revision": config.model_revision} if config.model_revision is not None else None
        ),
    }


def build_trainer_quantization_config(
    spec: QuantizationSpec,
    *,
    compute_dtype: Any,
    bitsandbytes_config_cls: Any,
) -> Any | None:
    """Construct a BnB config only for an explicitly quantized training run."""
    if not spec.load_in_4bit:
        return None
    return bitsandbytes_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type=spec.quant_type,
        bnb_4bit_use_double_quant=spec.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_model_init_kwargs(
    config: TrainingConfig,
    *,
    local_snapshot: bool,
) -> dict[str, Any]:
    """Build explicit, mutually consistent model-loading kwargs for SFTTrainer."""
    kwargs: dict[str, Any] = {}
    if local_snapshot:
        kwargs["local_files_only"] = True
    elif config.model_revision is not None:
        kwargs["revision"] = config.model_revision
    if not config.quantization.load_in_4bit:
        kwargs["dtype"] = config.quantization.compute_dtype
    if config.attn_implementation is not None:
        kwargs["attn_implementation"] = config.attn_implementation
    return kwargs


def training_completion_note(
    spec: QuantizationSpec, *, training_mode: str = "lora"
) -> str:
    dtype_label = "BF16" if spec.compute_dtype == "bfloat16" else "FP16"
    if training_mode == "full":
        return f"{dtype_label} full-parameter SFT completed and model artifacts were saved."
    recipe = "QLoRA" if spec.load_in_4bit else f"{dtype_label} LoRA"
    return f"{recipe} training completed and adapter artifacts were saved."


def _resolve_project_path(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"configured path escapes project root: {relative}")
    return resolved


def _project_relative_path(project_root: Path, path: Path) -> str:
    """Return a portable path for provenance records rooted at the project."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def dry_run_training(
    config: TrainingConfig, *, project_root: str | Path, artifact_path: str | Path | None = None
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve_project_path(root, config.train_data)
    manifest_path = _resolve_project_path(root, config.train_manifest)
    output_path = _resolve_project_path(root, config.output_dir)
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("training data or manifest is missing; run export-sft first")

    payload = data_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_count = sum(1 for line in payload.splitlines() if line)
    if manifest.get("sha256") != digest:
        raise ValueError("training dataset SHA-256 does not match manifest")
    if manifest.get("record_count") != record_count:
        raise ValueError("training dataset record count does not match manifest")

    adapter_path = None
    adapter_provenance = None
    if config.initial_adapter is not None:
        adapter_path = _resolve_project_path(root, config.initial_adapter)
        adapter_config_path = adapter_path / "adapter_config.json"
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        if not adapter_config_path.is_file() or not adapter_model_path.is_file():
            raise FileNotFoundError("initial_adapter is missing its config or model weights")
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        assert config.lora is not None
        expected_adapter = {
            "r": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "bias": config.lora.bias,
        }
        mismatches = {
            key: {"expected": expected, "actual": adapter_config.get(key)}
            for key, expected in expected_adapter.items()
            if adapter_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"initial_adapter LoRA config mismatch: {mismatches}")
        adapter_provenance = {
            "config_sha256": hashlib.sha256(adapter_config_path.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(adapter_model_path.read_bytes()).hexdigest(),
            "model_size_bytes": adapter_model_path.stat().st_size,
        }

    result = {
        "status": "ready",
        "path_format": "project-relative-posix",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "training_mode": config.training_mode,
        "initial_adapter": (
            _project_relative_path(root, adapter_path) if adapter_path is not None else None
        ),
        "initial_adapter_provenance": adapter_provenance,
        "dataset": {
            "path": _project_relative_path(root, data_path),
            "manifest": _project_relative_path(root, manifest_path),
            "record_count": record_count,
            "sha256": digest,
        },
        "output_dir": _project_relative_path(root, output_path),
        "effective_batch_size": config.effective_batch_size,
        "max_length": config.max_length,
        "training_recipe": {
            "seed": config.seed,
            "num_train_epochs": config.num_train_epochs,
            "max_steps": config.max_steps,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lr_scheduler_type": config.lr_scheduler_type,
            "warmup_ratio": config.warmup_ratio,
            "logging_steps": config.logging_steps,
            "save_strategy": config.save_strategy,
            "gradient_checkpointing": config.gradient_checkpointing,
            "attn_implementation": config.attn_implementation,
        },
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dependency_lock_sha256": (
            hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
            if (root / "uv.lock").is_file()
            else None
        ),
        "training_objective": (
            "completion-only long-horizon next-agent decision"
            if str(manifest.get("generator", "")).startswith("rl-sft-")
            else "completion-only next-agent decision"
        ),
        "lora": asdict(config.lora) if config.lora is not None else None,
        "quantization": asdict(config.quantization),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source": {
            **git_provenance(root),
            "tree_sha256": source_tree_sha256(),
        },
        "note": "Dry-run validates inputs only; it is not evidence that training completed.",
    }
    if artifact_path is not None:
        destination = Path(artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def train_qlora(config: TrainingConfig, *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_training(config, project_root=root)
    output_path = _resolve_project_path(root, config.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty training output: {output_path}")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; run `uv sync --extra train`"
        ) from exc

    requested_compute_dtype = config.quantization.compute_dtype
    if requested_compute_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "quantization.compute_dtype=bfloat16 requires a CUDA device with bfloat16 support"
            )
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    quantization_config = build_trainer_quantization_config(
        config.quantization,
        compute_dtype=compute_dtype,
        bitsandbytes_config_cls=BitsAndBytesConfig,
    )
    peft_config = None
    if config.training_mode == "lora":
        assert config.lora is not None
        # QLoRA-style all-linear targeting follows the current PEFT guidance.
        # Source: https://huggingface.co/docs/peft/developer_guides/lora#qlora-style-training
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
        )
    data_path = _resolve_project_path(root, config.train_data)
    dataset_cache = root / ".hf-cache" / "datasets"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
        cache_dir=str(dataset_cache),
    )
    output_dir = str(output_path)
    sft_kwargs = build_sft_config_kwargs(
        config,
        use_bf16=compute_dtype == torch.bfloat16,
    )
    sft_kwargs["output_dir"] = output_dir
    local_snapshot = cached_model_snapshot(
        model_id=config.model_id,
        model_revision=config.model_revision,
        cache_root=root / ".hf-cache",
    )
    model_source = str(local_snapshot) if local_snapshot is not None else config.model_id
    sft_kwargs["model_init_kwargs"] = build_model_init_kwargs(
        config,
        local_snapshot=local_snapshot is not None,
    )
    model: str | Any = model_source
    trainer_peft_config: LoraConfig | None = peft_config
    trainer_quantization_config: BitsAndBytesConfig | None = quantization_config
    processing_class = None
    if config.initial_adapter is not None:
        model_load_kwargs = {
            **build_model_init_kwargs(
                config,
                local_snapshot=local_snapshot is not None,
            ),
            "device_map": "auto",
        }
        if quantization_config is not None:
            model_load_kwargs["quantization_config"] = quantization_config
        base_model = AutoModelForCausalLM.from_pretrained(model_source, **model_load_kwargs)
        if quantization_config is not None:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )
        adapter_path = _resolve_project_path(root, config.initial_adapter)
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        processing_class = AutoTokenizer.from_pretrained(
            model_source,
            revision=None if local_snapshot is not None else config.model_revision,
            local_files_only=local_snapshot is not None,
        )
        trainer_peft_config = None
        trainer_quantization_config = None
        sft_kwargs["model_init_kwargs"] = None
    training_args = SFTConfig(**sft_kwargs)
    # SFTTrainer natively supports conversational tool-calling records with
    # `messages`, `tool_calls`, tool responses, and a `tools` column.
    # Source: https://huggingface.co/docs/trl/en/sft_trainer#tool-calling-with-sft
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": processing_class,
        "peft_config": trainer_peft_config,
    }
    if trainer_quantization_config is not None:
        trainer_kwargs["quantization_config"] = trainer_quantization_config
    trainer = SFTTrainer(**trainer_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 3) if torch.cuda.is_available() else 0.0
    )
    trainer.save_model(output_dir)
    trainer_state_path = Path(output_dir) / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state_path))
    weights_name = (
        "adapter_model.safetensors"
        if config.training_mode == "lora"
        else "model.safetensors"
    )
    weights_path = Path(output_dir) / weights_name
    base_model = (
        trainer.model.get_base_model()
        if callable(getattr(trainer.model, "get_base_model", None))
        else trainer.model
    )
    resolved_model_revision = getattr(base_model.config, "_commit_hash", None)
    package_versions = {}
    for package in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    result = {
        **ready,
        "status": "completed",
        "train_metrics": train_result.metrics,
        "peak_vram_mb": peak_vram_mb,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "resolved_compute_dtype": requested_compute_dtype,
        "resolved_model_revision": resolved_model_revision,
        "weights": {
            "kind": "adapter" if config.training_mode == "lora" else "full-model",
            "path": _project_relative_path(root, weights_path),
            "size_bytes": weights_path.stat().st_size,
            "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        },
        "trainer_state": _project_relative_path(root, trainer_state_path),
        "package_versions": package_versions,
        "note": training_completion_note(
            config.quantization, training_mode=config.training_mode
        ),
    }
    if config.training_mode == "lora":
        result["adapter"] = {
            "path": result["weights"]["path"],
            "size_bytes": result["weights"]["size_bytes"],
            "sha256": result["weights"]["sha256"],
        }
    destination = Path(output_dir) / "device_agent_arena_training_result.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result
