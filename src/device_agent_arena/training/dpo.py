from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import (
    cached_model_snapshot,
    uses_functiongemma_protocol,
)
from device_agent_arena.artifacts.run_io import git_provenance, source_tree_sha256
from device_agent_arena.training.preference import PREFERENCE_MANIFEST_VERSION
from device_agent_arena.training.qlora import (
    LoraSpec,
    QuantizationSpec,
    _project_relative_path,
    _resolve_project_path,
)


@dataclass(frozen=True)
class DpoTrainingConfig:
    format_version: str
    model_id: str
    model_revision: str
    train_data: str
    train_manifest: str
    output_dir: str
    initial_adapter: str
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
    beta: float
    loss_type: str
    precompute_ref_log_probs: bool
    activation_offloading: bool
    quantization: QuantizationSpec
    lora: LoraSpec
    max_steps: int = -1
    torch_empty_cache_steps: int = 0

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def load_dpo_training_config(path: str | Path) -> DpoTrainingConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["quantization"] = QuantizationSpec(**raw["quantization"])
    raw["lora"] = LoraSpec(**raw["lora"])
    config = DpoTrainingConfig(**raw)
    if config.format_version != "0.1":
        raise ValueError("unsupported DPO training config format_version")
    if config.loss_type != "sigmoid":
        raise ValueError("the frozen FCTR-DPO experiment requires sigmoid loss")
    if config.beta <= 0 or config.learning_rate <= 0:
        raise ValueError("DPO beta and learning rate must be positive")
    if config.max_length < 128 or config.max_steps == 0:
        raise ValueError("invalid DPO sequence or update budget")
    if config.torch_empty_cache_steps < 0:
        raise ValueError("torch_empty_cache_steps must be non-negative")
    if not config.precompute_ref_log_probs:
        raise ValueError("the 8 GB DPO recipe requires precomputed reference log-probabilities")
    return config


def build_dpo_config_kwargs(config: DpoTrainingConfig, *, use_bf16: bool) -> dict[str, Any]:
    kwargs = {
        "output_dir": config.output_dir,
        "seed": config.seed,
        "max_length": config.max_length,
        "max_steps": config.max_steps,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        # Transformers 5.14 accepts a float ratio through warmup_steps; warmup_ratio is deprecated.
        "warmup_steps": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "report_to": "none",
        "beta": config.beta,
        "loss_type": [config.loss_type],
        "precompute_ref_log_probs": config.precompute_ref_log_probs,
        "precompute_ref_batch_size": 1,
        "activation_offloading": config.activation_offloading,
    }
    if config.torch_empty_cache_steps:
        kwargs["torch_empty_cache_steps"] = config.torch_empty_cache_steps
    return kwargs


def dry_run_dpo_training(
    config: DpoTrainingConfig,
    *,
    project_root: str | Path,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve_project_path(root, config.train_data)
    manifest_path = _resolve_project_path(root, config.train_manifest)
    adapter_path = _resolve_project_path(root, config.initial_adapter)
    output_path = _resolve_project_path(root, config.output_dir)
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("DPO preference data or manifest is missing")
    if not (adapter_path / "adapter_config.json").is_file() or not (
        adapter_path / "adapter_model.safetensors"
    ).is_file():
        raise FileNotFoundError("DPO initial adapter is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != PREFERENCE_MANIFEST_VERSION:
        raise ValueError("DPO requires a verified FCTR preference manifest")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("DPO preference dataset SHA-256 does not match manifest")
    expected_protocol = (
        "functiongemma" if uses_functiongemma_protocol(config.model_id) else "openai"
    )
    protocol = manifest.get("protocol", "openai")
    if protocol != expected_protocol:
        raise ValueError("DPO preference protocol does not match the target model")
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    if manifest.get("pair_count") != len(records):
        raise ValueError("DPO preference pair count does not match manifest")
    for row in records:
        if not all(key in row for key in ("prompt", "chosen", "rejected", "tools")):
            raise ValueError("DPO record is missing a preference field")
        if row["chosen"] == row["rejected"]:
            raise ValueError("DPO chosen and rejected actions must differ")
    adapter_config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    resolved_adapter = {
        "rank": adapter_config.get("r"),
        "alpha": adapter_config.get("lora_alpha"),
        "dropout": adapter_config.get("lora_dropout"),
        "bias": adapter_config.get("bias"),
    }
    declared_adapter = {
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
        "dropout": config.lora.dropout,
        "bias": config.lora.bias,
    }
    if resolved_adapter != declared_adapter:
        raise ValueError("DPO LoRA config does not match the initial adapter")
    result = {
        "status": "ready",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset": {
            "path": _project_relative_path(root, data_path),
            "manifest": _project_relative_path(root, manifest_path),
            "pair_count": len(records),
            "sha256": digest,
            "protocol": protocol,
        },
        "initial_adapter": _project_relative_path(root, adapter_path),
        "initial_adapter_sha256": hashlib.sha256(
            (adapter_path / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "initial_adapter_config": resolved_adapter,
        "model_loading": (
            "qlora_4bit" if config.quantization.load_in_4bit else "bf16_lora"
        ),
        "output_dir": _project_relative_path(root, output_path),
        "effective_batch_size": config.effective_batch_size,
        "max_length": config.max_length,
        "training_objective": "state-conditioned tool-action DPO",
        "dpo": {
            "beta": config.beta,
            "loss_type": config.loss_type,
            "precompute_ref_log_probs": config.precompute_ref_log_probs,
            "activation_offloading": config.activation_offloading,
            "reference_policy": "frozen copy of initial PEFT adapter",
        },
        "quantization": asdict(config.quantization),
        "lora": asdict(config.lora),
        "training_recipe": {
            "seed": config.seed,
            "epochs": config.num_train_epochs,
            "max_steps": config.max_steps,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
        },
        "source": {**git_provenance(root), "tree_sha256": source_tree_sha256()},
        "note": "Dry-run validates inputs only; it is not training evidence.",
    }
    if artifact_path is not None:
        destination = Path(artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def train_dpo(config: DpoTrainingConfig, *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_dpo_training(config, project_root=root)
    output_path = _resolve_project_path(root, config.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty DPO output: {output_path}")
    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("DPO dependencies are missing; run `uv sync --extra train`") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen FCTR-DPO recipe requires CUDA")
    if config.quantization.compute_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("DPO bfloat16 requires CUDA bfloat16 support")
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    snapshot = cached_model_snapshot(
        model_id=config.model_id,
        model_revision=config.model_revision,
        cache_root=root / ".hf-cache",
    )
    source = str(snapshot) if snapshot is not None else config.model_id
    model_kwargs: dict[str, Any] = {
        "revision": None if snapshot is not None else config.model_revision,
        "device_map": "auto",
        "local_files_only": snapshot is not None,
    }
    if config.quantization.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype
    base_model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    if config.quantization.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    adapter_path = _resolve_project_path(root, config.initial_adapter)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if snapshot is not None else config.model_revision,
        local_files_only=snapshot is not None,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "json",
        data_files=str(_resolve_project_path(root, config.train_data)),
        split="train",
        cache_dir=str(root / ".hf-cache" / "datasets"),
    )
    kwargs = build_dpo_config_kwargs(config, use_bf16=compute_dtype == torch.bfloat16)
    kwargs["output_dir"] = str(output_path)
    # TRL supports conversational tool-calling records and precomputed reference log-probs:
    # https://huggingface.co/docs/trl/main/en/dpo_trainer#tool-calling-with-dpo
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**kwargs),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    torch.cuda.synchronize()
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    trainer.save_model(str(output_path))
    trainer_state_path = output_path / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state_path))
    adapter_model = output_path / "adapter_model.safetensors"
    result = {
        **ready,
        "status": "completed",
        "train_metrics": train_result.metrics,
        "peak_vram_mb": peak_vram_mb,
        "cuda_device": torch.cuda.get_device_name(0),
        "adapter": {
            "path": _project_relative_path(root, adapter_model),
            "size_bytes": adapter_model.stat().st_size,
            "sha256": hashlib.sha256(adapter_model.read_bytes()).hexdigest(),
        },
        "trainer_state": _project_relative_path(root, trainer_state_path),
        "note": "DPO completed; downstream environment evaluation is still required.",
    }
    (output_path / "device_agent_arena_dpo_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result
