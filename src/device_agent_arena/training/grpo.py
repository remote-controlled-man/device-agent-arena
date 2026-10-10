from __future__ import annotations

import hashlib
import json
import platform
import random
import statistics
import sys
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from device_agent_arena.artifacts.run_io import git_provenance, source_tree_sha256
from device_agent_arena.dataset.rl_suite import (
    load_rl_task_selection,
    validate_rl_task_suite,
)
from device_agent_arena.domain.models import TaskSpec
from device_agent_arena.domain.task_io import load_tasks
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT
from device_agent_arena.rl.rewards import REWARD_MODEL_VERSION, RewardModel
from device_agent_arena.rl.trl_environment import DeviceGRPOEnvironment
from device_agent_arena.training.qlora import LoraSpec, QuantizationSpec

GRPO_CONFIG_FORMAT_VERSION = "rl-train-v0.5"
TERMINAL_GRPO_CONFIG_FORMAT_VERSION = "rl-train-v0.4"
SELECTION_GRPO_CONFIG_FORMAT_VERSION = "rl-train-v0.3"
PROTOCOL_GRPO_CONFIG_FORMAT_VERSION = "rl-train-v0.2"
LEGACY_GRPO_CONFIG_FORMAT_VERSION = "rl-train-v0.1"
LOCAL_GPU_VRAM_MB = 8188
LOCAL_GPU_HARD_GUARD_MB = 7700


@dataclass(frozen=True)
class GRPOTrainingConfig:
    format_version: str
    stage: str
    model_id: str
    model_revision: str
    initial_adapter: str | None
    train_data: str
    train_manifest: str
    train_selection: str | None
    output_dir: str
    seed: int
    train_task_limit: int
    max_updates: int
    environment_max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    gradient_checkpointing: bool
    num_generations: int
    max_completion_length: int
    max_tool_calling_iterations: int
    beta: float
    loss_type: str
    mask_truncated_completions: bool
    use_vllm: bool
    temperature: float
    top_p: float
    max_peak_vram_mb: int
    quantization: QuantizationSpec
    lora: LoraSpec
    reward_model_version: str | None = None
    truncation_mask_mode: str = "token_eos"
    max_group_attempts: int = 1
    mixed_success_first: bool = False
    minimum_shaping_reward_std: float = 0.0


def load_grpo_training_config(path: str | Path) -> GRPOTrainingConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.setdefault("train_selection", None)
    raw.setdefault("truncation_mask_mode", "token_eos")
    raw.setdefault("max_group_attempts", 1)
    raw.setdefault("mixed_success_first", False)
    raw.setdefault("minimum_shaping_reward_std", 0.0)
    raw["quantization"] = QuantizationSpec(**raw["quantization"])
    raw["lora"] = LoraSpec(**raw["lora"])
    config = GRPOTrainingConfig(**raw)
    _validate_config(config)
    return config


def grpo_config_sha256(config: GRPOTrainingConfig) -> str:
    payload = asdict(config)
    if config.format_version not in {
        SELECTION_GRPO_CONFIG_FORMAT_VERSION,
        TERMINAL_GRPO_CONFIG_FORMAT_VERSION,
        GRPO_CONFIG_FORMAT_VERSION,
    }:
        payload.pop("train_selection", None)
    if config.format_version not in {
        TERMINAL_GRPO_CONFIG_FORMAT_VERSION,
        GRPO_CONFIG_FORMAT_VERSION,
    }:
        payload.pop("truncation_mask_mode", None)
    if config.format_version != GRPO_CONFIG_FORMAT_VERSION:
        payload.pop("max_group_attempts", None)
        payload.pop("mixed_success_first", None)
        payload.pop("minimum_shaping_reward_std", None)
    if config.format_version == LEGACY_GRPO_CONFIG_FORMAT_VERSION:
        payload.pop("reward_model_version", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_config(config: GRPOTrainingConfig) -> None:
    if config.format_version not in {
        LEGACY_GRPO_CONFIG_FORMAT_VERSION,
        PROTOCOL_GRPO_CONFIG_FORMAT_VERSION,
        SELECTION_GRPO_CONFIG_FORMAT_VERSION,
        TERMINAL_GRPO_CONFIG_FORMAT_VERSION,
        GRPO_CONFIG_FORMAT_VERSION,
    }:
        raise ValueError("unsupported GRPO training config format_version")
    if (
        config.format_version
        in {
            PROTOCOL_GRPO_CONFIG_FORMAT_VERSION,
            SELECTION_GRPO_CONFIG_FORMAT_VERSION,
            TERMINAL_GRPO_CONFIG_FORMAT_VERSION,
            GRPO_CONFIG_FORMAT_VERSION,
        }
        and config.reward_model_version != REWARD_MODEL_VERSION
    ):
        raise ValueError(
            f"{GRPO_CONFIG_FORMAT_VERSION} requires reward_model_version "
            f"{REWARD_MODEL_VERSION}"
        )
    if (
        config.format_version
        in {
            SELECTION_GRPO_CONFIG_FORMAT_VERSION,
            TERMINAL_GRPO_CONFIG_FORMAT_VERSION,
            GRPO_CONFIG_FORMAT_VERSION,
        }
        and not config.train_selection
    ):
        raise ValueError("GRPO config v0.3+ requires train_selection")
    if config.truncation_mask_mode not in {"token_eos", "environment_terminal"}:
        raise ValueError("truncation_mask_mode must be token_eos or environment_terminal")
    if (
        config.format_version
        not in {TERMINAL_GRPO_CONFIG_FORMAT_VERSION, GRPO_CONFIG_FORMAT_VERSION}
        and config.truncation_mask_mode != "token_eos"
    ):
        raise ValueError("environment_terminal masking requires GRPO config v0.4")
    if (
        config.truncation_mask_mode == "environment_terminal"
        and config.mask_truncated_completions
    ):
        raise ValueError(
            "environment_terminal masking requires mask_truncated_completions=false "
            "so TRL does not erase valid protocol-terminal rows first"
        )
    if config.max_group_attempts < 1 or config.max_group_attempts > 3:
        raise ValueError("max_group_attempts must be between 1 and 3")
    if config.format_version != GRPO_CONFIG_FORMAT_VERSION and (
        config.max_group_attempts != 1
        or config.mixed_success_first
        or config.minimum_shaping_reward_std != 0.0
    ):
        raise ValueError("group resampling requires GRPO config v0.5")
    if config.max_group_attempts > 1 and not config.mixed_success_first:
        raise ValueError("bounded group resampling requires mixed_success_first=true")
    if not 0 <= config.minimum_shaping_reward_std <= 1:
        raise ValueError("minimum_shaping_reward_std must be in [0, 1]")
    if not config.stage.strip() or not config.model_id.strip() or not config.model_revision.strip():
        raise ValueError("stage, model_id, and model_revision are required")
    if config.train_task_limit < 1 or config.max_updates < 1:
        raise ValueError("train_task_limit and max_updates must be positive")
    if not 7 <= config.environment_max_steps <= 12:
        raise ValueError("environment_max_steps must be between 7 and 12")
    if config.per_device_train_batch_size != 1:
        raise ValueError("the checked-in 8GB GRPO recipe requires micro batch size 1")
    if config.num_generations < 2:
        raise ValueError("GRPO requires at least two generations")
    generation_batch_size = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    if generation_batch_size % config.num_generations:
        raise ValueError("gradient accumulation must preserve a complete generation group")
    if config.learning_rate <= 0 or not 0 <= config.warmup_ratio < 1:
        raise ValueError("learning_rate must be positive and warmup_ratio must be in [0, 1)")
    if config.max_completion_length < 128:
        raise ValueError("max_completion_length must be at least 128")
    if config.max_tool_calling_iterations != config.environment_max_steps:
        raise ValueError("tool-calling and environment step limits must match")
    if config.beta < 0:
        raise ValueError("beta must be non-negative")
    if config.loss_type not in {"dapo", "dr_grpo"}:
        raise ValueError("loss_type must be dapo or dr_grpo")
    if config.use_vllm:
        raise ValueError("the native Windows 8GB recipe starts with use_vllm=false")
    if not 0 < config.temperature <= 2 or not 0 < config.top_p <= 1:
        raise ValueError("temperature and top_p are outside the supported range")
    if config.lora.rank < 1 or config.lora.alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if config.quantization.quant_type != "nf4" or not config.quantization.load_in_4bit:
        raise ValueError("the checked-in 8GB recipe requires 4-bit NF4")
    if config.quantization.compute_dtype not in {"bfloat16", "float16"}:
        raise ValueError("quantization compute_dtype must be bfloat16 or float16")


def build_grpo_config_kwargs(
    config: GRPOTrainingConfig,
    *,
    use_bf16: bool,
) -> dict[str, Any]:
    return {
        "output_dir": config.output_dir,
        "seed": config.seed,
        "max_steps": config.max_updates,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        # Transformers 5.x accepts a fractional warmup_steps value and deprecates warmup_ratio.
        "warmup_steps": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "report_to": "none",
        "remove_unused_columns": False,
        "num_generations": config.num_generations,
        "max_completion_length": config.max_completion_length,
        "max_tool_calling_iterations": config.max_tool_calling_iterations,
        "beta": config.beta,
        "loss_type": config.loss_type,
        "mask_truncated_completions": config.mask_truncated_completions,
        "use_vllm": config.use_vllm,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "chat_template_kwargs": {"enable_thinking": False},
        "log_completions": True,
        "optim": "adamw_torch",
    }


def build_grpo_dataset_rows(tasks: list[TaskSpec]) -> list[dict[str, Any]]:
    """Build raw conversational prompts that match benchmark inference."""
    return [
        {
            "task_id": task.id,
            "prompt": [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": ""},
            ],
        }
        for task in tasks
    ]


def enforce_vram_guard(*, allocated_mb: float, device_used_mb: float, limit_mb: int) -> None:
    """Guard live tensors and CUDA-reported global device use."""
    if allocated_mb > limit_mb:
        raise RuntimeError(
            f"peak CUDA allocated memory {allocated_mb:.1f} MiB exceeded "
            f"the {limit_mb} MiB hard guard"
        )
    if device_used_mb > limit_mb:
        raise RuntimeError(
            f"sampled global CUDA device usage {device_used_mb:.1f} MiB "
            f"(peak tensor allocation {allocated_mb:.1f} MiB) exceeded "
            f"the {limit_mb} MiB hard guard"
        )


def sample_cuda_memory_with_reclaim(cuda: Any, *, limit_mb: int) -> dict[str, Any]:
    """Release unused cache before treating global device use as unsafe."""
    mib = 1024**2
    free_bytes, total_bytes = cuda.mem_get_info()
    before_reclaim_mb = (total_bytes - free_bytes) / mib
    cache_reclaimed = before_reclaim_mb > limit_mb
    if cache_reclaimed:
        cuda.empty_cache()
        free_bytes, total_bytes = cuda.mem_get_info()
    after_reclaim_mb = (total_bytes - free_bytes) / mib
    return {
        "before_reclaim_mb": round(before_reclaim_mb, 3),
        "after_reclaim_mb": round(after_reclaim_mb, 3),
        "cache_reclaimed": cache_reclaimed,
    }


def apply_environment_terminal_loss_mask(
    *,
    completion_mask: Any,
    tool_mask: Any | None,
    protocol_terminated: list[bool],
    token_eos_terminated: list[bool],
) -> tuple[Any, Any | None, dict[str, int]]:
    """Mask only genuinely incomplete rollouts in a tool environment.

    TRL's default truncation heuristic only looks for a terminal token.  A
    stateful tool environment can finish at the protocol layer without the
    model emitting EOS after the final tool result.  Such trajectories are
    valid training examples and must not be discarded.  Conversely, a row
    that has neither a protocol terminal event nor EOS is still treated as a
    real length truncation and is fully masked.
    """
    batch_size = int(completion_mask.shape[0])
    if len(protocol_terminated) != batch_size or len(token_eos_terminated) != batch_size:
        raise ValueError("termination metadata must match the completion batch size")
    keep = [
        protocol_done or token_done
        for protocol_done, token_done in zip(
            protocol_terminated, token_eos_terminated, strict=True
        )
    ]
    keep_tensor = completion_mask.new_tensor(keep, dtype=completion_mask.dtype).unsqueeze(1)
    masked_completion = completion_mask * keep_tensor
    masked_tool = tool_mask * keep_tensor if tool_mask is not None else None
    return masked_completion, masked_tool, {
        "protocol_terminated_count": sum(protocol_terminated),
        "token_eos_terminated_count": sum(token_eos_terminated),
        "length_truncated_count": sum(not value for value in keep),
        "protocol_terminal_without_eos_count": sum(
            protocol_done and not token_done
            for protocol_done, token_done in zip(
                protocol_terminated, token_eos_terminated, strict=True
            )
        ),
    }


def protocol_terminal_prompt_mask(
    *, prompt_texts: list[str], terminal_markers: tuple[str, ...]
) -> list[bool]:
    """Detect the exact terminal tool-result marker in a post-tool prompt."""
    if not terminal_markers or any(not marker for marker in terminal_markers):
        raise ValueError("terminal tool-result markers must be non-empty")
    return [all(marker in prompt for marker in terminal_markers) for prompt in prompt_texts]


def rank_group_resample_candidates(
    *, task_ids: list[str], current_task_id: str, seed: int, update_index: int
) -> list[str]:
    """Return a deterministic without-replacement fallback order for one update."""
    if current_task_id not in task_ids:
        raise ValueError("current task must belong to the frozen resampling pool")
    candidates = [task_id for task_id in task_ids if task_id != current_task_id]
    random.Random(f"{seed}:{update_index}:{current_task_id}").shuffle(candidates)
    return candidates


def choose_group_resampling_attempt(signal_types: list[str]) -> int:
    """Choose mixed terminal signal, else first shaping signal, else final draw."""
    if not signal_types:
        raise ValueError("group resampling requires at least one attempt")
    with suppress(ValueError):
        return signal_types.index("mixed_terminal")
    with suppress(ValueError):
        return signal_types.index("shaping")
    return len(signal_types) - 1


def snapshot_trainer_metrics(metrics: dict[str, list[float]]) -> dict[str, int]:
    """Capture append positions so rejected resampling metrics can be removed."""
    return {name: len(values) for name, values in metrics.items()}


def pop_trainer_metric_delta(
    metrics: dict[str, list[float]], baseline: dict[str, int]
) -> dict[str, list[float]]:
    """Remove and return metrics appended by one candidate-group generation."""
    delta: dict[str, list[float]] = {}
    for name in list(metrics):
        start = baseline.get(name, 0)
        values = metrics[name]
        if len(values) < start:
            raise RuntimeError(f"trainer metric shrank during generation: {name}")
        appended = list(values[start:])
        if appended:
            delta[name] = appended
        del values[start:]
        if not values and name not in baseline:
            del metrics[name]
    return dict(sorted(delta.items()))


def restore_trainer_metric_delta(
    metrics: dict[str, list[float]], delta: dict[str, list[float]]
) -> None:
    """Append one accepted candidate group's isolated trainer metrics."""
    for name, values in delta.items():
        metrics.setdefault(name, []).extend(values)


def build_grpo_failure_result(
    *,
    ready: dict[str, Any],
    error: Exception,
    optimizer_updates: int,
    log_history: list[dict[str, Any]],
    peak_allocated_mb: float,
    peak_reserved_mb: float,
    peak_device_used_mb: float,
    cuda_device: str,
) -> dict[str, Any]:
    """Create a durable receipt for an interrupted or guarded GRPO run."""
    return {
        **ready,
        "status": "failed",
        "failure": {"type": type(error).__name__, "message": str(error)},
        "optimizer_updates_completed": optimizer_updates,
        "log_history": log_history,
        "peak_vram_mb": {
            "allocated": peak_allocated_mb,
            "reserved": peak_reserved_mb,
            "device_used": peak_device_used_mb,
        },
        "cuda_device": cuda_device,
        "note": "GRPO did not complete; no adapter from this run is valid for evaluation.",
    }


def dry_run_grpo_training(
    config: GRPOTrainingConfig,
    *,
    project_root: str | Path,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    if config.max_peak_vram_mb > LOCAL_GPU_HARD_GUARD_MB:
        raise ValueError("max_peak_vram_mb must not exceed the 7,700 MiB hard guard")
    root = Path(project_root).resolve()
    data_path = _resolve_project_path(root, config.train_data)
    manifest_path = _resolve_project_path(root, config.train_manifest)
    selection_path = (
        _resolve_project_path(root, config.train_selection)
        if config.train_selection is not None
        else None
    )
    output_path = _resolve_project_path(root, config.output_dir)
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("RL task suite is missing; run generate-rl-data first")
    validation = validate_rl_task_suite(manifest_path.parent)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_metadata = manifest["files"][data_path.name]
    if config.train_task_limit > file_metadata["record_count"]:
        raise ValueError("train_task_limit exceeds the frozen training split")
    selected_tasks = _load_grpo_training_tasks(
        data_path=data_path,
        selection_path=selection_path,
        train_task_limit=config.train_task_limit,
    )
    output_state = "non-empty" if output_path.exists() and any(output_path.iterdir()) else "empty"
    adapter_path = None
    adapter_provenance = None
    if config.initial_adapter is not None:
        adapter_path = _resolve_project_path(root, config.initial_adapter)
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("initial_adapter is missing adapter_model.safetensors")
        adapter_config_path = adapter_path / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
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
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        adapter_provenance = {
            "config_sha256": hashlib.sha256(adapter_config_path.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(adapter_model_path.read_bytes()).hexdigest(),
            "model_size_bytes": adapter_model_path.stat().st_size,
        }

    result = {
        "status": "ready",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "initial_adapter": (
            _project_relative_path(root, adapter_path) if adapter_path is not None else None
        ),
        "initial_adapter_provenance": adapter_provenance,
        "dataset": {
            "path": _project_relative_path(root, data_path),
            "manifest": _project_relative_path(root, manifest_path),
            "record_count": file_metadata["record_count"],
            "selected_task_count": config.train_task_limit,
            "sha256": file_metadata["sha256"],
            "suite_validation": validation,
            "selection": (
                {
                    "path": _project_relative_path(root, selection_path),
                    "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    "task_ids": [task.id for task in selected_tasks],
                    "task_ids_sha256": hashlib.sha256(
                        ("\n".join(task.id for task in selected_tasks) + "\n").encode()
                    ).hexdigest(),
                }
                if selection_path is not None
                else None
            ),
        },
        "output_dir": _project_relative_path(root, output_path),
        "output_state": output_state,
        "objective": "online multi-turn environment reward (trajectory GRPO)",
        "reward_model": {
            "version": config.reward_model_version or REWARD_MODEL_VERSION,
            "weights": asdict(
                RewardModel(version=config.reward_model_version or REWARD_MODEL_VERSION).weights
            ),
        },
        "prompt": {
            "messages": ["system", "user"],
            "system_prompt_sha256": hashlib.sha256(
                BASE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "generation_group_size": config.num_generations,
        "generation_batch_size": (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        ),
        "environment_max_steps": config.environment_max_steps,
        "truncation_mask": {
            "mode": config.truncation_mask_mode,
            "trl_mask_truncated_completions": config.mask_truncated_completions,
            "semantics": (
                "keep protocol-terminal or EOS-terminal rollouts; mask only rows with neither"
                if config.truncation_mask_mode == "environment_terminal"
                else "TRL EOS/PAD terminal-token heuristic"
            ),
        },
        "hardware_guard": {
            "target_gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "available_vram_mb": LOCAL_GPU_VRAM_MB,
            "max_peak_vram_mb": config.max_peak_vram_mb,
            "guard_metrics": [
                "peak_tensor_allocated",
                "sampled_global_device_used",
            ],
            "reserved_memory_role": "diagnostic_only",
            "cache_reclaim": "empty unused cache before enforcing sampled device usage",
        },
        "quantization": asdict(config.quantization),
        "lora": asdict(config.lora),
        "config_sha256": grpo_config_sha256(config),
        "dependency_lock_sha256": (
            hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
            if (root / "uv.lock").is_file()
            else None
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source": {
            **git_provenance(root),
            "tree_sha256": source_tree_sha256(),
        },
        "note": "Dry-run validates contracts only; it is not evidence that GRPO completed.",
    }
    if artifact_path is not None:
        destination = Path(artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def train_grpo(config: GRPOTrainingConfig, *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_grpo_training(config, project_root=root)
    output_path = _resolve_project_path(root, config.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty GRPO output: {output_path}")
    model_snapshot = _cached_model_snapshot(root, config.model_id, config.model_revision)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.trainer_callback import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "GRPO dependencies are missing; run `uv sync --extra train`"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU")
    _, device_total_bytes = torch.cuda.mem_get_info()
    allocator_limit_fraction = min(
        1.0,
        config.max_peak_vram_mb * 1024**2 / device_total_bytes,
    )
    torch.cuda.set_per_process_memory_fraction(allocator_limit_fraction, device=0)

    requested_dtype = config.quantization.compute_dtype
    if requested_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the checked-in GRPO recipe requires CUDA bfloat16 support")
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        local_files_only=True,
    )
    model: str | Any = str(model_snapshot)
    trainer_peft_config: LoraConfig | None = peft_config
    trainer_quantization_config: BitsAndBytesConfig | None = quantization_config
    if config.initial_adapter is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_snapshot,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        adapter_path = _resolve_project_path(root, config.initial_adapter)
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        trainer_peft_config = None
        trainer_quantization_config = None

    data_path = _resolve_project_path(root, config.train_data)
    selection_path = (
        _resolve_project_path(root, config.train_selection)
        if config.train_selection is not None
        else None
    )
    tasks = _load_grpo_training_tasks(
        data_path=data_path,
        selection_path=selection_path,
        train_task_limit=config.train_task_limit,
    )
    dataset = Dataset.from_list(build_grpo_dataset_rows(tasks))
    rollout_records: list[dict[str, Any]] = []

    class VRAMGuardCallback(TrainerCallback):
        peak_device_used_before_reclaim_mb = 0.0
        peak_device_used_after_reclaim_mb = 0.0
        cache_reclaim_count = 0

        def on_substep_end(self, args, state, control, **kwargs):
            self._check()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self._check()
            return control

        @classmethod
        def _check(cls) -> None:
            sample = sample_cuda_memory_with_reclaim(
                torch.cuda, limit_mb=config.max_peak_vram_mb
            )
            cls.peak_device_used_before_reclaim_mb = max(
                cls.peak_device_used_before_reclaim_mb,
                sample["before_reclaim_mb"],
            )
            cls.peak_device_used_after_reclaim_mb = max(
                cls.peak_device_used_after_reclaim_mb,
                sample["after_reclaim_mb"],
            )
            cls.cache_reclaim_count += int(sample["cache_reclaimed"])
            enforce_vram_guard(
                allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
                device_used_mb=sample["after_reclaim_mb"],
                limit_mb=config.max_peak_vram_mb,
            )

    grpo_kwargs = build_grpo_config_kwargs(
        config,
        use_bf16=compute_dtype == torch.bfloat16,
    )
    grpo_kwargs["output_dir"] = str(output_path)
    if isinstance(model, str):
        grpo_kwargs["model_init_kwargs"] = {"local_files_only": True}
    training_args = GRPOConfig(**grpo_kwargs)
    environment_factory = lambda: DeviceGRPOEnvironment(  # noqa: E731
        tasks,
        max_steps=config.environment_max_steps,
        reward_model_version=config.reward_model_version or REWARD_MODEL_VERSION,
        rollout_sink=rollout_records.append,
    )

    class TerminalAwareGRPOTrainer(GRPOTrainer):
        """Use environment protocol termination instead of EOS-only masking."""

        _post_terminal_generation_count = 0
        _terminal_tool_result_markers = (
            '"done": true',
            '"termination": "agent_finished"',
        )

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
            completion_ids, logprobs = super()._generate_single_turn(
                prompt_ids, images, multimodal_fields
            )
            if config.truncation_mask_mode == "environment_terminal":
                stopped = protocol_terminal_prompt_mask(
                    prompt_texts=self.processing_class.batch_decode(
                        prompt_ids, skip_special_tokens=False
                    ),
                    terminal_markers=self._terminal_tool_result_markers,
                )
                for index, stop in enumerate(stopped):
                    if stop:
                        completion_ids[index] = []
                        if logprobs is not None:
                            logprobs[index] = []
                self._post_terminal_generation_count += sum(stopped)
            return completion_ids, logprobs

        def _generate_and_score_completions(self, inputs):
            if config.max_group_attempts == 1:
                return self._generate_terminal_aware(inputs)

            current_task_id = str(inputs[0]["task_id"])
            if any(str(row["task_id"]) != current_task_id for row in inputs):
                raise RuntimeError("bounded resampling requires one task per generation group")
            candidates = rank_group_resample_candidates(
                task_ids=[task.id for task in tasks],
                current_task_id=current_task_id,
                seed=config.seed,
                update_index=int(self.state.global_step) + 1,
            )
            attempt_task_ids = [current_task_id, *candidates][
                : config.max_group_attempts
            ]
            attempts = []
            for attempt_index, task_id in enumerate(attempt_task_ids, 1):
                attempt_inputs = [{**row, "task_id": task_id} for row in inputs]
                audit_start = len(rollout_records)
                mode = "train" if self.model.training else "eval"
                metric_baseline = snapshot_trainer_metrics(self._metrics[mode])
                output = self._generate_terminal_aware(attempt_inputs)
                metric_delta = pop_trainer_metric_delta(
                    self._metrics[mode], metric_baseline
                )
                records = rollout_records[audit_start:]
                if len(records) != config.num_generations:
                    raise RuntimeError("resampled group did not emit the configured rollout count")
                rewards = [float(record["reward_trace"]["total"]) for record in records]
                successes = sum(bool(record["trial"]["grade"]["success"]) for record in records)
                reward_std = statistics.pstdev(rewards)
                signal_type = (
                    "mixed_terminal"
                    if 0 < successes < config.num_generations
                    else (
                        "shaping"
                        if reward_std >= config.minimum_shaping_reward_std
                        else "zero_variance"
                    )
                )
                attempt = {
                    "output": output,
                    "records": records,
                    "task_id": task_id,
                    "attempt": attempt_index,
                    "success_count": successes,
                    "reward_std": reward_std,
                    "signal_type": signal_type,
                    "metric_delta": metric_delta,
                }
                attempts.append(attempt)
                if signal_type == "mixed_terminal":
                    break

            accepted_index = choose_group_resampling_attempt(
                [str(attempt["signal_type"]) for attempt in attempts]
            )
            accepted = attempts[accepted_index]
            restore_trainer_metric_delta(
                self._metrics[mode], accepted["metric_delta"]
            )
            for index, attempt in enumerate(attempts):
                is_accepted = index == accepted_index
                for record in attempt["records"]:
                    record["group_resampling"] = {
                        "update_index": int(self.state.global_step) + 1,
                        "attempt": attempt["attempt"],
                        "task_id": attempt["task_id"],
                        "accepted": is_accepted,
                        "signal_type": attempt["signal_type"],
                        "success_count": attempt["success_count"],
                        "reward_std": round(attempt["reward_std"], 6),
                    }

            accepted_rewards = [
                float(record["reward_trace"]["total"]) for record in accepted["records"]
            ]
            accepted_mean = statistics.fmean(accepted_rewards)
            accepted_std = statistics.pstdev(accepted_rewards)
            # TRL's core reward metrics are recomputed explicitly from the same
            # accepted records used for the returned loss. Other core metrics now
            # come only from the restored accepted attempt, not rejected proposals.
            self._metrics[mode]["reward"][-1] = accepted_mean
            self._metrics[mode]["reward_std"][-1] = accepted_std
            self._metrics[mode]["frac_reward_zero_std"][-1] = float(
                accepted_std == 0.0
            )
            self._metrics[mode]["resampling/group_attempts"].append(float(len(attempts)))
            self._metrics[mode]["resampling/accepted_mixed_terminal"].append(
                float(accepted["signal_type"] == "mixed_terminal")
            )
            self._metrics[mode]["resampling/accepted_shaping"].append(
                float(accepted["signal_type"] == "shaping")
            )
            self._metrics[mode]["resampling/accepted_zero_variance"].append(
                float(accepted["signal_type"] == "zero_variance")
            )
            self._metrics[mode]["resampling/proposal_groups"].append(float(len(attempts)))
            self._metrics[mode]["resampling/proposal_rollouts"].append(
                float(len(attempts) * config.num_generations)
            )
            return accepted["output"]

        def _generate_terminal_aware(self, inputs):
            post_terminal_before = self._post_terminal_generation_count
            output = super()._generate_and_score_completions(inputs)
            if config.truncation_mask_mode != "environment_terminal":
                return output
            if len(self.environments) != output["completion_mask"].shape[0]:
                raise RuntimeError("environment batch does not align with GRPO completions")

            completion_mask = output["completion_mask"]
            completion_ids = output["completion_ids"]
            lengths = completion_mask.sum(dim=1).tolist()
            terminal_token_ids = {
                self._tokenizer.eos_token_id,
                self._tokenizer.pad_token_id,
            }
            token_eos_terminated = [
                bool(length) and int(completion_ids[index, int(length) - 1]) in terminal_token_ids
                for index, length in enumerate(lengths)
            ]
            protocol_terminated = []
            for environment in self.environments:
                trial = environment.last_trial
                terminal = bool(
                    trial is not None
                    and trial.termination == "agent_finished"
                    and any(
                        event.kind == "message"
                        and event.payload.get("kind") in {"answer", "refuse"}
                        for event in trial.events
                    )
                )
                protocol_terminated.append(terminal)

            completion_mask, tool_mask, diagnostics = (
                apply_environment_terminal_loss_mask(
                    completion_mask=completion_mask,
                    tool_mask=output.get("tool_mask"),
                    protocol_terminated=protocol_terminated,
                    token_eos_terminated=token_eos_terminated,
                )
            )
            output["completion_mask"] = completion_mask
            if tool_mask is not None:
                output["tool_mask"] = tool_mask
            loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
            output["num_items_in_batch"] = self.accelerator.gather(loss_mask.sum()).sum()

            mode = "train" if self.model.training else "eval"
            batch_size = len(protocol_terminated)
            for name, count in diagnostics.items():
                self._metrics[mode][f"termination/{name.removesuffix('_count')}_ratio"].append(
                    count / batch_size
                )
            self._metrics[mode]["termination/post_terminal_generation_stopped_ratio"].append(
                (self._post_terminal_generation_count - post_terminal_before) / batch_size
            )
            for environment, protocol_done, token_done in zip(
                self.environments,
                protocol_terminated,
                token_eos_terminated,
                strict=True,
            ):
                environment._annotate_termination_mask(
                    {
                        "mode": "environment_terminal",
                        "protocol_terminated": protocol_done,
                        "token_eos_terminated": token_done,
                        "length_truncated": not (protocol_done or token_done),
                        "loss_masked": not (protocol_done or token_done),
                    }
                )
            return output

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer = TerminalAwareGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        quantization_config=trainer_quantization_config,
        peft_config=trainer_peft_config,
        environment_factory=environment_factory,
        callbacks=[VRAMGuardCallback()],
    )
    VRAMGuardCallback._check()
    try:
        train_result = trainer.train()
    except Exception as error:
        with suppress(RuntimeError):
            torch.cuda.synchronize()
        failure = build_grpo_failure_result(
            ready=ready,
            error=error,
            optimizer_updates=int(trainer.state.global_step),
            log_history=list(trainer.state.log_history),
            peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 3),
            peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 1024**2, 3),
            peak_device_used_mb=round(
                VRAMGuardCallback.peak_device_used_after_reclaim_mb, 3
            ),
            cuda_device=torch.cuda.get_device_name(0),
        )
        failure["rollout_audit"] = write_rollout_audit(
            output_path=output_path / "rollout_audit.jsonl",
            ready=ready,
            records=rollout_records,
            project_root=root,
        )
        result_path = output_path / "device_agent_arena_grpo_result.json"
        result_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        raise
    torch.cuda.synchronize()
    peak_allocated_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
    numeric_metrics = [
        float(value)
        for value in train_result.metrics.values()
        if isinstance(value, (int, float))
    ]
    if any(not torch.isfinite(torch.tensor(value)) for value in numeric_metrics):
        raise RuntimeError("GRPO emitted NaN or Inf metrics")

    trainer.save_model(str(output_path))
    trainer_state_path = output_path / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state_path))
    adapter_path = output_path / "adapter_model.safetensors"
    package_versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    result = {
        **ready,
        "status": "completed",
        "train_metrics": train_result.metrics,
        "optimizer_updates": int(trainer.state.global_step),
        "peak_vram_mb": {
            "allocated": peak_allocated_mb,
            "reserved": peak_reserved_mb,
            "device_used": round(
                VRAMGuardCallback.peak_device_used_after_reclaim_mb, 3
            ),
        },
        "cache_reclaim": {
            "count": VRAMGuardCallback.cache_reclaim_count,
            "peak_device_used_before_mb": round(
                VRAMGuardCallback.peak_device_used_before_reclaim_mb, 3
            ),
            "peak_device_used_after_mb": round(
                VRAMGuardCallback.peak_device_used_after_reclaim_mb, 3
            ),
        },
        "cuda_allocator": {
            "backend": torch.cuda.memory.get_allocator_backend(),
            "process_limit_fraction": round(allocator_limit_fraction, 6),
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "package_versions": package_versions,
        "adapter": {
            "path": _project_relative_path(root, adapter_path),
            "size_bytes": adapter_path.stat().st_size,
            "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        },
        "rollout_audit": write_rollout_audit(
            output_path=output_path / "rollout_audit.jsonl",
            ready=ready,
            records=rollout_records,
            project_root=root,
        ),
        "trainer_state": _project_relative_path(root, trainer_state_path),
        "note": "GRPO optimizer updates completed and the PEFT adapter was saved.",
    }
    result_path = output_path / "device_agent_arena_grpo_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def _resolve_project_path(project_root: Path, relative: str) -> Path:
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"configured path escapes project root: {relative}")
    return resolved


def _load_grpo_training_tasks(
    *,
    data_path: Path,
    selection_path: Path | None,
    train_task_limit: int,
) -> list[TaskSpec]:
    tasks = load_tasks(data_path)
    if selection_path is None:
        return list(tasks[:train_task_limit])
    selected = load_rl_task_selection(
        selection_path=selection_path,
        source_task_path=data_path,
        tasks=tasks,
        split="train",
    )
    if len(selected) != train_task_limit:
        raise ValueError("train_task_limit must equal the frozen training selection size")
    return list(selected)


def write_rollout_audit(
    *,
    output_path: Path,
    ready: dict[str, Any],
    records: list[dict[str, Any]],
    project_root: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "rollout_audit_header",
        "format_version": "rl-rollout-audit-v0.1",
        "config_sha256": ready["config_sha256"],
        "dataset_sha256": ready["dataset"]["sha256"],
        "selection_sha256": (ready["dataset"].get("selection") or {}).get("sha256"),
        "source": ready["source"],
        "record_count": len(records),
    }
    rows = [
        header,
        *(
            {"record_type": "rollout", "rollout_index": index, **record}
            for index, record in enumerate(records, 1)
        ),
    ]
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for row in rows
    ).encode("utf-8")
    output_path.write_bytes(payload)
    termination_counts = Counter(record["trial"]["termination"] for record in records)
    success_count = sum(bool(record["trial"]["grade"]["success"]) for record in records)
    terminal_message_count = sum(
        any(
            event["kind"] == "message"
            and event["payload"].get("kind") in {"answer", "refuse"}
            for event in record["trial"]["events"]
        )
        for record in records
    )
    return {
        "path": (
            _project_relative_path(project_root, output_path)
            if project_root is not None
            else output_path.as_posix()
        ),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": len(records),
        "success_count": success_count,
        "terminal_message_count": terminal_message_count,
        "termination_counts": dict(sorted(termination_counts.items())),
    }


def _project_relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


def _cached_model_snapshot(project_root: Path, model_id: str, revision: str) -> Path:
    cache_name = f"models--{model_id.replace('/', '--')}"
    snapshot = project_root / ".hf-cache" / "hub" / cache_name / "snapshots" / revision
    required = ("config.json", "tokenizer_config.json")
    missing = [filename for filename in required if not (snapshot / filename).is_file()]
    has_weights = (snapshot / "model.safetensors").is_file() or (
        snapshot / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors[.index.json]")
    if missing:
        raise FileNotFoundError(
            f"cached model snapshot is incomplete ({', '.join(missing)}): {snapshot}"
        )
    return snapshot.resolve()
