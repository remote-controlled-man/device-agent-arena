from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from device_agent_arena.agents.hf_backend import cached_model_snapshot
from device_agent_arena.artifacts.run_io import git_provenance, source_tree_sha256
from device_agent_arena.training.dpo import DpoTrainingConfig, dry_run_dpo_training
from device_agent_arena.training.qlora import _project_relative_path, _resolve_project_path


def _is_update_boundary(processed: int, process_limit: int, accumulation: int) -> bool:
    return processed % accumulation == 0 or processed == process_limit


def dpo_pair_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    logit = beta * (
        (policy_chosen_logp - reference_chosen_logp)
        - (policy_rejected_logp - reference_rejected_logp)
    )
    return -functional.logsigmoid(logit)


def dpo_pair_coefficient(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Derivative of sigmoid DPO loss with respect to chosen policy log-probability."""
    logit = beta * (
        (policy_chosen_logp - reference_chosen_logp)
        - (policy_rejected_logp - reference_rejected_logp)
    )
    return (-beta * torch.sigmoid(-logit)).detach()


def tokenize_preference_completion(
    record: dict[str, Any],
    completion_key: str,
    tokenizer: Any,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from trl.data_utils import apply_chat_template

    rendered = apply_chat_template(record, tokenizer, tools=record["tools"])
    prompt_ids = tokenizer(rendered["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        rendered[completion_key], add_special_tokens=False
    )["input_ids"]
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        raise ValueError("sequential DPO record exceeds the frozen max_length")
    if not completion_ids:
        raise ValueError("sequential DPO completion is empty")
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)


def completion_logp(
    model: Any, input_ids: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = completion_mask[:, 1:]
    token_logps = torch.gather(
        functional.log_softmax(logits.float(), dim=-1),
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1).squeeze(0)


def _copy_reference_adapter(model: Any) -> None:
    config = model.peft_config["default"]
    model.add_adapter("ref", config)
    for name, parameter in model.named_parameters():
        if ".default." not in name:
            continue
        reference = model.get_parameter(name.replace(".default.", ".ref."))
        reference.data.copy_(parameter.data)
        reference.requires_grad_(False)
    model.set_adapter("default")


def train_sequential_dpo(
    config: DpoTrainingConfig, *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_dpo_training(config, project_root=root)
    output_path = _resolve_project_path(root, config.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite sequential DPO output: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("sequential DPO requires CUDA")
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )
    from trl.trainer.utils import use_adapter

    compute_dtype = (
        torch.bfloat16
        if config.quantization.compute_dtype == "bfloat16"
        else torch.float16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    snapshot = cached_model_snapshot(
        model_id=config.model_id,
        model_revision=config.model_revision,
        cache_root=root / ".hf-cache",
    )
    source = str(snapshot) if snapshot is not None else config.model_id
    base_model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=None if snapshot is not None else config.model_revision,
        quantization_config=quantization,
        device_map="auto",
        local_files_only=snapshot is not None,
    )
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    model = PeftModel.from_pretrained(
        base_model,
        _resolve_project_path(root, config.initial_adapter),
        is_trainable=True,
    )
    _copy_reference_adapter(model)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if snapshot is not None else config.model_revision,
        local_files_only=snapshot is not None,
    )
    records = [
        json.loads(line)
        for line in _resolve_project_path(root, config.train_data)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    encoded = [
        (
            tokenize_preference_completion(
                record, "chosen", tokenizer, max_length=config.max_length
            ),
            tokenize_preference_completion(
                record, "rejected", tokenizer, max_length=config.max_length
            ),
        )
        for record in records
    ]
    rng = random.Random(config.seed)
    rng.shuffle(encoded)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    update_count = (
        config.max_steps
        if config.max_steps > 0
        else math.ceil(len(encoded) * config.num_train_epochs / config.gradient_accumulation_steps)
    )
    warmup_steps = math.ceil(update_count * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, update_count)
    device = next(model.parameters()).device
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Match TRL DPO's default disable_dropout=True while retaining autograd.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    logs = []
    processed = 0
    global_step = 0
    process_limit = min(len(encoded), update_count * config.gradient_accumulation_steps)
    for chosen, rejected in encoded[:process_limit]:
        if global_step >= update_count:
            break
        chosen_ids, chosen_mask = (tensor.to(device) for tensor in chosen)
        rejected_ids, rejected_mask = (tensor.to(device) for tensor in rejected)
        with torch.no_grad(), use_adapter(model, "ref"):
            reference_chosen = completion_logp(model, chosen_ids, chosen_mask)
            reference_rejected = completion_logp(model, rejected_ids, rejected_mask)
        with torch.no_grad():
            policy_chosen = completion_logp(model, chosen_ids, chosen_mask)
            policy_rejected = completion_logp(model, rejected_ids, rejected_mask)
            coefficient = dpo_pair_coefficient(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=config.beta,
            )
            loss = dpo_pair_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=config.beta,
            )
            margin = config.beta * (
                (policy_chosen - reference_chosen)
                - (policy_rejected - reference_rejected)
            )
        group_start = (processed // config.gradient_accumulation_steps) * (
            config.gradient_accumulation_steps
        )
        group_size = min(
            config.gradient_accumulation_steps,
            process_limit - group_start,
        )
        chosen_logp = completion_logp(model, chosen_ids, chosen_mask)
        (coefficient * chosen_logp / group_size).backward()
        del chosen_logp
        rejected_logp = completion_logp(model, rejected_ids, rejected_mask)
        (-coefficient * rejected_logp / group_size).backward()
        del rejected_logp
        logged_loss = float(loss)
        logged_margin = float(margin)
        processed += 1
        boundary = _is_update_boundary(
            processed, process_limit, config.gradient_accumulation_steps
        )
        if boundary:
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            logs.append(
                {
                    "step": global_step,
                    "pair_index": processed,
                    "loss": logged_loss,
                    "reward_margin": logged_margin,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
        del (
            chosen_ids,
            chosen_mask,
            rejected_ids,
            rejected_mask,
            reference_chosen,
            reference_rejected,
            policy_chosen,
            policy_rejected,
            coefficient,
            loss,
            margin,
        )
        # Each pair is intentionally an activation boundary; release cached blocks so a
        # later long sequence cannot inherit fragmentation from shorter examples.
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    output_path.mkdir(parents=True, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(output_path, selected_adapters=["default"])
    tokenizer.save_pretrained(output_path)
    adapter_model = output_path / "adapter_model.safetensors"
    result = {
        **ready,
        "status": "completed",
        "implementation": "sequential-exact-gradient-v0.1",
        "dropout_disabled": True,
        "optimizer_weight_decay": 0.0,
        "completed_updates": global_step,
        "processed_pairs": processed,
        "peak_vram_mb": peak_vram_mb,
        "cuda_device": torch.cuda.get_device_name(0),
        "logs": logs,
        "adapter": {
            "path": _project_relative_path(root, adapter_model),
            "size_bytes": adapter_model.stat().st_size,
            "sha256": hashlib.sha256(adapter_model.read_bytes()).hexdigest(),
        },
        "source": {**git_provenance(root), "tree_sha256": source_tree_sha256()},
        "note": "Sequential DPO training completed; environment evaluation is still required.",
    }
    (output_path / "device_agent_arena_sequential_dpo_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
