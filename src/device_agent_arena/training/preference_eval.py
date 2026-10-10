from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from device_agent_arena.agents.hf_backend import (
    cached_model_snapshot,
    uses_functiongemma_protocol,
)
from device_agent_arena.artifacts.run_io import git_provenance, source_tree_sha256
from device_agent_arena.training.preference import PREFERENCE_MANIFEST_VERSION
from device_agent_arena.training.qlora import _project_relative_path, _resolve_project_path
from device_agent_arena.training.sequential_dpo import (
    completion_logp,
    tokenize_preference_completion,
)


def summarize_preference_scores(scores: Iterable[dict[str, float]]) -> dict[str, Any]:
    rows = list(scores)
    if not rows:
        raise ValueError("at least one preference score is required")

    def summarize(key: str) -> dict[str, float | int]:
        values = [float(row[key]) for row in rows]
        return {
            "mean_margin": round(mean(values), 6),
            "median_margin": round(median(values), 6),
            "chosen_win_rate": round(
                sum(value > 0 for value in values) / len(values), 6
            ),
            "tie_rate": round(sum(value == 0 for value in values) / len(values), 6),
            "pair_count": len(values),
        }

    return {
        "sum_logp": summarize("sum_margin"),
        "length_normalized_logp": summarize("mean_margin"),
    }


def _load_verified_records(
    *, root: Path, dataset_path: str, manifest_path: str, model_id: str
) -> tuple[list[dict[str, Any]], str]:
    data_path = _resolve_project_path(root, dataset_path)
    manifest_file = _resolve_project_path(root, manifest_path)
    if not data_path.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("preference dataset or manifest is missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format_version") != PREFERENCE_MANIFEST_VERSION:
        raise ValueError("unsupported preference manifest format")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != digest:
        raise ValueError("preference dataset SHA-256 does not match manifest")
    expected_protocol = (
        "functiongemma" if uses_functiongemma_protocol(model_id) else "openai"
    )
    protocol = manifest.get("protocol", "openai")
    if protocol != expected_protocol:
        raise ValueError("preference protocol does not match the target model")
    records = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if manifest.get("pair_count") != len(records):
        raise ValueError("preference pair count does not match manifest")
    return records, digest


def score_preferences(
    *,
    project_root: str | Path,
    dataset_path: str,
    manifest_path: str,
    model_id: str,
    model_revision: str,
    adapter_path: str | None,
    max_length: int,
    load_in_4bit: bool,
    compute_dtype: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    records, dataset_sha256 = _load_verified_records(
        root=root,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        model_id=model_id,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("preference scoring requires CUDA")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("preference scoring dependencies are missing") from exc

    if compute_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("preference scoring bfloat16 requires CUDA bfloat16 support")
        dtype = torch.bfloat16
    elif compute_dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError("compute_dtype must be bfloat16 or float16")

    snapshot = cached_model_snapshot(
        model_id=model_id,
        model_revision=model_revision,
        cache_root=root / ".hf-cache",
    )
    source = str(snapshot) if snapshot is not None else model_id
    model_kwargs: dict[str, Any] = {
        "revision": None if snapshot is not None else model_revision,
        "device_map": "auto",
        "local_files_only": snapshot is not None,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            _resolve_project_path(root, adapter_path),
            is_trainable=False,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if snapshot is not None else model_revision,
        local_files_only=snapshot is not None,
    )
    device = next(model.parameters()).device
    scores: list[dict[str, float]] = []
    for record in records:
        chosen_ids, chosen_mask = tokenize_preference_completion(
            record, "chosen", tokenizer, max_length=max_length
        )
        rejected_ids, rejected_mask = tokenize_preference_completion(
            record, "rejected", tokenizer, max_length=max_length
        )
        with torch.no_grad():
            chosen_ids = chosen_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)
            chosen_sum = completion_logp(model, chosen_ids, chosen_mask)
            rejected_sum = completion_logp(model, rejected_ids, rejected_mask)
            chosen_mean = chosen_sum / chosen_mask[:, 1:].sum().clamp_min(1)
            rejected_mean = rejected_sum / rejected_mask[:, 1:].sum().clamp_min(1)
        scores.append(
            {
                "sum_margin": float(chosen_sum - rejected_sum),
                "mean_margin": float(chosen_mean - rejected_mean),
            }
        )
        del chosen_ids, chosen_mask, rejected_ids, rejected_mask
        del chosen_sum, rejected_sum, chosen_mean, rejected_mean
        torch.cuda.empty_cache()

    result = {
        "status": "completed",
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter": adapter_path,
        "dataset": _project_relative_path(root, _resolve_project_path(root, dataset_path)),
        "dataset_sha256": dataset_sha256,
        "pair_count": len(records),
        "max_length": max_length,
        "load_in_4bit": load_in_4bit,
        "compute_dtype": compute_dtype,
        "scores": summarize_preference_scores(scores),
        "source": {**git_provenance(root), "tree_sha256": source_tree_sha256()},
        "note": (
            "Offline preference ranking is diagnostic; it is not an environment success metric."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result
