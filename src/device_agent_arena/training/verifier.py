from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import cached_model_snapshot


@dataclass(frozen=True)
class VerifierTrainingConfig:
    format_version: str
    model_id: str
    model_revision: str
    dataset: str
    manifest: str
    output_dir: str
    result_path: str
    seed: int
    max_length: int
    max_train_records: int
    max_steps: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    gradient_checkpointing: bool
    compute_dtype: str
    load_in_4bit: bool
    double_quant: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    minimum_overfit_accuracy: float = 0.9
    objective: str = "binary_classification"


def load_verifier_config(path: str | Path) -> VerifierTrainingConfig:
    config = VerifierTrainingConfig(
        **json.loads(Path(path).read_text(encoding="utf-8"))
    )
    if config.format_version != "0.3":
        raise ValueError("unsupported verifier config format_version")
    if config.max_length < 128:
        raise ValueError("max_length must be at least 128")
    if config.max_train_records < 5:
        raise ValueError("max_train_records must be at least 5")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if config.compute_dtype not in {"bfloat16", "float16"}:
        raise ValueError("compute_dtype must be bfloat16 or float16")
    if config.target_modules != "all-linear":
        raise ValueError("v0.3 verifier requires target_modules=all-linear")
    if config.lora_rank < 1 or config.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= config.lora_dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if config.objective not in {"binary_classification", "listwise_group_ranking"}:
        raise ValueError("unsupported verifier objective")
    if config.objective == "listwise_group_ranking" and config.max_train_records % 5:
        raise ValueError("listwise max_train_records must preserve groups of five")
    return config


def dry_run_verifier_training(
    config: VerifierTrainingConfig,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_path = _project_path(root, config.dataset)
    manifest_path = _project_path(root, config.manifest)
    payload = dataset_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("verifier dataset SHA-256 does not match manifest")
    audit = manifest.get("audit", {})
    record_count = sum(bool(line) for line in payload.splitlines())
    if record_count != audit.get("record_count"):
        raise ValueError("verifier record count does not match manifest audit")
    split_counts = dict(audit.get("split_counts", {}))
    if config.max_train_records > int(split_counts.get("train", 0)):
        raise ValueError("max_train_records exceeds the frozen train split")
    if audit.get("hidden_runtime_fields") != 0:
        raise ValueError("verifier runtime input contains forbidden hidden fields")
    if float(audit.get("schema_valid_negative_rate", 0.0)) < 0.70:
        raise ValueError("schema-valid hard negative gate failed")
    if float(audit.get("counterfactual_flip_record_rate", 0.0)) < 0.25:
        raise ValueError("state-flip counterfactual gate failed")
    return {
        "status": "ready",
        "objective": (
            "binary state-action verification"
            if config.objective == "binary_classification"
            else "listwise state-action ranking with auxiliary binary loss"
        ),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset": {
            "path": dataset_path.relative_to(root).as_posix(),
            "manifest": manifest_path.relative_to(root).as_posix(),
            "sha256": digest,
            "record_count": record_count,
            "train_records": int(split_counts["train"]),
            "dev_id_records": int(split_counts["dev_id"]),
            "dev_xschema_records": int(split_counts["dev_xschema"]),
        },
        "training_recipe": {
            "objective": config.objective,
            "seed": config.seed,
            "max_length": config.max_length,
            "max_train_records": config.max_train_records,
            "max_steps": config.max_steps,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": (
                config.per_device_train_batch_size
                * config.gradient_accumulation_steps
            ),
            "effective_candidate_records": (
                config.per_device_train_batch_size
                * config.gradient_accumulation_steps
                * (5 if config.objective == "listwise_group_ranking" else 1)
            ),
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "quantization": {
            "load_in_4bit": config.load_in_4bit,
            "quant_type": "nf4",
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
        },
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.target_modules,
        },
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "audit": audit,
        "note": "Dry-run validates frozen inputs only; it is not training evidence.",
    }


def select_verifier_training_records(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    train = [record for record in records if record["split"] == "train"]
    if max_records >= len(train):
        return train
    rng = random.Random(seed)
    positives = [record for record in train if int(record["label"]) == 1]
    negatives_by_type: dict[str, list[dict[str, Any]]] = {}
    for record in train:
        if int(record["label"]) == 0:
            negatives_by_type.setdefault(str(record["negative_type"]), []).append(record)
    rng.shuffle(positives)
    for values in negatives_by_type.values():
        rng.shuffle(values)
    positive_count = max_records // 2
    negative_count = max_records - positive_count
    selected_negatives: list[dict[str, Any]] = []
    negative_types = sorted(negatives_by_type)
    offsets = {name: 0 for name in negative_types}
    while len(selected_negatives) < negative_count:
        for name in negative_types:
            values = negatives_by_type[name]
            offset = offsets[name]
            if offset < len(values):
                selected_negatives.append(values[offset])
                offsets[name] += 1
                if len(selected_negatives) == negative_count:
                    break
    selected = [*positives[:positive_count], *selected_negatives]
    rng.shuffle(selected)
    return selected


def select_verifier_evaluation_records(
    records: list[dict[str, Any]],
    splits: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("evaluation splits must be non-empty and unique")
    selected = [record for record in records if record.get("split") in set(splits)]
    present = {str(record["split"]) for record in selected}
    missing = set(splits) - present
    if missing:
        raise ValueError(f"evaluation data is missing splits: {sorted(missing)}")
    return selected


def select_verifier_training_groups(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_records % 5:
        raise ValueError("listwise max_records must be divisible by five")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["split"] == "train":
            grouped.setdefault(str(record["group_id"]), []).append(record)
    invalid_groups = any(
        len(group) != 5 or sum(int(item["label"]) for item in group) != 1
        for group in grouped.values()
    )
    if invalid_groups:
        raise ValueError("listwise training requires one positive in every five-candidate group")
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    selected_groups = groups[: max_records // 5]
    return [
        record
        for group in selected_groups
        for record in sorted(group, key=lambda item: str(item["record_id"]))
    ]


def train_verifier(
    config: VerifierTrainingConfig,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_verifier_training(config, project_root=root)
    output_dir = _project_path(root, config.output_dir)
    result_path = _project_path(root, config.result_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty verifier output: {output_dir}")
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("verifier training requires the `train` optional dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("verifier training requires CUDA")
    if config.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    compute_dtype = (
        torch.bfloat16 if config.compute_dtype == "bfloat16" else torch.float16
    )
    set_seed(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache_root = root / ".hf-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    local_snapshot = cached_model_snapshot(
        model_id=config.model_id,
        model_revision=config.model_revision,
        cache_root=cache_root,
    )
    model_source = str(local_snapshot) if local_snapshot is not None else config.model_id
    common_load: dict[str, Any] = {
        "local_files_only": local_snapshot is not None,
        "cache_dir": str(cache_root),
    }
    if local_snapshot is None:
        common_load["revision"] = config.model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **common_load)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if config.load_in_4bit
        else None
    )
    model_kwargs = {
        **common_load,
        "num_labels": 2,
        "device_map": "auto",
        "attn_implementation": "eager",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["dtype"] = compute_dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    records = [
        json.loads(line)
        for line in _project_path(root, config.dataset).read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_records = (
        select_verifier_training_groups(
            records,
            max_records=config.max_train_records,
            seed=config.seed,
        )
        if config.objective == "listwise_group_ranking"
        else select_verifier_training_records(
            records,
            max_records=config.max_train_records,
            seed=config.seed,
        )
    )
    raw_lengths = tokenizer(
        [str(record["text"]) for record in train_records],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    truncation_count = sum(len(token_ids) > config.max_length for token_ids in raw_lengths)
    encoded = tokenizer(
        [str(record["text"]) for record in train_records],
        add_special_tokens=True,
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.tensor([int(record["label"]) for record in train_records], dtype=torch.long)

    class _TensorDataset(Dataset):
        def __len__(self) -> int:
            return len(labels) // 5 if config.objective == "listwise_group_ranking" else len(labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            if config.objective == "listwise_group_ranking":
                start = index * 5
                return {
                    "input_ids": encoded["input_ids"][start : start + 5],
                    "attention_mask": encoded["attention_mask"][start : start + 5],
                    "labels": labels[start : start + 5],
                }
            return {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "labels": labels[index],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=config.seed,
        data_seed=config.seed,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_ratio,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    class_counts = {
        label: sum(int(record["label"]) == label for record in train_records)
        for label in (0, 1)
    }
    class_weights = torch.tensor(
        [len(train_records) / (2 * class_counts[label]) for label in (0, 1)],
        dtype=torch.float32,
        device=model.device,
    )

    class _WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels_for_loss = inputs.pop("labels")
            if labels_for_loss.ndim == 2:
                batch_groups, group_size = labels_for_loss.shape
                flat_inputs = {
                    name: value.reshape(batch_groups * group_size, -1)
                    for name, value in inputs.items()
                }
                outputs = model(**flat_inputs)
                flat_labels = labels_for_loss.reshape(-1)
                positive_logits = outputs.logits.float()[:, 1].reshape(
                    batch_groups, group_size
                )
                target_positions = labels_for_loss.argmax(dim=1)
                ranking_loss = torch.nn.functional.cross_entropy(
                    positive_logits,
                    target_positions,
                )
                binary_loss = torch.nn.functional.cross_entropy(
                    outputs.logits.float(),
                    flat_labels,
                    weight=class_weights,
                )
                loss = ranking_loss + 0.25 * binary_loss
            else:
                outputs = model(**inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.float(),
                    labels_for_loss,
                    weight=class_weights,
                )
            return (loss, outputs) if return_outputs else loss

    trainer = _WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=_TensorDataset(),
    )
    train_result = trainer.train()
    torch.cuda.synchronize()
    peak_allocated_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    model.eval()
    diagnostic_count = (
        len(labels)
        if config.minimum_overfit_accuracy > 0
        else min(len(labels), 256)
    )
    correct = 0
    with torch.no_grad():
        for index in range(diagnostic_count):
            batch = {
                "input_ids": encoded["input_ids"][index : index + 1].to(model.device),
                "attention_mask": encoded["attention_mask"][index : index + 1].to(
                    model.device
                ),
            }
            prediction = int(model(**batch).logits.argmax(dim=-1).item())
            correct += int(prediction == int(labels[index]))
    overfit_accuracy = correct / diagnostic_count
    package_versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    result = {
        **ready,
        "status": "completed",
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "truncation_count": truncation_count,
        "overfit_accuracy": overfit_accuracy,
        "training_accuracy_diagnostic_records": diagnostic_count,
        "train_metrics": train_result.metrics,
        "package_versions": package_versions,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "note": "FunctionGemma sequence-classification LoRA training completed.",
        "hardware_gate": {
            "peak_allocated_le_6656_mb": peak_allocated_mb <= 6656,
            "peak_reserved_le_7372_mb": peak_reserved_mb <= 7372,
            "overfit_accuracy_ge_threshold": (
                overfit_accuracy >= config.minimum_overfit_accuracy
            ),
            "truncation_count_eq_0": truncation_count == 0,
        },
    }
    result["hardware_gate"]["passed"] = all(result["hardware_gate"].values())
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def summarize_verifier_scores(
    records: list[dict[str, Any]],
    scores: list[float],
    *,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    if len(records) != len(scores) or not records:
        raise ValueError("records and scores must have the same non-zero length")
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for record, score in zip(records, scores, strict=True):
        grouped.setdefault(str(record["group_id"]), []).append((record, float(score)))
    predictions = [int(score >= decision_threshold) for score in scores]
    labels = [int(record["label"]) for record in records]
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    positive_recall = sum(predictions[index] == 1 for index in positive_indices) / len(
        positive_indices
    )
    negative_recall = sum(predictions[index] == 0 for index in negative_indices) / len(
        negative_indices
    )
    top1_correct = 0
    pairwise_correct = 0.0
    pairwise_total = 0
    oracle_groups = 0
    group_sizes: list[int] = []
    for group in grouped.values():
        positives = [(record, score) for record, score in group if int(record["label"]) == 1]
        negatives = [(record, score) for record, score in group if int(record["label"]) == 0]
        if positives:
            oracle_groups += 1
        group_sizes.append(len(group))
        winner = max(group, key=lambda item: item[1])
        top1_correct += int(int(winner[0]["label"]) == 1)
        for _, positive_score in positives:
            for _, negative_score in negatives:
                pairwise_total += 1
                pairwise_correct += (
                    1.0
                    if positive_score > negative_score
                    else 0.5 if positive_score == negative_score else 0.0
                )
    negative_type_accuracy = {}
    for negative_type in sorted(
        {str(record["negative_type"]) for record in records if not record["label"]}
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if record["negative_type"] == negative_type
        ]
        negative_type_accuracy[negative_type] = sum(
            predictions[index] == 0 for index in indices
        ) / len(indices)
    return {
        "record_count": len(records),
        "decision_threshold": decision_threshold,
        "group_count": len(grouped),
        "record_accuracy": sum(
            prediction == label for prediction, label in zip(predictions, labels, strict=True)
        )
        / len(records),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
        "top1_action_accuracy": top1_correct / len(grouped),
        "pairwise_ranking_accuracy": pairwise_correct / pairwise_total,
        "candidate_pool_oracle": oracle_groups / len(grouped),
        "random_top1_baseline": sum(1 / size for size in group_sizes) / len(group_sizes),
        "negative_type_accuracy": negative_type_accuracy,
    }


def choose_balanced_decision_threshold(
    records: list[dict[str, Any]],
    scores: list[float],
) -> float:
    if len(records) != len(scores) or not records:
        raise ValueError("records and scores must have the same non-zero length")
    labels = [int(record["label"]) for record in records]
    if set(labels) != {0, 1}:
        raise ValueError("threshold calibration requires both labels")
    ordered = sorted(set(float(score) for score in scores))
    candidates = [0.0, 1.0, *ordered]
    candidates.extend(
        (left + right) / 2 for left, right in zip(ordered, ordered[1:], strict=False)
    )
    best_threshold = 0.5
    best_key = (-1.0, float("-inf"))
    positive = [index for index, label in enumerate(labels) if label == 1]
    negative = [index for index, label in enumerate(labels) if label == 0]
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        positive_recall = sum(predictions[index] == 1 for index in positive) / len(positive)
        negative_recall = sum(predictions[index] == 0 for index in negative) / len(negative)
        balanced = (positive_recall + negative_recall) / 2
        key = (balanced, -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def summarize_candidate_baselines(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank each frozen candidate group without using offline labels as features."""
    if not records:
        raise ValueError("records must be non-empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["group_id"]), []).append(record)
    no_ranker_correct = 0
    generic_correct = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: str(item["record_id"]))
        no_ranker_correct += int(int(ordered[0]["label"]) == 1)
        winner = max(
            ordered,
            key=lambda item: (
                score_generic_candidate(dict(item["input"])),
                _stable_action_tiebreak(dict(item["input"])),
            ),
        )
        generic_correct += int(int(winner["label"]) == 1)
    return {
        "group_count": len(grouped),
        "no_ranker_top1_accuracy": no_ranker_correct / len(grouped),
        "generic_rule_top1_accuracy": generic_correct / len(grouped),
    }


def build_verifier_promotion_gate(by_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dev_id = by_split["dev_id"]
    dev_xschema = by_split["dev_xschema"]
    baselines = dev_xschema["candidate_baselines"]
    negative_types = dict(dev_xschema["negative_type_accuracy"])
    top1 = float(dev_xschema["top1_action_accuracy"])
    checks = {
        "dev_id_pairwise_accuracy_ge_0_82": (
            float(dev_id["pairwise_ranking_accuracy"]) >= 0.82
        ),
        "dev_xschema_pairwise_accuracy_ge_0_75": (
            float(dev_xschema["pairwise_ranking_accuracy"]) >= 0.75
        ),
        "dev_xschema_balanced_accuracy_ge_0_75": (
            float(dev_xschema["balanced_accuracy"]) >= 0.75
        ),
        "dev_xschema_positive_recall_ge_0_70": (
            float(dev_xschema["positive_recall"]) >= 0.70
        ),
        "dev_xschema_top1_vs_no_ranker_ge_0_12": (
            top1 - float(baselines["no_ranker_top1_accuracy"]) >= 0.12
        ),
        "dev_xschema_top1_vs_generic_rule_ge_0_08": (
            top1 - float(baselines["generic_rule_top1_accuracy"]) >= 0.08
        ),
        "dev_xschema_candidate_oracle_ge_0_85": (
            float(dev_xschema["candidate_pool_oracle"]) >= 0.85
        ),
        "all_negative_types_ge_0_70": (
            bool(negative_types) and min(float(value) for value in negative_types.values()) >= 0.70
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "top1_delta_vs_no_ranker_pp": round(
            100 * (top1 - float(baselines["no_ranker_top1_accuracy"])), 2
        ),
        "top1_delta_vs_generic_rule_pp": round(
            100 * (top1 - float(baselines["generic_rule_top1_accuracy"])), 2
        ),
        "minimum_negative_type_accuracy": min(
            float(value) for value in negative_types.values()
        ),
    }


def score_generic_candidate(payload: dict[str, Any]) -> tuple[float, ...]:
    """A preregistered schema-only heuristic; no contract or oracle fields are read."""
    action = dict(payload["candidate_action"])
    arguments = dict(action.get("arguments", {}))
    tool = next(
        (
            dict(item["function"])
            for item in payload["tools"]
            if item.get("function", {}).get("name") == action.get("name")
        ),
        None,
    )
    if tool is None:
        return (0.0, 0.0, 0.0)
    parameters = dict(tool.get("parameters", {}))
    properties = dict(parameters.get("properties", {}))
    valid = not (set(arguments) - set(properties)) and not (
        set(parameters.get("required", ())) - set(arguments)
    )
    for name, value in arguments.items():
        specification = dict(properties.get(name, {}))
        valid = valid and ("enum" not in specification or value in specification["enum"])
    instruction = str(payload.get("instruction", "")).casefold()
    literal_matches = sum(
        str(value).casefold() in instruction for value in arguments.values()
    )
    description = str(tool.get("description", "")).casefold()
    terminal_penalty = int(
        not payload.get("successful_history")
        and any(token in description for token in ("finish", "complete", "close", "end"))
    )
    return (float(valid), float(literal_matches), float(-terminal_penalty))


def _stable_action_tiebreak(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload["candidate_action"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def evaluate_verifier(
    config: VerifierTrainingConfig,
    *,
    project_root: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    batch_size: int = 16,
    evaluation_data_path: str | Path | None = None,
    splits: tuple[str, ...] = ("dev_id", "dev_xschema"),
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    ready = dry_run_verifier_training(config, project_root=root)
    adapter = Path(adapter_path)
    if not adapter.is_absolute():
        adapter = _project_path(root, str(adapter))
    adapter = adapter.resolve()
    if not adapter.is_relative_to(root) or not adapter.is_dir():
        raise ValueError("adapter_path must be a project-local directory")
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("verifier evaluation requires the `train` dependencies") from exc
    compute_dtype = (
        torch.bfloat16 if config.compute_dtype == "bfloat16" else torch.float16
    )
    cache_root = root / ".hf-cache"
    snapshot = cached_model_snapshot(
        model_id=config.model_id,
        model_revision=config.model_revision,
        cache_root=cache_root,
    )
    if snapshot is None:
        raise FileNotFoundError("the frozen FunctionGemma snapshot is not cached")
    model_source = str(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if config.load_in_4bit
        else None
    )
    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "device_map": "auto",
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["dtype"] = compute_dtype
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        **model_kwargs,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    model.eval()
    source_dataset = (
        _project_path(root, config.dataset)
        if evaluation_data_path is None
        else _project_path(root, str(evaluation_data_path))
    )
    evaluation_bytes = source_dataset.read_bytes()
    records = [
        json.loads(line)
        for line in evaluation_bytes.decode("utf-8").splitlines()
        if line
    ]
    evaluation_records = select_verifier_evaluation_records(records, splits)
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(evaluation_records), batch_size):
            batch_records = evaluation_records[start : start + batch_size]
            encoded = tokenizer(
                [str(record["text"]) for record in batch_records],
                add_special_tokens=True,
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            probabilities = model(**encoded).logits.float().softmax(dim=-1)[:, 1]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
    if decision_threshold is None:
        if "dev_id" not in splits:
            raise ValueError("non-Dev evaluation requires a frozen decision_threshold")
        id_records = [record for record in evaluation_records if record["split"] == "dev_id"]
        id_scores = [
            score
            for record, score in zip(evaluation_records, scores, strict=True)
            if record["split"] == "dev_id"
        ]
        decision_threshold = choose_balanced_decision_threshold(id_records, id_scores)
    by_split = {
        split: {
            **summarize_verifier_scores(
                [record for record in evaluation_records if record["split"] == split],
                [
                    score
                    for record, score in zip(evaluation_records, scores, strict=True)
                    if record["split"] == split
                ],
                decision_threshold=decision_threshold,
            ),
            "candidate_baselines": summarize_candidate_baselines(
                [record for record in evaluation_records if record["split"] == split]
            ),
        }
        for split in splits
    }
    adapter_model = adapter / "adapter_model.safetensors"
    result = {
        "status": "completed",
        "objective": ready["objective"],
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "training_dataset_sha256": ready["dataset"]["sha256"],
        "evaluation_data": source_dataset.relative_to(root).as_posix(),
        "evaluation_data_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "evaluation_splits": list(splits),
        "decision_threshold": decision_threshold,
        "adapter": adapter.relative_to(root).as_posix(),
        "adapter_sha256": hashlib.sha256(adapter_model.read_bytes()).hexdigest(),
        "by_split": by_split,
        "scores": [
            {
                "record_id": record["record_id"],
                "split": record["split"],
                "group_id": record["group_id"],
                "label": record["label"],
                "negative_type": record["negative_type"],
                "score": score,
            }
            for record, score in zip(evaluation_records, scores, strict=True)
        ],
    }
    if set(splits) == {"dev_id", "dev_xschema"}:
        result["promotion_gate"] = build_verifier_promotion_gate(by_split)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = _project_path(root, str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"configured path escapes project root: {relative}")
    return path
