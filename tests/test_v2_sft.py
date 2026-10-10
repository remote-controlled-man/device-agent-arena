from __future__ import annotations

import json
from pathlib import Path

import pytest

from device_agent_arena.v2.sft import (
    audit_functiongemma_records,
    export_functiongemma_hard_state_sft,
    export_functiongemma_schema_sft,
    export_functiongemma_sft,
)


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(task_id: str, completion: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "安全执行"},
            {"role": "user", "content": "测试任务"},
        ],
        "completion": [{"role": "assistant", "content": completion}],
        "tools": [],
        "metadata": {
            "task_id": task_id,
            "decision_kind": "clarify" if "CLARIFY" in completion else "answer",
            "format_version": "deviceops-ul-sft-v1.1-next-decision",
        },
    }


def test_export_functiongemma_sft_combines_sources_with_native_calls(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.jsonl"
    grounded = tmp_path / "grounded.jsonl"
    output = tmp_path / "functiongemma.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write(oracle, [_record("train-1", "[CLARIFY field=device] 哪一台？")])
    _write(grounded, [_record("train-2", "[ANSWER] 完成")])

    manifest = export_functiongemma_sft(
        oracle_data=oracle,
        grounded_data=grounded,
        output=output,
        manifest_output=manifest_path,
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert manifest["record_count"] == 2
    assert manifest["source_record_counts"] == {"grounded": 1, "oracle": 1}
    assert manifest["completion_without_native_call_count"] == 0
    assert all(record["prompt"][0]["role"] == "developer" for record in records)
    assert all(record["completion"][0].get("tool_calls") for record in records)
    assert all("archetype" in record["metadata"] for record in records)
    assert manifest["sha256"]


def test_export_functiongemma_sft_excludes_ungrounded_shared_oracle_and_remaps_train_ids(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.jsonl"
    grounded = tmp_path / "grounded.jsonl"
    ungrounded = _record("v1-train-shared_impact_accept-000", "[ANSWER] 完成")
    ungrounded["completion"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {
                        "name": "inspect_device",
                        "arguments": json.dumps({"device_id": "tr-living-57"}),
                    },
                }
            ],
        }
    ]
    grounded_record = _record("v1-train-shared_impact_accept-000", "[ANSWER] 完成")
    grounded_record["prompt"][1]["content"] = "工具刚返回 tr-living-57"
    grounded_record["metadata"]["archetype"] = "shared_impact_accept"
    _write(oracle, [ungrounded])
    _write(grounded, [grounded_record])
    output = tmp_path / "out.jsonl"

    manifest = export_functiongemma_sft(
        oracle_data=oracle,
        grounded_data=grounded,
        output=output,
        manifest_output=tmp_path / "manifest.json",
    )
    payload = output.read_text(encoding="utf-8")

    assert manifest["excluded_oracle_records"] == 1
    assert manifest["model_visible_train_prefix_count"] == 0
    assert "tr-living-57" not in payload
    assert "grounded" in payload


def test_export_functiongemma_sft_refuses_non_train_records(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.jsonl"
    grounded = tmp_path / "grounded.jsonl"
    record = _record("test-1", "[ANSWER] 完成")
    record["metadata"]["task_id"] = "v1-test-direct-001"
    _write(oracle, [record])
    _write(grounded, [_record("train-2", "[ANSWER] 完成")])

    with pytest.raises(ValueError, match="Train-only"):
        export_functiongemma_sft(
            oracle_data=oracle,
            grounded_data=grounded,
            output=tmp_path / "out.jsonl",
            manifest_output=tmp_path / "manifest.json",
        )


def test_functiongemma_token_audit_verifies_completion_prefix_and_lengths() -> None:
    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            length = len(messages) * 50 + (10 if kwargs.get("add_generation_prompt") else 0)
            # A full conversation replaces the 10-token generation prefix with a
            # 50-token completion, so its first prompt length remains identical.
            if not kwargs.get("add_generation_prompt"):
                length += 10
            return {"input_ids": list(range(length))}

    records = [
        {
            "prompt": [{"role": "developer", "content": "x"}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "finish_task", "arguments": {}},
                        }
                    ],
                }
            ],
            "tools": [],
        }
    ]

    result = audit_functiongemma_records(
        records,
        processor=FakeProcessor(),
        max_length=128,
    )

    assert result["record_count"] == 1
    assert result["completion_only_loss_verified"] is True
    assert result["min_completion_tokens"] == 50
    assert result["truncation_count"] == 0


def test_schema_sft_balances_profiles_and_oversamples_hard_cases(tmp_path: Path) -> None:
    source = tmp_path / "native.jsonl"
    ordinary = _record("v1-train-direct_low_risk-000", "[ANSWER] 完成")
    hard = _record("v1-train-shared_impact_accept-000", "[ANSWER] 完成")
    hard["metadata"]["archetype"] = "shared_impact_accept"
    native = [
        {
            **record,
            "prompt": [{"role": "developer", "content": "use tools"}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "finish_task",
                                "arguments": {"message": "完成"},
                            },
                        }
                    ],
                }
            ],
            "tools": [],
        }
        for record in (ordinary, hard)
    ]
    _write(source, native)
    output = tmp_path / "schema.jsonl"

    manifest = export_functiongemma_schema_sft(
        data=source,
        output=output,
        manifest_output=tmp_path / "manifest.json",
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert manifest["source_record_count"] == 2
    assert manifest["record_count"] == 3
    assert manifest["hard_case_extra_record_count"] == 1
    assert sum(manifest["schema_profile_counts"].values()) == 3
    assert {record["metadata"]["schema_profile"] for record in records} <= {
        "identity",
        "schema-a",
        "schema-b",
    }
    hard_profiles = {
        record["metadata"]["schema_profile"]
        for record in records
        if record["metadata"].get("archetype") == "shared_impact_accept"
    }
    assert len(hard_profiles) == 2


def test_hard_state_sft_weights_identity_critical_states_and_keeps_replay(
    tmp_path: Path,
) -> None:
    critical = {
        "prompt": [
            {"role": "developer", "content": "use tools"},
            {
                "role": "tool",
                "content": {
                    "name": "inspect_routines",
                    "response": {
                        "data": {
                            "routines": [
                                {"id": "r1", "device_id": "d1", "active": True}
                            ]
                        }
                    },
                },
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "request_confirmation",
                            "arguments": {
                                "operation": "pause_routine:r1",
                                "message": "continue?",
                            },
                        },
                    }
                ],
            }
        ],
        "tools": [],
        "metadata": {
            "task_id": "v1-train-shared_impact_accept-000",
            "archetype": "shared_impact_accept",
            "schema_profile": "identity",
        },
    }
    replay = {
        **critical,
        "metadata": {
            "task_id": "v1-train-direct_low_risk-000",
            "archetype": "direct_low_risk",
            "schema_profile": "schema-a",
        },
        "completion": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish_task",
                            "arguments": {"message": "done"},
                        },
                    }
                ],
            }
        ],
    }
    source = tmp_path / "source.jsonl"
    _write(source, [critical, replay])

    manifest = export_functiongemma_hard_state_sft(
        data=source,
        output=(output := tmp_path / "hard.jsonl"),
        manifest_output=tmp_path / "manifest.json",
        replay_modulus=1,
    )

    assert manifest["critical_source_record_count"] == 1
    assert manifest["identity_critical_extra_record_count"] == 2
    assert manifest["replay_record_count"] == 1
    assert manifest["record_count"] == 4
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    counterfactuals = [
        record for record in records if record["metadata"].get("counterfactual_role_ids")
    ]
    assert len(counterfactuals) == 2
    assert all("pause_routine:r1" not in json.dumps(record) for record in counterfactuals)
