from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random
from typing import Literal

from device_agent_arena.domain.models import (
    DeviceState,
    FaultSpec,
    GoalSpec,
    TaskSpec,
)

Split = Literal["train", "dev", "test"]

_SPLIT_ENTITY_PREFIX = {
    "train": "训练场目标",
    "dev": "联调嘉宾",
    "test": "巡演对象",
}
_SPLIT_PRESET_PREFIX = {
    "train": "训练预设",
    "dev": "联调预设",
    "test": "巡演预设",
}
_SPLIT_SEED_BASE = {"train": 10_000, "dev": 20_000, "test": 30_000}
_V2_TARGET_PREFIX = {
    "train": {"person": "训练演员", "object": "训练道具"},
    "dev": {"person": "联调嘉宾", "object": "联调展品"},
    "test": {"person": "巡演角色", "object": "巡演装置"},
}
_V3_TARGET_PREFIX = {
    "train": {"person": "云杉成员", "object": "云杉道具"},
    "dev": {"person": "海棠嘉宾", "object": "海棠展件"},
    "test": {"person": "松石演员", "object": "松石装置"},
}
_V3_PRESET_PREFIX = {"train": "云杉方案", "dev": "海棠方案", "test": "松石方案"}
_V3_HISTORY_PREFIX = {"train": "云杉旧目标", "dev": "海棠旧目标", "test": "松石旧目标"}


@dataclass(frozen=True)
class _V3Profile:
    name: str
    initial_style: str
    goal_style: str
    required_tools: tuple[str, ...]
    inspect_state: bool = False
    save_preset: bool = False


_V3_PROFILES: dict[Split, tuple[_V3Profile, ...]] = {
    "train": (
        _V3Profile(
            "configure_save|cold_opposite|explicit|follow_active",
            "cold_opposite",
            "follow_active",
            (
                "connect_device",
                "calibrate_device",
                "set_mode",
                "set_orientation",
                "set_tracking",
                "save_preset",
            ),
            save_preset=True,
        ),
        _V3Profile(
            "reconfigure|ready_off_opposite|inspect|follow_active",
            "ready_off_opposite",
            "follow_active",
            ("get_device_state", "set_orientation", "set_tracking"),
            inspect_state=True,
        ),
        _V3Profile(
            "configure_save|connected_uncalibrated_correct|explicit|follow_active",
            "connected_uncalibrated_correct",
            "follow_active",
            ("calibrate_device", "set_mode", "set_tracking", "save_preset"),
            save_preset=True,
        ),
        _V3Profile(
            "reconfigure|cold_opposite|inspect|presentation_off",
            "cold_opposite",
            "presentation_off",
            ("get_device_state", "connect_device", "set_orientation", "set_mode"),
            inspect_state=True,
        ),
        _V3Profile(
            "reconfigure|ready_active_opposite|explicit|presentation_off",
            "ready_active_opposite",
            "presentation_off",
            ("set_orientation", "set_tracking", "set_mode"),
        ),
        _V3Profile(
            "recall_preset|recall_cold|explicit|recall_presentation",
            "recall_cold",
            "recall_presentation",
            ("connect_device", "calibrate_device", "recall_preset", "set_mode"),
        ),
    ),
    "dev": (
        _V3Profile(
            "reconfigure|cold_opposite|inspect|follow_active",
            "cold_opposite",
            "follow_active",
            (
                "get_device_state",
                "connect_device",
                "calibrate_device",
                "set_mode",
                "set_orientation",
                "set_tracking",
            ),
            inspect_state=True,
        ),
        _V3Profile(
            "configure_save|ready_off_correct|explicit|follow_active",
            "ready_off_correct",
            "follow_active",
            ("set_tracking", "save_preset"),
            save_preset=True,
        ),
        _V3Profile(
            "recall_preset|recall_ready_opposite|inspect|recall_follow",
            "recall_ready_opposite",
            "recall_follow",
            ("get_device_state", "recall_preset", "set_orientation"),
            inspect_state=True,
        ),
        _V3Profile(
            "configure_save|ready_active_correct|explicit|presentation_off",
            "ready_active_correct",
            "presentation_off",
            ("set_tracking", "set_mode", "save_preset"),
            save_preset=True,
        ),
    ),
    "test": (
        _V3Profile(
            "reconfigure|connected_uncalibrated_opposite|inspect|follow_active",
            "connected_uncalibrated_opposite",
            "follow_active",
            (
                "get_device_state",
                "calibrate_device",
                "set_orientation",
                "set_mode",
                "set_tracking",
            ),
            inspect_state=True,
        ),
        _V3Profile(
            "reconfigure|cold_opposite|explicit|presentation_off",
            "cold_opposite",
            "presentation_off",
            ("connect_device", "set_mode", "set_orientation"),
        ),
        _V3Profile(
            "recall_preset|recall_cold|inspect|recall_presentation",
            "recall_cold",
            "recall_presentation",
            (
                "get_device_state",
                "connect_device",
                "calibrate_device",
                "recall_preset",
                "set_mode",
            ),
            inspect_state=True,
        ),
        _V3Profile(
            "configure_save|ready_off_opposite|explicit|follow_active",
            "ready_off_opposite",
            "follow_active",
            ("set_orientation", "set_tracking", "save_preset"),
            save_preset=True,
        ),
    ),
}


def _axis_rng(seed: int, axis: str) -> Random:
    digest = hashlib.sha256(f"{seed}:{axis}".encode()).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def generate_rl_tasks(split: Split, *, count: int) -> tuple[TaskSpec, ...]:
    """Generate deterministic, split-disjoint long-horizon RL tasks."""
    if split not in _SPLIT_ENTITY_PREFIX:
        raise ValueError("split must be train, dev, or test")
    if count < 1 or count > 256:
        raise ValueError("count must be between 1 and 256")
    return tuple(_build_task(split, index) for index in range(count))


def _build_task(split: Split, index: int) -> TaskSpec:
    serial = index + 1
    target = f"{_SPLIT_ENTITY_PREFIX[split]}{serial:03d}"
    preset = f"{_SPLIT_PRESET_PREFIX[split]}{serial:03d}"
    tracking = "person" if index % 2 == 0 else "object"
    orientation = "landscape" if index % 4 < 2 else "portrait"
    initial_orientation = "portrait" if orientation == "landscape" else "landscape"
    requires_clarification = index % 3 == 0
    has_fault = index % 2 == 1
    fault_tool = "calibrate_device" if index % 4 == 1 else "set_tracking"

    target_phrase = "指定目标" if requires_clarification else target
    instructions = (
        "连接并校准设备，切换到跟随模式和{orientation}，跟踪{target}，最后保存为“{preset}”。",
        "为场景{serial:03d}完成设备初始化：连接、校准、设为跟随和{orientation}，锁定{target}并保存“{preset}”。",
        "请从未连接状态开始，把设备配置成{orientation}跟随模式，追踪{target}，配置成功后存成“{preset}”。",
        "执行完整拍摄准备：接入设备、完成校准、开启{orientation}跟随，跟踪{target}并创建预设“{preset}”。",
    )
    instruction = instructions[index % len(instructions)].format(
        serial=serial,
        orientation="横屏" if orientation == "landscape" else "竖屏",
        target=target_phrase,
        preset=preset,
    )
    reference_instruction = instruction.replace("指定目标", target)
    snapshot = {
        "orientation": orientation,
        "tracking": tracking,
        "mode": "follow",
        "target": target,
    }
    categories = ["multi_step", "state_dependency", "preset", "rl_compositional", "rl_long_horizon"]
    if requires_clarification:
        categories.append("missing_parameter")
    if has_fault:
        categories.extend(("tool_failure", "rl_fault"))

    return TaskSpec(
        id=f"{split}-rl_agent-{serial:03d}",
        version="1.0",
        split=split,
        categories=tuple(categories),
        seed=_SPLIT_SEED_BASE[split] + serial,
        instruction=instruction,
        reference_instruction=reference_instruction,
        initial_state=DeviceState(orientation=initial_orientation),
        fault_plan=(FaultSpec(tool=fault_tool, invocation=1),) if has_fault else (),
        clarifications={"target": target} if requires_clarification else {},
        goal=GoalSpec(
            final_state={
                "connected": True,
                "calibrated": True,
                "orientation": orientation,
                "tracking": tracking,
                "mode": "follow",
                "target": target,
                "presets": {preset: snapshot},
            },
            required_tools=(
                "connect_device",
                "calibrate_device",
                "set_mode",
                "set_orientation",
                "set_tracking",
                "save_preset",
            ),
            forbidden_tools=("recall_preset",),
            required_clarifications=("target",) if requires_clarification else (),
            terminal_behavior="answer",
        ),
        provenance={
            "template": f"rl_composite_{index % len(instructions)}",
            "generator": "rl-builtin-v0.1",
        },
    )


def generate_rl_tasks_v2(split: Split, *, count: int) -> tuple[TaskSpec, ...]:
    """Generate v0.2 tasks whose hidden labels are fully observable from interaction."""
    if split not in _V2_TARGET_PREFIX:
        raise ValueError("split must be train, dev, or test")
    if count < 1 or count > 256:
        raise ValueError("count must be between 1 and 256")
    return tuple(_build_task_v2(split, index) for index in range(count))


def _build_task_v2(split: Split, index: int) -> TaskSpec:
    serial = index + 1
    tracking = "person" if index % 2 == 0 else "object"
    kind_word = "人物" if tracking == "person" else "物体"
    target = f"{_V2_TARGET_PREFIX[split][tracking]}{serial:03d}"
    preset = f"{_SPLIT_PRESET_PREFIX[split]}{serial:03d}"
    orientation = "landscape" if index % 4 < 2 else "portrait"
    orientation_word = "横屏" if orientation == "landscape" else "竖屏"
    initial_orientation = "portrait" if orientation == "landscape" else "landscape"
    requires_clarification = index % 3 == 0
    has_fault = index % 2 == 1
    fault_tool = "calibrate_device" if index % 4 == 1 else "set_tracking"
    target_phrase = f"指定{kind_word}" if requires_clarification else target
    instructions = (
        "连接并校准设备，切换到跟随模式和{orientation}，以{kind}跟踪方式锁定{target}，"
        "最后保存为“{preset}”。",
        "为场景{serial:03d}完成设备初始化：连接、校准、设为{orientation}跟随模式，"
        "使用{kind}跟踪锁定{target}并保存“{preset}”。",
        "请从未连接状态开始，把设备配置成{orientation}跟随模式，以{kind}跟踪{target}，"
        "配置成功后存成“{preset}”。",
        "执行完整拍摄准备：接入设备、完成校准、开启{orientation}跟随，"
        "用{kind}跟踪{target}并创建预设“{preset}”。",
    )
    instruction = instructions[index % len(instructions)].format(
        serial=serial,
        orientation=orientation_word,
        kind=kind_word,
        target=target_phrase,
        preset=preset,
    )
    reference_instruction = instruction.replace(f"指定{kind_word}", target)
    snapshot = {
        "orientation": orientation,
        "tracking": tracking,
        "mode": "follow",
        "target": target,
    }
    categories = ["multi_step", "state_dependency", "preset", "rl_compositional", "rl_long_horizon"]
    if requires_clarification:
        categories.append("missing_parameter")
    if has_fault:
        categories.extend(("tool_failure", "rl_fault"))
    return TaskSpec(
        id=f"{split}-rl_agent-{serial:03d}",
        version="1.1",
        split=split,
        categories=tuple(categories),
        seed=_SPLIT_SEED_BASE[split] + serial,
        instruction=instruction,
        reference_instruction=reference_instruction,
        initial_state=DeviceState(orientation=initial_orientation),
        fault_plan=(FaultSpec(tool=fault_tool, invocation=1),) if has_fault else (),
        clarifications={"target": target} if requires_clarification else {},
        goal=GoalSpec(
            final_state={
                "connected": True,
                "calibrated": True,
                "orientation": orientation,
                "tracking": tracking,
                "mode": "follow",
                "target": target,
                "presets": {preset: snapshot},
            },
            required_tools=(
                "connect_device",
                "calibrate_device",
                "set_mode",
                "set_orientation",
                "set_tracking",
                "save_preset",
            ),
            forbidden_tools=("recall_preset",),
            required_clarifications=("target",) if requires_clarification else (),
            terminal_behavior="answer",
        ),
        provenance={
            "template": f"rl_composite_v2_{index % len(instructions)}",
            "generator": "rl-builtin-v0.2",
        },
    )


def generate_rl_tasks_v3(
    split: Split,
    *,
    count: int,
    root_seed: int = 0,
) -> tuple[TaskSpec, ...]:
    """Generate state-diverse tasks with seeded faults and structural split holdouts."""
    if split not in _V3_PROFILES:
        raise ValueError("split must be train, dev, or test")
    if count < 1 or count > 256:
        raise ValueError("count must be between 1 and 256")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    return tuple(_build_task_v3(split, index, root_seed=root_seed) for index in range(count))


def _build_task_v3(split: Split, index: int, *, root_seed: int) -> TaskSpec:
    serial = index + 1
    seed = root_seed * 1_000_000 + _SPLIT_SEED_BASE[split] + serial
    profile = _V3_PROFILES[split][index % len(_V3_PROFILES[split])]
    orientation = (
        "landscape" if _axis_rng(seed, "orientation").random() < 0.5 else "portrait"
    )
    opposite_orientation = "portrait" if orientation == "landscape" else "landscape"
    tracking = "person" if _axis_rng(seed, "tracking").random() < 0.5 else "object"
    kind_word = "人物" if tracking == "person" else "物体"
    target = f"{_V3_TARGET_PREFIX[split][tracking]}{serial:03d}"
    historical_target = f"{_V3_HISTORY_PREFIX[split]}{serial:03d}"
    preset = f"{_V3_PRESET_PREFIX[split]}{serial:03d}"
    desired_mode = (
        "presentation"
        if profile.goal_style.endswith("presentation")
        or profile.goal_style == "presentation_off"
        else "follow"
    )
    desired_tracking = "off" if profile.goal_style == "presentation_off" else tracking
    desired_target = None if desired_tracking == "off" else target

    saved_orientation = (
        opposite_orientation
        if profile.initial_style == "recall_ready_opposite"
        else orientation
    )
    saved_mode = "follow"
    saved_snapshot = {
        "orientation": saved_orientation,
        "tracking": tracking,
        "mode": saved_mode,
        "target": target,
    }
    initial_state = _v3_initial_state(
        profile=profile,
        desired_orientation=orientation,
        opposite_orientation=opposite_orientation,
        tracking=tracking,
        historical_target=historical_target,
        preset=preset,
        saved_snapshot=saved_snapshot,
    )

    can_clarify_target = (
        "set_tracking" in profile.required_tools and desired_tracking != "off"
    )
    requires_clarification = can_clarify_target and (
        _axis_rng(seed, "clarification").random() < 0.4
    )
    target_phrase = f"指定{kind_word}" if requires_clarification else target
    instruction = _v3_instruction(
        profile=profile,
        serial=serial,
        orientation=orientation,
        kind_word=kind_word,
        target_phrase=target_phrase,
        preset=preset,
        seed=seed,
    )
    reference_instruction = instruction.replace(f"指定{kind_word}", target)

    final_state: dict[str, object] = {
        "connected": True,
        "orientation": orientation,
        "tracking": desired_tracking,
        "mode": desired_mode,
        "target": desired_target,
    }
    if (
        desired_mode == "follow"
        or desired_tracking != "off"
        or "recall_preset" in profile.required_tools
    ):
        final_state["calibrated"] = True
    if profile.save_preset:
        final_state["presets"] = {
            preset: {
                "orientation": orientation,
                "tracking": desired_tracking,
                "mode": desired_mode,
                "target": desired_target,
            }
        }

    fault_candidates = tuple(
        tool for tool in profile.required_tools if tool != "get_device_state"
    )
    has_fault = bool(fault_candidates) and _axis_rng(seed, "fault_presence").random() < 0.5
    fault_plan: tuple[FaultSpec, ...] = ()
    if has_fault:
        fault_tool = fault_candidates[
            _axis_rng(seed, "fault_tool").randrange(len(fault_candidates))
        ]
        fault_plan = (
            FaultSpec(
                tool=fault_tool,
                invocation=1,
                code="transient_failure",
                probability=0.75,
            ),
        )

    if "recall_preset" in profile.required_tools:
        forbidden_tools = ("save_preset",)
        workflow = "recall_preset"
    elif profile.save_preset:
        forbidden_tools = ("recall_preset",)
        workflow = "configure_save"
    else:
        forbidden_tools = ("save_preset", "recall_preset")
        workflow = "reconfigure"
    information_mode = "inspect" if profile.inspect_state else "explicit"
    fault_stage = fault_plan[0].tool if fault_plan else "none"
    initial_profile = _v3_initial_profile_group(profile.initial_style)
    goal_shape = {
        "follow_active": "active_tracking",
        "presentation_off": "tracking_off",
        "recall_follow": "preset_recall",
        "recall_presentation": "preset_recall",
    }[profile.goal_style]

    categories = ["multi_step", "state_dependency", "rl_compositional"]
    action_horizon = len(profile.required_tools) + int(requires_clarification)
    if action_horizon >= 5:
        categories.append("rl_long_horizon")
    if profile.save_preset or "recall_preset" in profile.required_tools:
        categories.append("preset")
    if requires_clarification:
        categories.append("missing_parameter")
    if fault_plan:
        categories.extend(("tool_failure", "rl_fault"))

    return TaskSpec(
        id=f"{split}-rl_agent-{serial:03d}",
        version="1.2",
        split=split,
        categories=tuple(categories),
        seed=seed,
        instruction=instruction,
        reference_instruction=reference_instruction,
        initial_state=initial_state,
        fault_plan=fault_plan,
        clarifications={"target": target} if requires_clarification else {},
        goal=GoalSpec(
            final_state=final_state,
            required_tools=profile.required_tools,
            forbidden_tools=forbidden_tools,
            required_clarifications=("target",) if requires_clarification else (),
            terminal_behavior="answer",
        ),
        provenance={
            "template": f"{split}_v3_{_axis_rng(seed, 'template').randrange(3)}",
            "generator": "rl-builtin-v0.3",
            "structure": profile.name,
            "structure_id": f"{profile.name}|fault={fault_stage}",
            "workflow": workflow,
            "initial_profile": initial_profile,
            "initial_style": profile.initial_style,
            "information_mode": information_mode,
            "goal_shape": goal_shape,
            "goal_variant": profile.goal_style,
            "fault_stage": fault_stage,
            "fault_model": "seeded-bernoulli-v0.1",
        },
    )


def _v3_initial_state(
    *,
    profile: _V3Profile,
    desired_orientation: str,
    opposite_orientation: str,
    tracking: str,
    historical_target: str,
    preset: str,
    saved_snapshot: dict[str, object],
) -> DeviceState:
    style = profile.initial_style
    if style == "cold_opposite":
        return DeviceState(orientation=opposite_orientation)
    if style == "connected_uncalibrated_correct":
        return DeviceState(connected=True, orientation=desired_orientation)
    if style == "connected_uncalibrated_opposite":
        return DeviceState(connected=True, orientation=opposite_orientation)
    if style == "ready_off_opposite":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=opposite_orientation,
            mode="follow",
        )
    if style == "ready_off_correct":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=desired_orientation,
            mode="follow",
        )
    if style == "ready_active_opposite":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=opposite_orientation,
            tracking=tracking,
            mode="follow",
            target=historical_target,
        )
    if style == "ready_active_correct":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=desired_orientation,
            tracking=tracking,
            mode="follow",
            target=historical_target,
        )
    if style == "recall_cold":
        return DeviceState(
            orientation=opposite_orientation,
            presets={preset: saved_snapshot},
        )
    if style == "recall_ready_opposite":
        return DeviceState(
            connected=True,
            calibrated=True,
            orientation=desired_orientation,
            presets={preset: saved_snapshot},
        )
    raise AssertionError(f"unknown v0.3 initial-state profile: {style}")


def _v3_initial_profile_group(style: str) -> str:
    if style in {"cold_opposite", "recall_cold"}:
        return "cold"
    if style.startswith("connected_uncalibrated"):
        return "connected"
    if style.startswith("ready_off") or style.startswith("recall_ready"):
        return "ready"
    if style.startswith("ready_active"):
        return "active"
    raise AssertionError(f"unknown v0.3 initial-state profile: {style}")


def _v3_instruction(
    *,
    profile: _V3Profile,
    serial: int,
    orientation: str,
    kind_word: str,
    target_phrase: str,
    preset: str,
    seed: int,
) -> str:
    orientation_word = "横屏" if orientation == "landscape" else "竖屏"
    templates = _axis_rng(seed, "template")
    variant = templates.randrange(3)
    if profile.inspect_state:
        prefix = ("先检查当前设备状态，", "读取设备状态后，", "请先确认设备现状，再")[variant]
    elif profile.initial_style.startswith("cold") or profile.initial_style == "recall_cold":
        prefix = ("设备当前未连接，请", "从未连接状态开始，", "请先接入设备，再")[variant]
    elif "uncalibrated" in profile.initial_style:
        prefix = (
            "设备已连接但尚未校准，请",
            "从已连接未校准状态开始，",
            "请先完成校准，再",
        )[variant]
    else:
        prefix = ("设备已在线，请", "基于当前在线设备，", "保持设备连接，并")[variant]

    if profile.goal_style.startswith("recall"):
        suffix = f"调用预设“{preset}”恢复拍摄配置"
        if profile.goal_style == "recall_presentation":
            suffix += "，随后切换到演示模式"
        elif "set_orientation" in profile.required_tools:
            suffix += f"，随后调整为{orientation_word}"
    elif profile.goal_style == "presentation_off":
        suffix = f"关闭跟踪并调整为{orientation_word}演示模式"
        if profile.save_preset:
            suffix += f"，保存为“{preset}”"
    else:
        suffix = f"调整为{orientation_word}跟随模式，使用{kind_word}跟踪{target_phrase}"
        if profile.save_preset:
            suffix += f"，保存为“{preset}”"
    return f"{prefix}{suffix}。场景编号{serial:03d}。"
