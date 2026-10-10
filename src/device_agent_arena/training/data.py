from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from device_agent_arena.agents.hf_backend import normalize_functiongemma_messages
from device_agent_arena.agents.oracle import OracleAgent
from device_agent_arena.domain.models import TaskSpec
from device_agent_arena.harness.runner import EvaluationHarness, HarnessConfig
from device_agent_arena.prompts import BASE_SYSTEM_PROMPT, LEGACY_SYSTEM_PROMPT
from device_agent_arena.simulator.tools import tool_schemas

PREFIXES = ("", "请", "麻烦", "现在", "立刻", "可以帮我", "请马上", "帮我")
SUFFIXES = (
    "",
    "，谢谢",
    "，就这样",
    "，完成后告诉我",
    "，不要改其他设置",
    "，确认执行结果",
    "，如果失败请处理",
    "，按安全流程执行",
)
CATEGORIES = (
    "direct",
    "multi_step",
    "state_dependency",
    "missing_parameter",
    "irrelevant",
    "asr_noise",
    "tool_failure",
    "preset",
)


def _assistant_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                },
            }
        ],
    }


def _tool_result(name: str, payload: dict[str, Any], call_id: str = "call_1") -> dict:
    ok = bool(payload.get("ok"))
    normalized_payload = {
        "name": name,
        "ok": ok,
        "code": payload.get("code", "ok" if ok else "error"),
        "message": payload.get("message", "operation completed" if ok else "operation failed"),
        "state_delta": payload.get("state_delta", {}),
        "data": payload.get("data", {}),
    }
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True),
    }


def _success_result(
    name: str, arguments: dict[str, Any] | None = None, call_id: str = "call_1"
) -> dict:
    arguments = arguments or {}
    if name == "connect_device":
        message = "device connected"
        state_delta = {"connected": True}
    elif name == "calibrate_device":
        message = "device calibrated"
        state_delta = {"calibrated": True}
    elif name == "set_orientation":
        orientation = arguments["orientation"]
        message = f"orientation set to {orientation}"
        state_delta = {"orientation": orientation}
    elif name == "set_mode":
        mode = arguments["mode"]
        message = f"mode set to {mode}"
        state_delta = {"mode": mode}
    elif name == "set_tracking":
        kind = arguments["kind"]
        target = None if kind == "off" else arguments.get("target")
        message = f"tracking set to {kind}"
        state_delta = {"tracking": kind, "target": target}
    elif name == "recall_preset":
        preset_name = arguments["name"]
        message = f"preset recalled: {preset_name}"
        state_delta = {
            "orientation": "landscape",
            "mode": "presentation",
            "tracking": "off",
            "target": None,
        }
    else:
        message = "operation completed"
        state_delta = {}
    return _tool_result(
        name,
        {
            "ok": True,
            "code": "ok",
            "message": message,
            "state_delta": state_delta,
        },
        call_id,
    )


def _variant(index: int, variant_count: int = 4) -> tuple[int, str, str]:
    combinations_per_variant = 256 // variant_count
    core = index // combinations_per_variant
    within_variant = index % combinations_per_variant
    prefix = PREFIXES[(within_variant // 8) % len(PREFIXES)]
    suffix = SUFFIXES[within_variant % len(SUFFIXES)]
    return core, prefix, suffix


def _direct(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index, variant_count=8)
    variants = (
        ("把屏幕切换成横屏", "set_orientation", {"orientation": "landscape"}, "已切换为横屏。"),
        ("把屏幕切换成竖屏", "set_orientation", {"orientation": "portrait"}, "已切换为竖屏。"),
        ("进入演示模式", "set_mode", {"mode": "presentation"}, "已进入演示模式。"),
        ("返回正常模式", "set_mode", {"mode": "normal"}, "已返回正常模式。"),
        ("停止当前目标跟踪", "set_tracking", {"kind": "off"}, "已停止目标跟踪。"),
        (
            "开始跟踪演讲者",
            "set_tracking",
            {"kind": "person", "target": "演讲者"},
            "已开始跟踪演讲者。",
        ),
        (
            "开始跟踪白板",
            "set_tracking",
            {"kind": "object", "target": "白板"},
            "已开始跟踪白板。",
        ),
        ("进入跟随模式", "set_mode", {"mode": "follow"}, "已进入跟随模式。"),
    )
    request, tool, arguments, answer = variants[core]
    return [
        {"role": "user", "content": f"{prefix}{request}{suffix}"},
        _assistant_call(tool, arguments),
        _success_result(tool, arguments),
        {"role": "assistant", "content": answer},
    ]


def _multi_step(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    variants = (
        ("横屏", "landscape", "presentation", "演示"),
        ("竖屏", "portrait", "normal", "正常"),
        ("横屏", "landscape", "normal", "正常"),
        ("竖屏", "portrait", "presentation", "演示"),
    )
    orientation_word, orientation, mode, mode_word = variants[core]
    return [
        {
            "role": "user",
            "content": f"{prefix}先切换成{orientation_word}，再进入{mode_word}模式{suffix}",
        },
        _assistant_call("set_orientation", {"orientation": orientation}, "call_1"),
        _success_result("set_orientation", {"orientation": orientation}, "call_1"),
        _assistant_call("set_mode", {"mode": mode}, "call_2"),
        _success_result("set_mode", {"mode": mode}, "call_2"),
        {"role": "assistant", "content": "两项设备设置均已完成。"},
    ]


def _state_dependency(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    if core % 2 == 0:
        orientation = "landscape" if core == 0 else "portrait"
        word = "横屏" if orientation == "landscape" else "竖屏"
        return [
            {"role": "user", "content": f"{prefix}把设备切换成{word}{suffix}"},
            _assistant_call("set_orientation", {"orientation": orientation}, "call_1"),
            _tool_result(
                "set_orientation",
                {"ok": False, "code": "precondition_failed", "message": "device is not connected"},
                "call_1",
            ),
            _assistant_call("connect_device", {}, "call_2"),
            _success_result("connect_device", {}, "call_2"),
            _assistant_call("set_orientation", {"orientation": orientation}, "call_3"),
            _success_result("set_orientation", {"orientation": orientation}, "call_3"),
            {"role": "assistant", "content": f"设备已连接并切换为{word}。"},
        ]
    mode_word = "跟随" if core == 1 else "自动跟随"
    return [
        {"role": "user", "content": f"{prefix}让设备进入{mode_word}模式{suffix}"},
        _assistant_call("set_mode", {"mode": "follow"}, "call_1"),
        _tool_result(
            "set_mode",
            {"ok": False, "code": "precondition_failed", "message": "device is not calibrated"},
            "call_1",
        ),
        _assistant_call("calibrate_device", {}, "call_2"),
        _success_result("calibrate_device", {}, "call_2"),
        _assistant_call("set_mode", {"mode": "follow"}, "call_3"),
        _success_result("set_mode", {"mode": "follow"}, "call_3"),
        {"role": "assistant", "content": "校准完成，设备已进入跟随模式。"},
    ]


def _missing_parameter(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    targets = (("讲师", "person"), ("参会者", "person"), ("黑板", "object"), ("展品", "object"))
    target, kind = targets[core]
    request = ("开始跟踪", "开始目标跟踪", "帮我跟踪一下", "开启跟踪")[(index % 64) // 16]
    return [
        {"role": "user", "content": f"{prefix}{request}{suffix}"},
        {"role": "assistant", "content": "请问需要跟踪谁或什么目标？"},
        {"role": "user", "content": target},
        _assistant_call("set_tracking", {"kind": kind, "target": target}),
        _success_result("set_tracking", {"kind": kind, "target": target}),
        {"role": "assistant", "content": f"已开始跟踪{target}。"},
    ]


CLARIFICATION_REQUESTS = (
    "开始一个跟拍任务，但目标还没确定",
    "启动目标跟踪，先向我确认对象",
    "帮我开启追随功能，跟谁还没说",
    "打开镜头锁定，目标待补充",
    "新建一次持续跟拍，暂未指定目标",
    "让设备开始追踪，先别猜对象",
    "准备开启人物或物体跟踪",
    "开启自动取景，但还缺少跟拍目标",
    "启动一次镜头追随，具体对象稍后告诉你",
    "进入目标锁定流程，当前没有目标参数",
    "打开持续对准功能，先问清楚跟什么",
    "建立新的追踪任务，目标信息未给出",
    "让镜头保持跟随，先确认要跟谁",
    "启用物体或人物锁定，目标尚未说明",
    "开始追踪流程，不要自行填写目标",
    "运行跟拍功能，但目标字段是空的",
    "先准备目标跟踪，具体目标需要确认",
    "开启镜头追随，我还没有告诉你对象",
    "创建锁定任务，必须先询问目标",
    "启动持续追踪，缺少被跟踪对象",
    "打开跟随能力，目标名称稍后补充",
    "进入跟拍状态前先确认具体目标",
    "安排一次自动追踪，当前只缺目标",
    "启用镜头锁定，不能默认跟踪任何人",
    "开始人物或物体追随，但对象不明确",
    "启动跟踪功能，目标不能由你推测",
    "新建目标锁定，先收集目标参数",
    "让摄像头持续跟随，尚未选定对象",
    "打开智能跟拍，先问我要跟拍什么",
    "进入追踪模式前确认人物或物体",
    "准备执行目标跟拍，但目标还空着",
    "开始镜头锁定流程，等我补充目标",
)
CLARIFICATION_QUESTIONS = (
    "请问需要跟踪谁或什么目标？",
    "请提供要锁定的具体目标。",
    "需要跟拍谁或哪个对象？",
    "请先告诉我要追随的对象。",
    "目标参数还缺失，请问具体跟踪什么？",
    "请确认需要持续对准的目标。",
    "要开启跟拍，还需要提供目标名称。",
    "请问这次镜头需要锁定哪个对象？",
)
CLARIFICATION_PEOPLE = (
    "场记员",
    "灯光师",
    "音响师",
    "化妆师",
    "服装师",
    "道具师",
    "制片助理",
    "舞台经理",
    "安全员",
    "检票员",
    "主持助理",
    "摄像助理",
    "采访嘉宾",
    "课程助教",
    "展馆向导",
    "活动志愿者",
    "登台选手",
    "发言代表",
    "乐队指挥",
    "会议记录员",
    "器材管理员",
    "实验操作员",
    "仓库盘点员",
    "直播运营员",
    "节目统筹",
    "现场翻译",
    "技术支持员",
    "安保负责人",
    "赛事计时员",
    "候场演员",
    "产品演示员",
    "新闻发言人",
)
CLARIFICATION_OBJECTS = (
    "移动讲台",
    "展柜模型",
    "黄色路牌",
    "运输推车",
    "巡检小车",
    "黑色设备箱",
    "触控一体机",
    "悬挂横幅",
    "物料周转箱",
    "协作机械臂",
    "舞台聚光灯",
    "移动白板架",
    "展示转盘",
    "测量仪器",
    "红色消防箱",
    "样品托盘",
    "摄像轨道车",
    "导览标识牌",
    "便携工作站",
    "蓝色工具柜",
    "物流机器人",
    "大型包装箱",
    "实验台模型",
    "升降显示器",
    "比赛计分牌",
    "移动照明架",
    "展厅导览屏",
    "自动售货机",
    "白色储物柜",
    "演示无人机",
    "产品陈列架",
    "会议扬声器",
)


def _missing_parameter_clarification_v01(index: int) -> list[dict]:
    request = CLARIFICATION_REQUESTS[index % len(CLARIFICATION_REQUESTS)]
    prefix = PREFIXES[(index // len(CLARIFICATION_REQUESTS)) % len(PREFIXES)]
    suffix = SUFFIXES[(index // 4) % len(SUFFIXES)]
    entity_index = index % len(CLARIFICATION_PEOPLE)
    if (index // len(CLARIFICATION_PEOPLE)) % 2 == 0:
        target, kind = CLARIFICATION_PEOPLE[entity_index], "person"
    else:
        target, kind = CLARIFICATION_OBJECTS[entity_index], "object"
    question = CLARIFICATION_QUESTIONS[index % len(CLARIFICATION_QUESTIONS)]
    return [
        {"role": "user", "content": f"{prefix}{request}{suffix}"},
        {"role": "assistant", "content": question},
        {"role": "user", "content": target},
        _assistant_call("set_tracking", {"kind": kind, "target": target}),
        _success_result("set_tracking", {"kind": kind, "target": target}),
        {"role": "assistant", "content": f"已开始跟踪{target}。"},
    ]


PERSON_CLARIFICATION_REQUESTS_V02 = (
    "开启人物跟拍，但还没指定哪位人员",
    "我想自动跟拍某位工作人员",
    "让镜头追着一个人走",
    "锁定一位现场人员",
    "开始人物追随，具体是谁还没说",
    "新建人像跟拍任务，目标人物待确认",
    "让设备持续跟随某位人员",
    "打开人物锁定，先确认是哪一位",
    "创建人员跟踪任务，不要默认选人",
    "启动人像追踪，名字稍后补充",
    "镜头需要跟着某位工作人员",
    "开始跟拍一个人，先问清楚身份",
    "启用人物追随，目前只知道是人员",
    "准备锁定一位参与者",
    "开启人员跟踪，目标姓名为空",
    "让摄像头跟住某位现场人员",
)
OBJECT_CLARIFICATION_REQUESTS_V02 = (
    "开启物体跟拍，但还没指定哪件物品",
    "我想自动跟拍某个移动物体",
    "让镜头追着一件东西移动",
    "锁定某个现场物品",
    "开始物体追随，具体是什么还没说",
    "新建物品跟拍任务，目标物体待确认",
    "让设备持续跟随某件物品",
    "打开物体锁定，先确认是哪一个",
    "创建物品跟踪任务，不要默认选东西",
    "启动物体追踪，名称稍后补充",
    "镜头需要跟着某个移动装置",
    "开始跟拍一件物品，先问清楚名称",
    "启用物体追随，目前只知道是物品",
    "准备锁定某个设备",
    "开启物品跟踪，目标名称为空",
    "让摄像头跟住某件现场物品",
)
GENERIC_CLARIFICATION_REQUESTS_V02 = (
    "开始一次跟拍任务",
    "打开持续跟踪功能",
    "让镜头锁定一个目标",
    "创建新的追随任务",
    "启动自动跟拍",
    "帮我开启目标跟踪",
    "进入镜头锁定流程",
    "准备执行持续追随",
    "新建一个跟踪目标",
    "开始镜头跟随",
    "启用智能锁定",
    "安排一次目标追踪",
    "打开自动取景跟拍",
    "让设备持续对准目标",
    "运行目标跟随功能",
    "启动镜头追踪模式",
)
GENERIC_CLARIFICATION_REQUESTS_V02_B = (
    "开启一项新的跟拍",
    "启动持续锁定功能",
    "请镜头跟住一个对象",
    "建立新的追踪任务",
    "运行自动追随",
    "帮我打开目标锁定",
    "开始镜头追随流程",
    "准备进行持续跟拍",
    "新增一个锁定目标",
    "让镜头开始追踪",
    "启用自动追随",
    "安排一次镜头锁定",
    "打开智能取景追踪",
    "让设备保持跟拍",
    "运行持续跟踪能力",
    "启动自动锁定模式",
)
PERSON_CLARIFICATION_QUESTIONS_V02 = (
    "请问需要跟拍哪位人物？",
    "请提供要追随的人员姓名。",
    "请确认具体要锁定哪位人员。",
    "需要跟踪谁？请告诉我目标人物。",
)
OBJECT_CLARIFICATION_QUESTIONS_V02 = (
    "请问需要跟拍哪个物体？",
    "请提供要追随的物品名称。",
    "请确认具体要锁定哪个对象。",
    "需要跟踪什么？请告诉我目标物体。",
)


def _missing_parameter_clarification_v02(index: int) -> list[dict]:
    mode = index // 64
    within_mode = index % 64
    if mode == 0:
        requests = PERSON_CLARIFICATION_REQUESTS_V02
        kind = "person"
        questions = PERSON_CLARIFICATION_QUESTIONS_V02
    elif mode == 1:
        requests = OBJECT_CLARIFICATION_REQUESTS_V02
        kind = "object"
        questions = OBJECT_CLARIFICATION_QUESTIONS_V02
    else:
        requests = (
            GENERIC_CLARIFICATION_REQUESTS_V02
            if mode == 2
            else GENERIC_CLARIFICATION_REQUESTS_V02_B
        )
        kind = "person" if mode == 2 else "object"
        questions = CLARIFICATION_QUESTIONS
    request = requests[within_mode % len(requests)]
    prefix = PREFIXES[(within_mode // len(requests)) % 4]
    suffix = SUFFIXES[(within_mode // 8) % len(SUFFIXES)]
    target_pool = CLARIFICATION_PEOPLE if kind == "person" else CLARIFICATION_OBJECTS
    target = target_pool[index % len(target_pool)]
    question = questions[index % len(questions)]
    return [
        {"role": "user", "content": f"{prefix}{request}{suffix}"},
        {"role": "assistant", "content": question},
        {"role": "user", "content": target},
        _assistant_call("set_tracking", {"kind": kind, "target": target}),
        _success_result("set_tracking", {"kind": kind, "target": target}),
        {"role": "assistant", "content": f"已开始跟踪{target}。"},
    ]


def _irrelevant(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    requests = ("查询明天的天气", "预订一杯咖啡", "写一首短诗", "计算北京到上海的距离")
    return [
        {"role": "user", "content": f"{prefix}{requests[core]}{suffix}"},
        {"role": "assistant", "content": "该请求不属于设备控制能力范围，我不会调用设备工具。"},
    ]


def _asr_noise(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index, variant_count=8)
    variants = (
        ("把屏幕切成横平", "set_orientation", {"orientation": "landscape"}),
        ("把屏幕切成树屏", "set_orientation", {"orientation": "portrait"}),
        ("进入眼示模式", "set_mode", {"mode": "presentation"}),
        ("停止跟棕", "set_tracking", {"kind": "off"}),
        ("开始跟棕演讲者", "set_tracking", {"kind": "person", "target": "演讲者"}),
        ("开始跟棕白板", "set_tracking", {"kind": "object", "target": "白板"}),
        ("把屏幕切成横萍", "set_orientation", {"orientation": "landscape"}),
        ("进入正长模式", "set_mode", {"mode": "normal"}),
    )
    noisy, tool, arguments = variants[core]
    return [
        {"role": "user", "content": f"{prefix}{noisy}{suffix}"},
        _assistant_call(tool, arguments),
        _success_result(tool, arguments),
        {"role": "assistant", "content": "设备设置已完成。"},
    ]


def _tool_failure(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    variants = (
        ("把屏幕切换成横屏", "set_orientation", {"orientation": "landscape"}),
        ("把屏幕切换成竖屏", "set_orientation", {"orientation": "portrait"}),
        ("进入演示模式", "set_mode", {"mode": "presentation"}),
        ("返回正常模式", "set_mode", {"mode": "normal"}),
    )
    request, tool, arguments = variants[core]
    return [
        {"role": "user", "content": f"{prefix}{request}{suffix}"},
        _assistant_call(tool, arguments, "call_1"),
        _tool_result(
            tool,
            {
                "ok": False,
                "code": "transient_failure",
                "message": f"injected fault for {tool}",
            },
            "call_1",
        ),
        _assistant_call(tool, arguments, "call_2"),
        _success_result(tool, arguments, "call_2"),
        {"role": "assistant", "content": "临时故障已恢复，设备设置已完成。"},
    ]


def _preset(index: int) -> list[dict]:
    core, prefix, suffix = _variant(index)
    names = ("会议", "授课", "展台", "访谈")
    name = names[core]
    return [
        {"role": "user", "content": f"{prefix}恢复{name}预设{suffix}"},
        _assistant_call("recall_preset", {"name": name}),
        _success_result("recall_preset", {"name": name}),
        {"role": "assistant", "content": f"已恢复{name}预设。"},
    ]


BUILDERS = {
    "direct": _direct,
    "multi_step": _multi_step,
    "state_dependency": _state_dependency,
    "missing_parameter": _missing_parameter,
    "irrelevant": _irrelevant,
    "asr_noise": _asr_noise,
    "tool_failure": _tool_failure,
    "preset": _preset,
}


def generate_sft_records(
    per_category: int = 256,
    *,
    category_counts: Mapping[str, int] | None = None,
    clarification_curriculum: str | None = None,
) -> list[dict]:
    if per_category < 1 or per_category > 256:
        raise ValueError("per_category must be between 1 and 256")
    counts = {category: per_category for category in CATEGORIES}
    if category_counts is not None:
        unknown = set(category_counts) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown SFT categories: {sorted(unknown)}")
        counts.update(category_counts)
    for category, count in counts.items():
        maximum = 768 if category == "missing_parameter" else 256
        if count < 1 or count > maximum:
            raise ValueError(f"{category} count must be between 1 and {maximum}")

    if clarification_curriculum not in {None, "v0.1", "v0.2"}:
        raise ValueError("clarification_curriculum must be v0.1 or v0.2")
    curriculum = clarification_curriculum
    if curriculum is None and counts["missing_parameter"] > 256:
        curriculum = "v0.1"
    generator = (
        f"sft-builtin-clarification-{curriculum}" if curriculum is not None else "sft-builtin-v0.5"
    )
    records: list[dict] = []
    schemas = tool_schemas(include_interaction=False)
    seen_hashes: set[str] = set()
    for category in CATEGORIES:
        desired_count = counts[category]
        decision_buckets: dict[int, list[tuple[int, list[dict], dict, str]]] = defaultdict(list)
        bucket_hashes: dict[int, set[str]] = defaultdict(set)
        for source_index in range(256):
            if category == "missing_parameter" and curriculum == "v0.1":
                conversation = _missing_parameter_clarification_v01(source_index)
            elif category == "missing_parameter" and curriculum == "v0.2":
                conversation = _missing_parameter_clarification_v02(source_index)
            else:
                conversation = BUILDERS[category](source_index)
            assistant_turn = 0
            for message_index, message in enumerate(conversation):
                if message["role"] != "assistant":
                    continue
                prompt = [
                    {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
                    *conversation[:message_index],
                ]
                supervised = {"prompt": prompt, "completion": [message]}
                canonical = json.dumps(supervised, ensure_ascii=False, sort_keys=True)
                candidate_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if (
                    candidate_hash not in bucket_hashes[assistant_turn]
                    and candidate_hash not in seen_hashes
                ):
                    bucket_hashes[assistant_turn].add(candidate_hash)
                    decision_buckets[assistant_turn].append(
                        (source_index, prompt, message, canonical)
                    )
                assistant_turn += 1

        turns = sorted(decision_buckets)
        turn_count = len(turns)
        count_per_turn, remainder = divmod(desired_count, turn_count)
        allocations = {
            assistant_turn: min(
                count_per_turn + (position < remainder),
                len(decision_buckets[assistant_turn]),
            )
            for position, assistant_turn in enumerate(turns)
        }
        unallocated = desired_count - sum(allocations.values())
        while unallocated:
            progress = False
            for assistant_turn in turns:
                if allocations[assistant_turn] >= len(decision_buckets[assistant_turn]):
                    continue
                allocations[assistant_turn] += 1
                unallocated -= 1
                progress = True
                if unallocated == 0:
                    break
            if not progress:
                raise ValueError(f"not enough unique decisions to sample {category}")

        for assistant_turn, bucket in sorted(decision_buckets.items()):
            selected_count = allocations[assistant_turn]
            if selected_count == 0:
                continue
            selected_positions = [
                (position * len(bucket)) // selected_count for position in range(selected_count)
            ]
            for position in selected_positions:
                source_index, prompt, completion, canonical = bucket[position]
                supervised = {"prompt": prompt, "completion": [completion]}
                canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if canonical_hash in seen_hashes:
                    raise ValueError(
                        f"duplicate SFT decision in {category} source {source_index} "
                        f"turn {assistant_turn}"
                    )
                seen_hashes.add(canonical_hash)
                records.append(
                    {
                        **supervised,
                        "tools": schemas,
                        "metadata": {
                            "category": category,
                            "source_split": "train",
                            "source_index": source_index,
                            "assistant_turn": assistant_turn,
                            "decision_type": (
                                "tool_call" if completion.get("tool_calls") else "message"
                            ),
                            "generator": generator,
                            "canonical_hash": canonical_hash,
                        },
                    }
                )
    return records


def export_sft_dataset(
    data_path: str | Path, manifest_path: str | Path, records: list[dict]
) -> dict:
    destination = Path(data_path)
    manifest_destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n"
    )
    destination.write_bytes(payload.encode("utf-8"))
    generators = {record["metadata"]["generator"] for record in records}
    if len(generators) != 1:
        raise ValueError("all SFT records must share one generator version")
    manifest = {
        "format_version": "0.1",
        "generator": generators.pop(),
        "record_count": len(records),
        "unique_supervision_count": len(
            {record["metadata"]["canonical_hash"] for record in records}
        ),
        "oversampled_record_count": sum(
            "oversample_index" in record["metadata"] for record in records
        ),
        "category_counts": dict(
            sorted(Counter(record["metadata"]["category"] for record in records).items())
        ),
        "decision_type_counts": dict(
            sorted(
                Counter(record["metadata"]["decision_type"] for record in records).items()
            )
        ),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    manifest_destination.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def convert_sft_records_for_functiongemma(records: Iterable[dict]) -> list[dict]:
    """Convert train-only supervision to FunctionGemma's native tool protocol."""
    converted: list[dict] = []
    for source in records:
        record = deepcopy(source)
        prompt = list(record["prompt"])
        completion = list(record["completion"])
        messages = normalize_functiongemma_messages([*prompt, *completion])
        record["prompt"] = messages[: len(prompt)]
        record["completion"] = messages[len(prompt) :]
        source_hash = str(record["metadata"]["canonical_hash"])
        record["metadata"]["source_canonical_hash"] = source_hash
        record["metadata"]["generator"] = (
            f"{record['metadata']['generator']}-functiongemma-v0.1"
        )
        canonical = json.dumps(
            {"prompt": record["prompt"], "completion": record["completion"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["metadata"]["canonical_hash"] = canonical_hash
        converted.append(record)
    return converted


def _interaction_completion_for_atomic_record(record: dict) -> dict:
    completion = deepcopy(record["completion"][0])
    if completion.get("tool_calls"):
        return completion

    category = str(record["metadata"].get("category", ""))
    has_tool_history = any(message.get("role") == "tool" for message in record["prompt"])
    content = str(completion.get("content", "")).strip()
    if category == "missing_parameter" and not has_tool_history:
        return _assistant_call(
            "request_clarification",
            {
                "field": "target",
                "content": content or "请提供要跟踪的具体目标。",
            },
        )
    return _assistant_call(
        "finish_task",
        {
            "kind": "refuse" if category == "irrelevant" else "answer",
            "content": content or "设备配置已完成。",
        },
    )


def build_functiongemma_curriculum_v02(
    atomic_records: Iterable[dict], structural_records: Iterable[dict]
) -> list[dict]:
    """Build a 10-tool atomic-to-structural FunctionGemma curriculum.

    Atomic supervision teaches individual actions, preconditions, recovery, presets,
    clarification, and refusal. Structural supervision then teaches full stateful
    trajectories. Exact clarification replicas from v0.4 are removed because the
    atomic curriculum already supplies diverse clarification decisions.
    """
    schemas = tool_schemas(include_interaction=True)
    mixed: list[dict] = []
    seen_structural_hashes: set[str] = set()

    for source in atomic_records:
        record = deepcopy(source)
        prompt = list(record["prompt"])
        if prompt and prompt[0].get("role") == "system":
            prompt[0] = {"role": "system", "content": BASE_SYSTEM_PROMPT}
        else:
            prompt.insert(0, {"role": "system", "content": BASE_SYSTEM_PROMPT})
        record["prompt"] = prompt
        record["completion"] = [_interaction_completion_for_atomic_record(record)]
        record["tools"] = deepcopy(schemas)
        record["metadata"]["source_generator"] = record["metadata"].get("generator")
        record["metadata"]["generator"] = "functiongemma-curriculum-v0.2"
        record["metadata"]["curriculum_axis"] = "atomic"
        record["metadata"]["decision_type"] = "tool_call"
        mixed.append(record)

    for source in structural_records:
        metadata = source["metadata"]
        canonical_hash = str(metadata.get("canonical_hash", ""))
        if "oversample_index" in metadata or canonical_hash in seen_structural_hashes:
            continue
        seen_structural_hashes.add(canonical_hash)
        record = deepcopy(source)
        record["tools"] = deepcopy(schemas)
        record["metadata"].pop("oversample_index", None)
        record["metadata"]["source_generator"] = record["metadata"].get("generator")
        record["metadata"]["generator"] = "functiongemma-curriculum-v0.2"
        record["metadata"]["curriculum_axis"] = "structural"
        mixed.append(record)

    for record in mixed:
        canonical = json.dumps(
            {"prompt": record["prompt"], "completion": record["completion"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        record["metadata"]["canonical_hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
    return convert_sft_records_for_functiongemma(mixed)


def build_functiongemma_oec_replay_v03(
    replay_records: Iterable[dict], correction_records: Iterable[dict]
) -> list[dict]:
    """Mix unique clean trajectories with caller-verified on-policy corrections."""
    schemas = tool_schemas(include_interaction=True)
    mixed: list[dict] = []
    seen_hashes: set[str] = set()
    for axis, sources in (
        ("replay", replay_records),
        ("oec", correction_records),
    ):
        for source in sources:
            if "oversample_index" in source.get("metadata", {}):
                continue
            record = deepcopy(source)
            record["tools"] = deepcopy(schemas)
            record["metadata"].pop("oversample_index", None)
            record["metadata"]["source_generator"] = record["metadata"].get("generator")
            record["metadata"]["generator"] = "functiongemma-oec-replay-v0.3"
            record["metadata"]["curriculum_axis"] = axis
            record["metadata"].setdefault(
                "category", "rl_long_horizon" if axis == "replay" else "oec_repair"
            )
            record["metadata"].setdefault("decision_type", "tool_call")
            canonical = json.dumps(
                {"prompt": record["prompt"], "completion": record["completion"]},
                ensure_ascii=False,
                sort_keys=True,
            )
            canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical_hash in seen_hashes:
                continue
            seen_hashes.add(canonical_hash)
            record["metadata"]["canonical_hash"] = canonical_hash
            mixed.append(record)
    return convert_sft_records_for_functiongemma(mixed)


def load_verified_oec_records(
    dataset_path: Path, manifest_path: Path, audit_path: Path
) -> list[dict]:
    """Load OEC records only after binding the dataset and audit to their manifest hashes."""
    dataset_bytes = dataset_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "oec-sft-manifest-v0.1":
        raise ValueError("unsupported OEC manifest format")
    if hashlib.sha256(dataset_bytes).hexdigest() != manifest.get("dataset_sha256"):
        raise ValueError("OEC dataset SHA-256 does not match its manifest")
    if hashlib.sha256(audit_bytes).hexdigest() != manifest.get("audit_sha256"):
        raise ValueError("OEC audit SHA-256 does not match its manifest")
    if manifest.get("heldout_instruction_overlap_count") != 0:
        raise ValueError("OEC manifest reports held-out instruction overlap")
    if manifest.get("heldout_completion_entities"):
        raise ValueError("OEC manifest reports held-out completion entities")
    records = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line
    ]
    if len(records) != manifest.get("record_count"):
        raise ValueError("OEC record count does not match its manifest")
    expected_generator = manifest.get("generator")
    if not expected_generator or any(
        record.get("metadata", {}).get("generator") != expected_generator
        or not record.get("metadata", {}).get("repair_id")
        for record in records
    ):
        raise ValueError("OEC records are missing verified repair provenance")
    return records


def generate_rl_sft_records(
    tasks: Iterable[TaskSpec],
    *,
    max_records: int | None = None,
    interaction_tools: bool = False,
    clarification_oversample: int = 1,
) -> list[dict]:
    """Export every Oracle decision from train-only long-horizon trajectories."""
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("at least one RL training task is required")
    if any(task.split != "train" for task in task_list):
        raise ValueError("RL SFT curriculum accepts train-split tasks only")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if clarification_oversample < 1:
        raise ValueError("clarification_oversample must be positive")
    if clarification_oversample > 1 and not interaction_tools:
        raise ValueError("clarification oversampling requires interaction tools")
    source_generators = {task.provenance.get("generator") for task in task_list}
    if len(source_generators) != 1:
        raise ValueError("RL SFT curriculum cannot mix task generator versions")
    source_generator = source_generators.pop()
    if interaction_tools:
        curriculum_generator = (
            "rl-sft-oracle-v0.4"
            if source_generator == "rl-builtin-v0.3"
            else "rl-sft-oracle-v0.3"
        )
    else:
        curriculum_generator = (
            "rl-sft-oracle-v0.2"
            if source_generator == "rl-builtin-v0.2"
            else "rl-sft-oracle-v0.1"
        )

    schemas = tool_schemas(include_interaction=interaction_tools)
    system_prompt = BASE_SYSTEM_PROMPT if interaction_tools else LEGACY_SYSTEM_PROMPT
    records: list[dict] = []
    for task in task_list:
        trial = EvaluationHarness(HarnessConfig(max_steps=10)).run(
            task,
            OracleAgent(task, interaction_tools=interaction_tools),
        )
        if not trial.grade.success:
            raise ValueError(f"Oracle failed RL curriculum task {task.id}")
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.instruction},
        ]
        events_by_step: dict[int, list] = defaultdict(list)
        for event in trial.events:
            events_by_step[event.step].append(event)

        assistant_turn = 0
        for step, events in sorted(events_by_step.items()):
            action = next(
                (event for event in events if event.kind in {"tool_call", "message"}),
                None,
            )
            if action is None:
                continue
            if action.kind == "tool_call":
                completion = _assistant_call(
                    str(action.payload["name"]),
                    dict(action.payload.get("arguments", {})),
                    str(action.payload["id"]),
                )
                decision_type = "tool_call"
            else:
                completion = {
                    "role": "assistant",
                    "content": str(action.payload.get("content", "")),
                }
                decision_type = str(action.payload["kind"])
            canonical = json.dumps(
                {"prompt": messages, "completion": [completion]},
                ensure_ascii=False,
                sort_keys=True,
            )
            record = {
                "prompt": deepcopy(messages),
                "completion": [deepcopy(completion)],
                "tools": schemas,
                "metadata": {
                    "category": "rl_long_horizon",
                    "source_split": "train",
                    "task_id": task.id,
                    "assistant_turn": assistant_turn,
                    "trajectory_step": step,
                    "decision_type": decision_type,
                    "generator": curriculum_generator,
                    "canonical_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                },
            }
            records.append(record)
            if (
                action.kind == "tool_call"
                and action.payload["name"] == "request_clarification"
            ):
                for oversample_index in range(1, clarification_oversample):
                    replica = deepcopy(record)
                    replica["metadata"]["oversample_index"] = oversample_index
                    records.append(replica)
            messages.append(deepcopy(completion))
            for event in events:
                if event.kind == "tool_result":
                    messages.append(
                        _tool_result(
                            str(event.payload["name"]),
                            dict(event.payload),
                            str(event.payload["id"]),
                        )
                    )
                elif event.kind == "user" and not event.payload.get("via_tool"):
                    messages.append(
                        {"role": "user", "content": str(event.payload["content"])}
                    )
            assistant_turn += 1

    if max_records is None or max_records >= len(records):
        return records
    positions = [(index * len(records)) // max_records for index in range(max_records)]
    return [records[position] for position in positions]
