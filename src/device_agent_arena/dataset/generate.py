from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from device_agent_arena.domain.models import (
    DeviceState,
    FaultSpec,
    GoalSpec,
    TaskSpec,
)

DEVICE_TOOLS = (
    "get_device_state",
    "connect_device",
    "calibrate_device",
    "set_orientation",
    "set_mode",
    "set_tracking",
    "save_preset",
    "recall_preset",
)


def _task(
    *,
    split: str,
    category: str,
    index: int,
    instruction: str,
    reference: str | None = None,
    state: DeviceState | None = None,
    goal: GoalSpec,
    faults: tuple[FaultSpec, ...] = (),
    clarifications: dict[str, str] | None = None,
    extra_categories: tuple[str, ...] = (),
) -> TaskSpec:
    return TaskSpec(
        id=f"{split}-{category}-{index + 1:03d}",
        version="0.4",
        split=split,  # type: ignore[arg-type]
        categories=(category, *extra_categories),
        seed=(20_000 if split == "test" else 2_000) + index,
        instruction=instruction,
        reference_instruction=reference or instruction,
        initial_state=state or DeviceState(connected=True, calibrated=True),
        fault_plan=faults,
        clarifications=clarifications or {},
        goal=goal,
        provenance={"template": f"{category}-lexical-holdout", "generator": "builtin-v0.4"},
    )


def _cases(
    split: str,
    count: int,
    dev: Sequence[tuple[Any, ...]],
    test: Sequence[tuple[Any, ...]],
) -> Sequence[tuple[Any, ...]]:
    selected = test if split == "test" else dev
    if len(selected) != count:
        raise ValueError(f"{split} case table has {len(selected)} rows, expected {count}")
    return selected


DIRECT_DEV = (
    ("将取景画面调整为横向", "set_orientation", {"orientation": "landscape"}),
    ("让显示内容保持纵向", "set_orientation", {"orientation": "portrait"}),
    ("开启用于汇报的展示模式", "set_mode", {"mode": "presentation"}),
    ("退出展示状态并回归普通模式", "set_mode", {"mode": "normal"}),
    ("读取并告诉我设备现在的完整状态", "get_device_state", {}),
    ("关闭镜头的持续锁定", "set_tracking", {"kind": "off"}),
    ("镜头持续跟随现场导播", "set_tracking", {"kind": "person", "target": "现场导播"}),
    ("锁定门口的访客并跟拍", "set_tracking", {"kind": "person", "target": "门口访客"}),
    ("持续对准墙边的指示牌", "set_tracking", {"kind": "object", "target": "墙边指示牌"}),
    ("追踪桌面上的奖杯", "set_tracking", {"kind": "object", "target": "桌面奖杯"}),
)

DIRECT_TEST = (
    ("将显示方向调整为横向布局", "set_orientation", {"orientation": "landscape"}),
    ("屏幕请改成竖向显示", "set_orientation", {"orientation": "portrait"}),
    ("旋转画面，让长边保持水平", "set_orientation", {"orientation": "landscape"}),
    ("将画面恢复为纵向排列", "set_orientation", {"orientation": "portrait"}),
    ("启用用于正式展示的工作模式", "set_mode", {"mode": "presentation"}),
    ("把设备状态调到演示档", "set_mode", {"mode": "presentation"}),
    ("结束演示，回到普通工作模式", "set_mode", {"mode": "normal"}),
    ("恢复默认的正常工作状态", "set_mode", {"mode": "normal"}),
    ("检查当前连接、方向和跟踪状态", "get_device_state", {}),
    ("关闭正在运行的人物追随", "set_tracking", {"kind": "off"}),
    ("锁定并跟拍台上的主持人", "set_tracking", {"kind": "person", "target": "主持人"}),
    ("追踪正在陈述的答辩人", "set_tracking", {"kind": "person", "target": "答辩人"}),
    ("让镜头持续追随走动的摄影师", "set_tracking", {"kind": "person", "target": "摄影师"}),
    ("把入口处的引导员设成跟踪目标", "set_tracking", {"kind": "person", "target": "引导员"}),
    ("镜头跟着场边裁判移动", "set_tracking", {"kind": "person", "target": "裁判"}),
    ("持续对准右侧投影幕", "set_tracking", {"kind": "object", "target": "投影幕"}),
    ("追踪桌上的产品样机", "set_tracking", {"kind": "object", "target": "产品样机"}),
    ("让镜头跟随移动机器人", "set_tracking", {"kind": "object", "target": "移动机器人"}),
    ("把展区中央的沙盘设成跟踪目标", "set_tracking", {"kind": "object", "target": "沙盘"}),
    ("锁定左侧电子屏并持续追踪", "set_tracking", {"kind": "object", "target": "电子屏"}),
)


def _direct(split: str, count: int) -> list[TaskSpec]:
    tasks = []
    for index, (instruction, tool, arguments) in enumerate(
        _cases(split, count, DIRECT_DEV, DIRECT_TEST)
    ):
        if tool == "set_tracking":
            final_state = {
                "tracking": arguments["kind"],
                "target": arguments.get("target"),
            }
            state = (
                DeviceState(
                    connected=True,
                    calibrated=True,
                    tracking="person",
                    target="现场导播" if split == "dev" else "主持人",
                )
                if arguments["kind"] == "off"
                else None
            )
        else:
            final_state = dict(arguments)
            state = None
        tasks.append(
            _task(
                split=split,
                category="direct",
                index=index,
                instruction=instruction,
                state=state,
                goal=GoalSpec(final_state=final_state, required_tools=(tool,)),
            )
        )
    return tasks


MULTI_STEP_DEV = (
    ("画面改为横向，接着启用展示状态", "landscape", "presentation"),
    ("保持纵向画面并回到普通模式", "portrait", "normal"),
    ("先横向显示，再恢复日常工作状态", "landscape", "normal"),
    ("先将画面纵置，随后打开汇报模式", "portrait", "presentation"),
    ("完成横向旋转后切入演示工作档", "landscape", "presentation"),
)

MULTI_STEP_TEST = (
    ("先把画面转为横向，再启用正式展示模式", "landscape", "presentation"),
    ("将屏幕纵向放置，然后恢复普通工作状态", "portrait", "normal"),
    ("画面横置以后退出演示状态", "landscape", "normal"),
    ("保持屏幕纵向，同时进入用于汇报的模式", "portrait", "presentation"),
    ("先旋转成长边水平，随后打开演示档", "landscape", "presentation"),
    ("把显示改成竖向，再切回正常工作档", "portrait", "normal"),
    ("横向展示画面，并确保设备处在普通模式", "landscape", "normal"),
    ("将内容纵向排布，接着启动展示状态", "portrait", "presentation"),
    ("依次完成横向旋转和演示模式切换", "landscape", "presentation"),
    ("先恢复纵向显示，再结束当前展示模式", "portrait", "normal"),
)


def _multi_step(split: str, count: int) -> list[TaskSpec]:
    return [
        _task(
            split=split,
            category="multi_step",
            index=index,
            instruction=instruction,
            goal=GoalSpec(
                final_state={"orientation": orientation, "mode": mode},
                required_tools=("set_orientation", "set_mode"),
            ),
        )
        for index, (instruction, orientation, mode) in enumerate(
            _cases(split, count, MULTI_STEP_DEV, MULTI_STEP_TEST)
        )
    ]


STATE_DEV = (
    ("设备离线时也请把显示改成横向", "orientation", "landscape"),
    ("这台尚未连接的设备需要纵向显示", "orientation", "portrait"),
    ("连接当前设备后将画面横置", "orientation", "landscape"),
    ("设备尚未校准，请最终打开自动跟随", "mode", "follow"),
    ("完成必要准备后启动跟随工作状态", "mode", "follow"),
)

STATE_TEST = (
    ("这台断开连接的设备最终要横向显示", "orientation", "landscape"),
    ("先处理未连接状态，再将屏幕调整为纵向", "orientation", "portrait"),
    ("设备当前离线，请让画面以横向呈现", "orientation", "landscape"),
    ("把尚未接入的终端设置成竖向布局", "orientation", "portrait"),
    ("即使设备没有连接，也要最终完成横向旋转", "orientation", "landscape"),
    ("当前没有校准，请准备好后进入跟随工作档", "mode", "follow"),
    ("先满足校准条件，再开启自动跟随状态", "mode", "follow"),
    ("这台未校准设备需要最终运行在跟随模式", "mode", "follow"),
    ("完成设备校准并切换到持续跟随档", "mode", "follow"),
    ("处理好校准前置条件后启用跟随功能", "mode", "follow"),
)


def _state_dependency(split: str, count: int) -> list[TaskSpec]:
    tasks = []
    for index, (instruction, field, value) in enumerate(
        _cases(split, count, STATE_DEV, STATE_TEST)
    ):
        if field == "orientation":
            state = DeviceState(connected=False)
            goal = GoalSpec(
                final_state={"connected": True, "orientation": value},
                required_tools=("connect_device", "set_orientation"),
            )
        else:
            state = DeviceState(connected=True, calibrated=False)
            goal = GoalSpec(
                final_state={"calibrated": True, "mode": value},
                required_tools=("calibrate_device", "set_mode"),
            )
        tasks.append(
            _task(
                split=split,
                category="state_dependency",
                index=index,
                instruction=instruction,
                state=state,
                goal=goal,
            )
        )
    return tasks


MISSING_DEV = (
    ("开启一个新的跟拍任务", "现场导播", "person"),
    ("请帮我锁定一个目标", "门口访客", "person"),
    ("让镜头追随某个对象", "墙边指示牌", "object"),
    ("启动持续目标跟踪", "桌面奖杯", "object"),
    ("我想让设备开始跟拍", "门口访客", "person"),
)

MISSING_TEST = (
    ("请新建一个人物或物体跟拍任务", "主持人", "person"),
    ("让镜头锁定一个目标", "答辩人", "person"),
    ("我需要开启持续追随", "摄影师", "person"),
    ("启动目标跟拍，但先确认对象", "引导员", "person"),
    ("现在开始一项镜头追随任务", "裁判", "person"),
    ("帮我持续对准某个东西", "投影幕", "object"),
    ("开启物体锁定功能", "产品样机", "object"),
    ("让镜头跟住一个移动目标", "移动机器人", "object"),
    ("开始跟拍指定展区对象", "沙盘", "object"),
    ("请启动目标锁定，目标待确认", "电子屏", "object"),
)


def _missing_parameter(split: str, count: int) -> list[TaskSpec]:
    return [
        _task(
            split=split,
            category="missing_parameter",
            index=index,
            instruction=instruction,
            goal=GoalSpec(
                final_state={"tracking": kind, "target": target},
                required_tools=("set_tracking",),
                required_clarifications=("target",),
            ),
            clarifications={"target": target},
        )
        for index, (instruction, target, kind) in enumerate(
            _cases(split, count, MISSING_DEV, MISSING_TEST)
        )
    ]


IRRELEVANT_DEV = (
    ("查一下周末的气温",),
    ("推荐一本科幻小说",),
    ("帮我写一封请假邮件",),
    ("计算三百二十乘以四十七",),
    ("讲讲量子纠缠是什么",),
)

IRRELEVANT_TEST = (
    ("帮我规划一条去机场的路线",),
    ("总结一下今天的财经新闻",),
    ("给这段文章拟一个标题",),
    ("提醒我明早七点起床",),
    ("翻译一句西班牙语",),
    ("推荐附近适合聚餐的餐厅",),
    ("算一下这笔贷款的月供",),
    ("给我生成一张生日贺卡",),
    ("解释一下什么是强化学习",),
    ("查询下周飞往深圳的机票",),
)


def _irrelevant(split: str, count: int) -> list[TaskSpec]:
    return [
        _task(
            split=split,
            category="irrelevant",
            index=index,
            instruction=instruction,
            goal=GoalSpec(forbidden_tools=DEVICE_TOOLS, terminal_behavior="refuse"),
        )
        for index, (instruction,) in enumerate(
            _cases(split, count, IRRELEVANT_DEV, IRRELEVANT_TEST)
        )
    ]


ASR_DEV = (
    ("显示改成横像", "显示改成横向", "set_orientation", {"orientation": "landscape"}),
    ("画面调成竖想", "画面调成竖向", "set_orientation", {"orientation": "portrait"}),
    ("打开展是状态", "打开展示状态", "set_mode", {"mode": "presentation"}),
    ("别再追棕了", "别再跟踪了", "set_tracking", {"kind": "off"}),
    (
        "镜头锁住墙边指是牌",
        "镜头锁住墙边指示牌",
        "set_tracking",
        {"kind": "object", "target": "墙边指示牌"},
    ),
)

ASR_TEST = (
    ("把显示改成横像布局", "把显示改成横向布局", "set_orientation", {"orientation": "landscape"}),
    ("画面转成竖想排列", "画面转成竖向排列", "set_orientation", {"orientation": "portrait"}),
    ("启用眼讲展示状态", "启用演讲展示状态", "set_mode", {"mode": "presentation"}),
    ("回到普同工作模式", "回到普通工作模式", "set_mode", {"mode": "normal"}),
    ("别再追棕当前目标了", "别再追踪当前目标了", "set_tracking", {"kind": "off"}),
    ("镜头跟住主齿人", "镜头跟住主持人", "set_tracking", {"kind": "person", "target": "主持人"}),
    (
        "一直对着投引幕跟",
        "一直对着投影幕跟",
        "set_tracking",
        {"kind": "object", "target": "投影幕"},
    ),
    ("把画面横过乃", "把画面横过来", "set_orientation", {"orientation": "landscape"}),
    ("切到演式档位", "切到演示档位", "set_mode", {"mode": "presentation"}),
    ("锁定产品养机", "锁定产品样机", "set_tracking", {"kind": "object", "target": "产品样机"}),
)


def _asr_noise(split: str, count: int) -> list[TaskSpec]:
    tasks = []
    for index, (noisy, clean, tool, arguments) in enumerate(
        _cases(split, count, ASR_DEV, ASR_TEST)
    ):
        state = (
            DeviceState(
                connected=True,
                calibrated=True,
                tracking="person",
                target="现场导播" if split == "dev" else "主持人",
            )
            if arguments.get("kind") == "off"
            else None
        )
        final_state = (
            {"tracking": arguments["kind"], "target": arguments.get("target")}
            if tool == "set_tracking"
            else dict(arguments)
        )
        tasks.append(
            _task(
                split=split,
                category="asr_noise",
                index=index,
                instruction=noisy,
                reference=clean,
                state=state,
                goal=GoalSpec(final_state=final_state, required_tools=(tool,)),
            )
        )
    return tasks


FAILURE_DEV = (
    ("横向调整若遇到临时错误请继续完成", "set_orientation", {"orientation": "landscape"}),
    ("纵向旋转失败一次也要恢复并完成", "set_orientation", {"orientation": "portrait"}),
    ("启用展示状态，出现瞬时故障时自行重试", "set_mode", {"mode": "presentation"}),
)

FAILURE_TEST = (
    ("把画面改为横向，若首次报临时错误请恢复", "set_orientation", {"orientation": "landscape"}),
    ("最终保持纵向显示，偶发失败不能中断任务", "set_orientation", {"orientation": "portrait"}),
    ("切入正式展示状态，遇到瞬时故障继续处理", "set_mode", {"mode": "presentation"}),
    ("恢复普通工作模式，第一次失败后仍需完成", "set_mode", {"mode": "normal"}),
    (
        "持续跟拍主持人，临时错误后要再次执行",
        "set_tracking",
        {"kind": "person", "target": "主持人"},
    ),
    (
        "锁定产品样机，工具偶发失败时请自行恢复",
        "set_tracking",
        {"kind": "object", "target": "产品样机"},
    ),
)


def _tool_failure(split: str, count: int) -> list[TaskSpec]:
    tasks = []
    for index, (instruction, tool, arguments) in enumerate(
        _cases(split, count, FAILURE_DEV, FAILURE_TEST)
    ):
        final_state = (
            {"tracking": arguments["kind"], "target": arguments.get("target")}
            if tool == "set_tracking"
            else dict(arguments)
        )
        tasks.append(
            _task(
                split=split,
                category="tool_failure",
                index=index,
                instruction=instruction,
                goal=GoalSpec(final_state=final_state, required_tools=(tool,)),
                faults=(FaultSpec(tool=tool, invocation=1),),
            )
        )
    return tasks


PRESET_DEV = (
    ("晨会", "把当前配置保存为晨会方案", "save_preset"),
    ("录课", "恢复录课方案", "recall_preset"),
)

PRESET_TEST = (
    ("路演", "把当前设备设置存成路演方案", "save_preset"),
    ("答辩", "将眼下这套配置保存为答辩预设", "save_preset"),
    ("演播", "恢复演播场景", "recall_preset"),
    ("巡展", "切换到巡展预设", "recall_preset"),
)


def _preset(split: str, count: int) -> list[TaskSpec]:
    tasks = []
    for index, (name, instruction, action) in enumerate(
        _cases(split, count, PRESET_DEV, PRESET_TEST)
    ):
        snapshot = {
            "orientation": "landscape" if index % 2 == 0 else "portrait",
            "tracking": "off",
            "mode": "presentation" if index % 2 == 0 else "normal",
            "target": None,
        }
        if action == "save_preset":
            state = DeviceState(
                connected=True,
                calibrated=True,
                orientation=snapshot["orientation"],
                tracking=snapshot["tracking"],
                mode=snapshot["mode"],
                target=snapshot["target"],
            )
            final_state = {"presets": {name: snapshot}}
        else:
            state = DeviceState(connected=True, calibrated=True, presets={name: snapshot})
            final_state = snapshot
        tasks.append(
            _task(
                split=split,
                category="preset",
                index=index,
                instruction=instruction,
                state=state,
                goal=GoalSpec(final_state=final_state, required_tools=(action,)),
            )
        )
    return tasks


GENERATORS: tuple[tuple[str, int, int, Callable[[str, int], list[TaskSpec]]], ...] = (
    ("direct", 10, 20, _direct),
    ("multi_step", 5, 10, _multi_step),
    ("state_dependency", 5, 10, _state_dependency),
    ("missing_parameter", 5, 10, _missing_parameter),
    ("irrelevant", 5, 10, _irrelevant),
    ("asr_noise", 5, 10, _asr_noise),
    ("tool_failure", 3, 6, _tool_failure),
    ("preset", 2, 4, _preset),
)


def generate_evaluation_tasks() -> tuple[TaskSpec, ...]:
    tasks: list[TaskSpec] = []
    for _category, dev_count, test_count, generator in GENERATORS:
        tasks.extend(generator("dev", dev_count))
        tasks.extend(generator("test", test_count))
    return tuple(tasks)


CLARIFICATION_HOLDOUT_DEV = (
    ("开启跟拍，不过我还没决定拍谁", "会场协调员", "person"),
    ("启动人物锁定，具体是哪位稍后告诉你", "礼仪人员", "person"),
    ("让镜头追着一个人走，先问清楚对象", "排练导演", "person"),
    ("创建一项人物追随任务，目标尚未填写", "设备操作员", "person"),
    ("我需要自动跟拍某位人员", "演出领队", "person"),
    ("打开物体跟踪，先确认具体东西", "银色航空箱", "object"),
    ("让镜头锁住一件物品，但对象还没给", "透明展示盒", "object"),
    ("新建物体追踪任务，目标参数待补", "移动签到台", "object"),
    ("开始对准某个设备，先别自行选择", "绿色控制柜", "object"),
    ("启用持续物体锁定，我随后提供名称", "升降展板", "object"),
)

CLARIFICATION_HOLDOUT_TEST = (
    ("我要开始跟拍，但现在还没说跟谁", "主讲嘉宾", "person"),
    ("开启人物追随功能，目标人员待确认", "会务负责人", "person"),
    ("让镜头持续跟着某个人，请先询问对象", "舞台监督", "person"),
    ("创建人物锁定任务，不要默认选择任何人", "现场记者", "person"),
    ("启动人像跟拍，具体人物信息还缺着", "签到人员", "person"),
    ("准备跟随一位人员，但我稍后才会说明是谁", "参赛领队", "person"),
    ("进入人物跟踪流程，第一步先确认目标", "展厅讲解员", "person"),
    ("镜头需要追随某位工作人员，名字尚未提供", "评审专家", "person"),
    ("新开一个人物跟拍任务，目标字段为空", "裁判助理", "person"),
    ("让设备跟住一位表演者，先问我要跟谁", "领舞者", "person"),
    ("开始物体跟踪，不过还没有指定物品", "旋转展台", "object"),
    ("打开物体锁定功能，目标名称等待补充", "激光雷达模型", "object"),
    ("让镜头持续对准某件东西，先确认是哪件", "蓝色路标", "object"),
    ("创建物品追随任务，不能自行猜测目标", "移动货架", "object"),
    ("启动设备目标跟拍，但目标参数尚未填写", "巡展无人车", "object"),
    ("准备锁定一个物体，我之后再告诉你名称", "白色手提箱", "object"),
    ("进入物体追踪流程，先收集具体对象", "演示终端", "object"),
    ("镜头要跟着某件移动物品，当前对象未知", "红色气球", "object"),
    ("新建物体跟拍任务，目标字段暂时为空", "样品箱", "object"),
    ("让设备对准某台装置，请先询问目标", "大型机械臂", "object"),
)


def generate_clarification_evaluation_tasks() -> tuple[TaskSpec, ...]:
    """Return a sealed clarification-only suite that is disjoint from SFT entities."""

    tasks: list[TaskSpec] = []
    for split, cases in (
        ("dev", CLARIFICATION_HOLDOUT_DEV),
        ("test", CLARIFICATION_HOLDOUT_TEST),
    ):
        for index, (instruction, target, kind) in enumerate(cases):
            tasks.append(
                TaskSpec(
                    id=f"{split}-missing_parameter-{index + 1:03d}",
                    version="0.1",
                    split=split,  # type: ignore[arg-type]
                    categories=("missing_parameter",),
                    seed=(31_000 if split == "test" else 3_100) + index,
                    instruction=instruction,
                    reference_instruction=instruction,
                    initial_state=DeviceState(connected=True, calibrated=True),
                    fault_plan=(),
                    clarifications={"target": target},
                    goal=GoalSpec(
                        final_state={"tracking": kind, "target": target},
                        required_tools=("set_tracking",),
                        required_clarifications=("target",),
                    ),
                    provenance={
                        "template": "clarification-lexical-holdout",
                        "generator": "clarification-builtin-v0.1",
                    },
                )
            )
    return tuple(tasks)
