from __future__ import annotations

from collections import deque

from device_agent_arena.domain.models import MessageAction, TaskSpec, ToolCall
from device_agent_arena.harness.runner import AgentContext


class OracleAgent:
    """Hidden-goal policy used only to validate task solvability and graders."""

    name = "oracle-v0.2"

    def __init__(self, task: TaskSpec, *, interaction_tools: bool = False) -> None:
        self._task = task
        self._interaction_tools = interaction_tools
        self._actions = deque(self._build_actions(task, interaction_tools=interaction_tools))
        self._last_tool_call: ToolCall | None = None

    def act(self, context: AgentContext) -> ToolCall | MessageAction:
        last_result = next(
            (event for event in reversed(context.events) if event.kind == "tool_result"),
            None,
        )
        if (
            last_result is not None
            and not last_result.payload["ok"]
            and self._last_tool_call is not None
            and last_result.payload["code"] == "transient_failure"
        ):
            return self._last_tool_call

        if self._actions:
            action = self._actions.popleft()
            if isinstance(action, ToolCall):
                self._last_tool_call = action
            return action
        if self._interaction_tools:
            return ToolCall(
                "finish_task",
                {
                    "kind": self._task.goal.terminal_behavior,
                    "content": "设备配置已完成。",
                },
            )
        return MessageAction(
            kind=self._task.goal.terminal_behavior,
            content="设备配置已完成。",
        )

    @staticmethod
    def _build_actions(
        task: TaskSpec, *, interaction_tools: bool = False
    ) -> tuple[ToolCall | MessageAction, ...]:
        actions: list[ToolCall | MessageAction] = []
        for field in task.goal.required_clarifications:
            content = "请提供要跟踪的具体目标。" if field == "target" else f"请提供{field}。"
            if interaction_tools:
                actions.append(
                    ToolCall(
                        "request_clarification",
                        {"field": field, "content": content},
                    )
                )
            else:
                actions.append(MessageAction(kind="clarify", field=field, content=content))

        for tool in task.goal.required_tools:
            if tool in {"get_device_state", "connect_device", "calibrate_device"}:
                actions.append(ToolCall(tool))
            elif tool == "set_orientation":
                actions.append(
                    ToolCall(tool, {"orientation": task.goal.final_state["orientation"]})
                )
            elif tool == "set_mode":
                actions.append(ToolCall(tool, {"mode": task.goal.final_state["mode"]}))
            elif tool == "set_tracking":
                kind = task.goal.final_state["tracking"]
                arguments = {"kind": kind}
                if kind != "off":
                    arguments["target"] = task.goal.final_state["target"]
                actions.append(ToolCall(tool, arguments))
            elif tool == "save_preset":
                expected_presets = task.goal.final_state.get("presets", {})
                if len(expected_presets) != 1:
                    raise ValueError(f"{task.id}: save-preset goal must name one preset")
                name = next(iter(expected_presets))
                actions.append(ToolCall(tool, {"name": name}))
            elif tool == "recall_preset":
                if len(task.initial_state.presets) == 1:
                    matching_names = list(task.initial_state.presets)
                else:
                    snapshot_fields = {"orientation", "tracking", "mode", "target"}
                    matching_names = [
                        name
                        for name, snapshot in task.initial_state.presets.items()
                        if all(
                            key not in snapshot_fields or snapshot.get(key) == value
                            for key, value in task.goal.final_state.items()
                        )
                    ]
                if not matching_names:
                    raise ValueError(f"{task.id}: no preset satisfies oracle goal")
                actions.append(ToolCall(tool, {"name": matching_names[0]}))
            else:
                raise ValueError(f"{task.id}: unsupported oracle tool {tool}")
        return tuple(actions)
