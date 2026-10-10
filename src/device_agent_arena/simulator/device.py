from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from random import Random

from device_agent_arena.domain.models import DeviceState, FaultSpec, ToolCall, ToolResult
from device_agent_arena.simulator.tools import TOOL_NAMES, validate_tool_arguments


class DeviceSimulator:
    def __init__(
        self,
        initial_state: DeviceState,
        faults: Iterable[FaultSpec] = (),
        *,
        seed: int = 0,
    ) -> None:
        self.state = initial_state
        self._faults = tuple(faults)
        self._invocations: Counter[str] = Counter()
        self._rng = Random(seed)

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in TOOL_NAMES:
            return ToolResult(False, "unknown_tool", f"Unknown tool: {call.name}")
        validation_error = validate_tool_arguments(call.name, call.arguments)
        if validation_error is not None:
            return ToolResult(False, "invalid_arguments", validation_error)

        self._invocations[call.name] += 1
        for fault in self._faults:
            if (
                fault.tool == call.name
                and fault.invocation == self._invocations[call.name]
                and self._rng.random() < fault.probability
            ):
                return ToolResult(False, fault.code, f"injected fault for {call.name}")

        if call.name == "get_device_state":
            return ToolResult(True, "ok", "device state", data=asdict(self.state))

        if call.name == "connect_device":
            if self.state.connected:
                return ToolResult(True, "already_satisfied", "device already connected")
            self.state.connected = True
            return ToolResult(True, "ok", "device connected", {"connected": True})

        if call.name == "set_mode":
            mode = call.arguments.get("mode")
            if mode not in {"normal", "presentation", "follow"}:
                return ToolResult(False, "invalid_arguments", "invalid device mode")
            if not self.state.connected:
                return _failed_precondition("connected", False, "device is not connected")
            if mode == "follow" and not self.state.calibrated:
                return _failed_precondition("calibrated", False, "device is not calibrated")
            self.state.mode = mode
            return ToolResult(True, "ok", f"mode set to {mode}", {"mode": mode})

        if call.name == "save_preset":
            if not self.state.connected:
                return _failed_precondition("connected", False, "device is not connected")
            name = call.arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                return ToolResult(False, "invalid_arguments", "preset name is required")
            canonical_name = name.strip()
            self.state.presets[canonical_name] = {
                "orientation": self.state.orientation,
                "tracking": self.state.tracking,
                "mode": self.state.mode,
                "target": self.state.target,
            }
            return ToolResult(True, "ok", f"preset saved: {canonical_name}")

        if call.name == "recall_preset":
            if not self.state.connected:
                return _failed_precondition("connected", False, "device is not connected")
            name = call.arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                return ToolResult(False, "invalid_arguments", "preset name is required")
            canonical_name = name.strip()
            snapshot = self.state.presets.get(canonical_name)
            if snapshot is None:
                return ToolResult(False, "not_found", f"preset not found: {canonical_name}")
            needs_calibration = (
                snapshot.get("mode") == "follow"
                or snapshot.get("tracking", "off") != "off"
            )
            if needs_calibration and not self.state.calibrated:
                return ToolResult(
                    False,
                    "precondition_failed",
                    "active preset requires device calibration",
                    details={"fact": "calibrated", "actual": False},
                )
            for field, value in snapshot.items():
                setattr(self.state, field, value)
            return ToolResult(True, "ok", f"preset recalled: {canonical_name}", dict(snapshot))

        if call.name == "calibrate_device":
            if not self.state.connected:
                return _failed_precondition("connected", False, "device is not connected")
            self.state.calibrated = True
            return ToolResult(True, "ok", "device calibrated", {"calibrated": True})

        if call.name == "set_tracking":
            kind = call.arguments.get("kind")
            target = call.arguments.get("target")
            if kind not in {"off", "person", "object"}:
                return ToolResult(False, "invalid_arguments", "invalid tracking kind")
            if not self.state.connected:
                return _failed_precondition("connected", False, "device is not connected")
            if kind != "off" and not self.state.calibrated:
                return _failed_precondition("calibrated", False, "device is not calibrated")
            if kind != "off" and (not isinstance(target, str) or not target.strip()):
                return ToolResult(False, "invalid_arguments", "tracking target is required")
            self.state.tracking = kind
            self.state.target = None if kind == "off" else target.strip()
            return ToolResult(
                True,
                "ok",
                f"tracking set to {kind}",
                {"tracking": self.state.tracking, "target": self.state.target},
            )

        if call.name != "set_orientation":
            raise AssertionError(f"unhandled validated tool: {call.name}")
        orientation = call.arguments.get("orientation")
        if orientation not in {"portrait", "landscape"}:
            return ToolResult(
                False,
                "invalid_arguments",
                "orientation must be 'portrait' or 'landscape'",
            )
        if not self.state.connected:
            return _failed_precondition("connected", False, "device is not connected")

        self.state.orientation = orientation
        return ToolResult(
            True,
            "ok",
            f"orientation set to {orientation}",
            {"orientation": orientation},
        )


def _failed_precondition(fact: str, actual: object, message: str) -> ToolResult:
    return ToolResult(
        False,
        "precondition_failed",
        message,
        details={"fact": fact, "actual": actual},
    )
