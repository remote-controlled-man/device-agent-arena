from __future__ import annotations

from enum import StrEnum
from statistics import fmean

from device_agent_arena.rl.rewards import RewardTrace


class CreditStrategy(StrEnum):
    TRAJECTORY = "trajectory"
    TURN = "turn"


class CreditAssigner:
    def __init__(self, gamma: float = 0.95) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        self.gamma = gamma

    def assign(
        self,
        traces: tuple[RewardTrace, ...],
        *,
        strategy: CreditStrategy,
    ) -> tuple[tuple[float, ...], ...]:
        if not traces:
            raise ValueError("at least one reward trace is required")
        if any(not trace.steps for trace in traces):
            raise ValueError("every reward trace must contain at least one step")
        if strategy is CreditStrategy.TRAJECTORY:
            baseline = fmean(trace.total for trace in traces)
            return tuple(
                tuple(round(trace.total - baseline, 6) for _ in trace.steps) for trace in traces
            )
        if strategy is not CreditStrategy.TURN:
            raise ValueError(f"unsupported credit strategy: {strategy}")

        returns = tuple(self._discounted_returns(trace) for trace in traces)
        baselines = tuple(
            fmean(values[turn] for values in returns if turn < len(values))
            for turn in range(max(map(len, returns)))
        )
        return tuple(
            tuple(round(value - baselines[turn], 6) for turn, value in enumerate(values))
            for values in returns
        )

    def _discounted_returns(self, trace: RewardTrace) -> tuple[float, ...]:
        returns = [0.0] * len(trace.steps)
        future = 0.0
        for index in range(len(trace.steps) - 1, -1, -1):
            future = trace.steps[index].total + self.gamma * future
            returns[index] = future
        return tuple(returns)
