from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from device_agent_arena.domain.models import TaskSpec, TrialResult
from device_agent_arena.harness.runner import Agent, EvaluationHarness, HarnessConfig
from device_agent_arena.rl.credit import CreditAssigner, CreditStrategy
from device_agent_arena.rl.rewards import RewardModel, RewardTrace

PolicyFactory = Callable[[TaskSpec, int], Agent]


@dataclass(frozen=True)
class RolloutConfig:
    num_generations: int = 2
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    credit_strategy: CreditStrategy = CreditStrategy.TRAJECTORY
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2 for grouped credit")


@dataclass(frozen=True)
class RolloutTrajectory:
    task_id: str
    generation: int
    trial: TrialResult
    reward: RewardTrace
    advantages: tuple[float, ...]
    credit_strategy: CreditStrategy
    gamma: float


class RolloutCollector:
    def __init__(
        self,
        config: RolloutConfig | None = None,
        *,
        reward_model: RewardModel | None = None,
    ) -> None:
        self.config = config or RolloutConfig()
        self.reward_model = reward_model or RewardModel()
        self.credit_assigner = CreditAssigner(gamma=self.config.gamma)

    def collect(
        self,
        policy_factory: PolicyFactory,
        tasks: Iterable[TaskSpec],
    ) -> tuple[RolloutTrajectory, ...]:
        trajectories: list[RolloutTrajectory] = []
        for task in tasks:
            trials: list[TrialResult] = []
            rewards: list[RewardTrace] = []
            for generation in range(self.config.num_generations):
                agent = policy_factory(task, generation)
                trial = EvaluationHarness(self.config.harness).run(task, agent)
                trials.append(trial)
                rewards.append(self.reward_model.score(task, trial))
            grouped_rewards = tuple(rewards)
            grouped_advantages = self.credit_assigner.assign(
                grouped_rewards,
                strategy=self.config.credit_strategy,
            )
            trajectories.extend(
                RolloutTrajectory(
                    task_id=task.id,
                    generation=generation,
                    trial=trials[generation],
                    reward=grouped_rewards[generation],
                    advantages=grouped_advantages[generation],
                    credit_strategy=self.config.credit_strategy,
                    gamma=self.config.gamma,
                )
                for generation in range(self.config.num_generations)
            )
        return tuple(trajectories)
