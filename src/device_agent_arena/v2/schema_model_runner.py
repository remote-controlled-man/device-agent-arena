from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from device_agent_arena.v1.models import V1TaskSpec
from device_agent_arena.v1.runner import _run_attested_v2_model_tasks
from device_agent_arena.v2.suite import profile_id_for_task


def run_attested_schema_model_tasks(
    *,
    tasks: Iterable[V1TaskSpec],
    backend: object,
    split: str,
    output: str | Path,
    schema_profile_id: str,
    shared_impact_guard: bool,
) -> Path:
    """Run a V2 schema candidate through the concrete local HF backend."""
    from device_agent_arena.agents.hf_backend import TransformersBackend
    from device_agent_arena.v2.coordinator import MutationCoordinatorV2
    from device_agent_arena.v2.protocol import V2ModelAgent
    from device_agent_arena.v2.schema_shift import schema_profile
    from device_agent_arena.v2.shared_guard import SharedImpactGuardV2

    if type(backend) is not TransformersBackend:
        raise TypeError("attested model runs require the concrete TransformersBackend")
    provenance = backend.provenance()
    if not provenance.get("model_artifact_sha256"):
        raise ValueError("attested model runs require a local checkpoint-tree hash")

    def agent_factory(task: V1TaskSpec):
        profile_id = (
            profile_id_for_task(task)
            if schema_profile_id == "mixed"
            else schema_profile_id
        )
        policy = V2ModelAgent(backend, contract_transform=schema_profile(profile_id))
        if shared_impact_guard:
            policy = SharedImpactGuardV2(policy)
        return MutationCoordinatorV2(policy)

    return _run_attested_v2_model_tasks(
        tasks=tasks,
        agent_factory=agent_factory,
        agent_id=f"v2-model:{backend.name}",
        split=split,
        output=output,
        run_metadata={
            **provenance,
            "attestation_version": "deviceops-ul-model-run-attestation-v2",
            "execution_authority": "deviceops-ul-transformers-runner-v1",
            "protocol": "deviceops-ul-v2-public",
            "runtime_managed_idempotency": True,
            "shared_impact_guard": shared_impact_guard,
            "schema_profile": schema_profile_id,
        },
    )
