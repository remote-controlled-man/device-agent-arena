from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROLES = {"inspect", "mutate", "repair", "clarify", "terminal"}


def _schema_sha256(tool: dict[str, Any]) -> str:
    canonical = json.dumps(
        tool,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reject_repair_cycles(operations: list[ToolContract]) -> None:
    graph = {
        operation.operation_id: tuple(
            str(precondition[branch]["operation_id"])
            for precondition in operation.preconditions
            for branch in ("on_unknown", "on_false")
            if precondition.get(branch, {}).get("operation_id") is not None
        )
        for operation in operations
    }
    visited: set[str] = set()
    active: set[str] = set()
    stack: list[str] = []

    def visit(operation_id: str) -> None:
        if operation_id in visited:
            return
        if operation_id in active:
            cycle_start = stack.index(operation_id)
            cycle = [*stack[cycle_start:], operation_id]
            raise ValueError(f"repair dependency cycle: {' -> '.join(cycle)}")
        active.add(operation_id)
        stack.append(operation_id)
        for repair_id in graph[operation_id]:
            visit(repair_id)
        stack.pop()
        active.remove(operation_id)
        visited.add(operation_id)

    for operation in operations:
        visit(operation.operation_id)


def _validate_operation(operation: dict[str, Any], operation_id: str) -> None:
    if operation.get("role") not in _ROLES:
        raise ValueError(f"{operation_id} has unsupported role: {operation.get('role')}")
    tool = operation.get("tool")
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise ValueError(f"{operation_id} tool must be a function schema")
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError(f"{operation_id} tool function name must be a string")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(f"{operation_id} tool parameters must be an object schema")

    for index, effect in enumerate(operation.get("effects", [])):
        if not isinstance(effect, dict) or len(effect) != 1:
            raise ValueError(f"{operation_id} effect {index} has unsupported expression")
        expression = effect.get("set") or effect.get("set_from_argument")
        if not isinstance(expression, dict):
            raise ValueError(f"{operation_id} effect {index} has unsupported expression")
        if not isinstance(expression.get("fact"), str):
            raise ValueError(f"{operation_id} effect {index} fact must be a string")
        if "set" in effect and "value" not in expression:
            raise ValueError(f"{operation_id} effect {index} set requires value")
        if "set_from_argument" in effect and not isinstance(
            expression.get("argument"), str
        ):
            raise ValueError(
                f"{operation_id} effect {index} set_from_argument requires argument"
            )

    for commitment in operation.get("commitments", []):
        if not isinstance(commitment, dict) or not isinstance(commitment.get("id"), str):
            raise ValueError(f"{operation_id} commitment requires a string id")
        commitment_id = str(commitment["id"])
        contains_any = commitment.get("contains_any", [])
        if not isinstance(contains_any, list) or not all(
            isinstance(marker, str) and marker for marker in contains_any
        ):
            raise ValueError(
                f"{operation_id} commitment {commitment_id} contains_any must be strings"
            )
        arguments = commitment.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(
                f"{operation_id} commitment {commitment_id} arguments must be an object"
            )
        for argument, specification in arguments.items():
            if not isinstance(specification, dict) or set(specification) != {"enum_mentions"}:
                raise ValueError(
                    f"{operation_id} commitment {commitment_id} argument {argument} "
                    "must use enum_mentions"
                )
            mentions = specification["enum_mentions"]
            if not isinstance(mentions, dict) or not mentions or not all(
                isinstance(markers, list)
                and markers
                and all(isinstance(marker, str) and marker for marker in markers)
                for markers in mentions.values()
            ):
                raise ValueError(
                    f"{operation_id} commitment {commitment_id} argument {argument} "
                    "enum_mentions must map values to marker lists"
                )

    retry = operation.get("retry", {})
    if not isinstance(retry, dict):
        raise ValueError(f"{operation_id} retry must be an object")
    max_attempts = retry.get("max_attempts", 0)
    allowed_codes = retry.get("allowed_codes", [])
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 0:
        raise ValueError(f"{operation_id} retry max_attempts must be a non-negative integer")
    if not isinstance(allowed_codes, list) or not all(
        isinstance(code, str) for code in allowed_codes
    ):
        raise ValueError(f"{operation_id} retry allowed_codes must be strings")


@dataclass(frozen=True)
class ToolContract:
    operation_id: str
    role: str
    schema_sha256: str
    tool: dict[str, Any]
    preconditions: tuple[dict[str, Any], ...]
    effects: tuple[dict[str, Any], ...]
    commitments: tuple[dict[str, Any], ...]
    idempotent: bool
    retry: dict[str, Any]

    @property
    def public_name(self) -> str:
        return str(self.tool["function"]["name"])


@dataclass(frozen=True)
class ContractCatalog:
    catalog_id: str
    format_version: str
    operations: tuple[ToolContract, ...]

    def operation(self, operation_id: str) -> ToolContract:
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        raise KeyError(operation_id)

    def operation_for_tool(self, public_name: str) -> ToolContract:
        for operation in self.operations:
            if operation.public_name == public_name:
                return operation
        raise KeyError(public_name)

    def model_tools(self) -> list[dict[str, Any]]:
        return deepcopy([operation.tool for operation in self.operations])

    def public_contracts(self) -> list[dict[str, Any]]:
        public_names = {
            operation.operation_id: operation.public_name for operation in self.operations
        }
        return [
            {
                "name": operation.public_name,
                "role": operation.role,
                "preconditions": [
                    {
                        "fact": precondition["fact"],
                        "equals": precondition.get("equals"),
                        "on_unknown": public_names[
                            str(precondition["on_unknown"]["operation_id"])
                        ],
                        "on_false": public_names[
                            str(precondition["on_false"]["operation_id"])
                        ],
                    }
                    for precondition in operation.preconditions
                ],
                "effects": deepcopy(list(operation.effects)),
                "idempotent": operation.idempotent,
                "retry": deepcopy(operation.retry),
            }
            for operation in self.operations
        ]


def rename_contract_catalog(
    catalog: ContractCatalog,
    *,
    catalog_id: str,
    public_names: dict[str, str],
    argument_names: dict[str, dict[str, str]] | None = None,
    fact_names: dict[str, str] | None = None,
) -> ContractCatalog:
    """Create a digest-rebound metamorphic Tool Surface without changing graph semantics."""
    argument_names = argument_names or {}
    fact_names = fact_names or {}
    operations: list[ToolContract] = []
    for operation in catalog.operations:
        tool = deepcopy(operation.tool)
        tool["function"]["name"] = public_names.get(
            operation.operation_id, operation.public_name
        )
        renames = argument_names.get(operation.operation_id, {})
        parameters = tool["function"]["parameters"]
        properties = dict(parameters.get("properties", {}))
        unknown = set(renames) - set(properties)
        if unknown:
            raise ValueError(
                f"{operation.operation_id} cannot rename unknown arguments: {sorted(unknown)}"
            )
        parameters["properties"] = {
            renames.get(name, name): specification
            for name, specification in properties.items()
        }
        parameters["required"] = [
            renames.get(name, name) for name in parameters.get("required", [])
        ]
        effects = deepcopy(operation.effects)
        for effect in effects:
            expression = effect.get("set") or effect.get("set_from_argument")
            old_fact = str(expression["fact"])
            expression["fact"] = fact_names.get(old_fact, old_fact)
            if "set_from_argument" in effect:
                old_name = str(effect["set_from_argument"]["argument"])
                effect["set_from_argument"]["argument"] = renames.get(old_name, old_name)
        commitments = deepcopy(operation.commitments)
        for commitment in commitments:
            commitment["arguments"] = {
                renames.get(name, name): specification
                for name, specification in commitment.get("arguments", {}).items()
            }
        preconditions = deepcopy(operation.preconditions)
        for precondition in preconditions:
            old_fact = str(precondition["fact"])
            precondition["fact"] = fact_names.get(old_fact, old_fact)
        operations.append(
            ToolContract(
                operation_id=operation.operation_id,
                role=operation.role,
                schema_sha256=_schema_sha256(tool),
                tool=tool,
                preconditions=preconditions,
                effects=effects,
                commitments=commitments,
                idempotent=operation.idempotent,
                retry=deepcopy(operation.retry),
            )
        )
    names = [operation.public_name for operation in operations]
    if len(names) != len(set(names)):
        raise ValueError("metamorphic aliases create duplicate public tool names")
    return ContractCatalog(
        catalog_id=catalog_id,
        format_version=catalog.format_version,
        operations=tuple(operations),
    )


def load_contract_catalog(path: str | Path) -> ContractCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format_version") != "1.0":
        raise ValueError(f"unsupported contract format_version: {raw.get('format_version')}")
    if not isinstance(raw.get("catalog_id"), str) or not raw["catalog_id"]:
        raise ValueError("catalog_id must be a non-empty string")
    if not isinstance(raw.get("operations"), list) or not raw["operations"]:
        raise ValueError("operations must be a non-empty list")
    operations: list[ToolContract] = []
    operation_ids: set[str] = set()
    public_names: set[str] = set()
    for operation in raw["operations"]:
        if not isinstance(operation, dict):
            raise ValueError("operation must be an object")
        operation_id = str(operation["operation_id"])
        if operation_id in operation_ids:
            raise ValueError(f"duplicate operation_id: {operation_id}")
        _validate_operation(operation, operation_id)
        tool = deepcopy(operation["tool"])
        public_name = str(tool["function"]["name"])
        if public_name in public_names:
            raise ValueError(f"duplicate public tool name: {public_name}")
        schema_sha256 = str(operation["schema_sha256"])
        if _schema_sha256(tool) != schema_sha256:
            raise ValueError(
                f"{operation_id} schema_sha256 does not match its model-visible tool"
            )
        operations.append(
            ToolContract(
                operation_id=operation_id,
                role=str(operation["role"]),
                schema_sha256=schema_sha256,
                tool=tool,
                preconditions=tuple(deepcopy(operation.get("preconditions", []))),
                effects=tuple(deepcopy(operation.get("effects", []))),
                commitments=tuple(deepcopy(operation.get("commitments", []))),
                idempotent=bool(operation.get("idempotent", False)),
                retry=deepcopy(operation.get("retry", {})),
            )
        )
        operation_ids.add(operation_id)
        public_names.add(public_name)
    for operation in operations:
        for precondition in operation.preconditions:
            for branch in ("on_unknown", "on_false"):
                target = precondition.get(branch, {}).get("operation_id")
                if target is not None and target not in operation_ids:
                    precondition_id = precondition.get("id", "<missing>")
                    raise ValueError(
                        f"{operation.operation_id} precondition {precondition_id} {branch} "
                        f"references unknown operation {target}"
                    )
    _reject_repair_cycles(operations)
    return ContractCatalog(
        catalog_id=str(raw["catalog_id"]),
        format_version=str(raw["format_version"]),
        operations=tuple(operations),
    )
