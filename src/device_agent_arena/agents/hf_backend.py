from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from device_agent_arena.agents.model_agent import ModelResponse
from device_agent_arena.artifacts.run_io import artifact_tree_sha256
from device_agent_arena.domain.models import ToolCall

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
CHAT_BOUNDARY_PATTERN = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
FUNCTIONGEMMA_CALL_PATTERN = re.compile(
    r"<start_function_call>\s*call:([A-Za-z_]\w*)\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
FUNCTIONGEMMA_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]\w*")
FUNCTIONGEMMA_BOUNDARY_PATTERN = re.compile(
    r"<start_of_turn>(?:developer|user|model|tool)?\s*|<end_of_turn>"
    r"|<(?:start|end)_function_response>|<(?:bos|eos)>",
    re.DOTALL,
)
FUNCTIONGEMMA_CALL_ENVELOPE_PATTERN = re.compile(
    r"<start_function_call>.*?<end_function_call>", re.DOTALL
)
FUNCTIONGEMMA_DEVELOPER_PROMPT = (
    "You are a model that can do function calling with the following functions"
)


def cached_model_snapshot(
    *,
    model_id: str,
    model_revision: str | None,
    cache_root: str | Path,
) -> Path | None:
    """Resolve a complete pinned Hub snapshot without making a network request."""
    if not model_revision:
        return None
    snapshot = (
        Path(cache_root)
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / model_revision
    )
    required = ("config.json", "tokenizer_config.json")
    has_weights = (snapshot / "model.safetensors").is_file() or (
        snapshot / "model.safetensors.index.json"
    ).is_file()
    if has_weights and all((snapshot / name).is_file() for name in required):
        return snapshot.resolve()
    return None


def _tool_call_from_payload(payload: Any) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload.get("name"), str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=payload["name"], arguments=arguments)


def parse_qwen_response(text: str) -> ModelResponse:
    calls: list[ToolCall] = []
    for payload_text in TOOL_CALL_PATTERN.findall(text):
        try:
            payload = json.loads(payload_text)
            call = _tool_call_from_payload(payload)
            if call is not None:
                calls.append(call)
        except json.JSONDecodeError:
            continue

    visible_text = THINK_PATTERN.sub("", text)
    visible_text = TOOL_CALL_PATTERN.sub("", visible_text).strip()
    visible_text = CHAT_BOUNDARY_PATTERN.sub("", visible_text).strip()
    if not calls and visible_text:
        try:
            bare_call = _tool_call_from_payload(json.loads(visible_text))
        except json.JSONDecodeError:
            bare_call = None
        if bare_call is not None:
            calls.append(bare_call)
            visible_text = ""
    return ModelResponse(text=visible_text, tool_calls=tuple(calls))


def _parse_functiongemma_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return value.strip("'\"")


def _parse_functiongemma_arguments(body: str) -> dict[str, Any] | None:
    try:
        return _FunctionGemmaArgumentParser(body).parse()
    except ValueError:
        return None


class _FunctionGemmaArgumentParser:
    """Parse FunctionGemma's JSON-like argument syntax, including nested payloads."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> dict[str, Any]:
        result = self._parse_mapping(end=None)
        self._skip_space()
        if self.position != len(self.text):
            raise ValueError("unexpected trailing FunctionGemma argument content")
        return result

    def _parse_mapping(self, *, end: str | None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._skip_space()
        if end is not None and self._consume(end):
            return result
        if end is None and self.position == len(self.text):
            return result
        while True:
            name = self._identifier()
            self._skip_space()
            self._expect(":")
            self._skip_space()
            if name in result:
                raise ValueError("duplicate FunctionGemma argument")
            result[name] = self._value()
            self._skip_space()
            if end is not None and self._consume(end):
                return result
            if end is None and self.position == len(self.text):
                return result
            self._expect(",")
            self._skip_space()
            if self.position == len(self.text) or (
                end is not None and self.text.startswith(end, self.position)
            ):
                raise ValueError("trailing FunctionGemma argument separator")

    def _value(self) -> Any:
        if self.text.startswith("<escape>", self.position):
            self.position += len("<escape>")
            end = self.text.find("<escape>", self.position)
            if end < 0:
                raise ValueError("unterminated FunctionGemma escaped string")
            value = self.text[self.position : end]
            self.position = end + len("<escape>")
            return value
        if self._consume("{"):
            return self._parse_mapping(end="}")
        if self._consume("["):
            return self._parse_list()
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in ",}]":
            self.position += 1
        if start == self.position:
            raise ValueError("missing FunctionGemma argument value")
        return _parse_functiongemma_scalar(self.text[start : self.position])

    def _parse_list(self) -> list[Any]:
        values: list[Any] = []
        self._skip_space()
        if self._consume("]"):
            return values
        while True:
            values.append(self._value())
            self._skip_space()
            if self._consume("]"):
                return values
            self._expect(",")
            self._skip_space()
            if self.position == len(self.text) or self.text.startswith("]", self.position):
                raise ValueError("trailing FunctionGemma list separator")

    def _identifier(self) -> str:
        self._skip_space()
        match = FUNCTIONGEMMA_IDENTIFIER_PATTERN.match(self.text, self.position)
        if match is None:
            raise ValueError("invalid FunctionGemma argument name")
        self.position = match.end()
        return match.group(0)

    def _skip_space(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def _consume(self, token: str) -> bool:
        if self.text.startswith(token, self.position):
            self.position += len(token)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._consume(token):
            raise ValueError(f"expected {token!r} in FunctionGemma arguments")


def parse_functiongemma_response(text: str) -> ModelResponse:
    """Parse FunctionGemma's documented control-token function-call format."""
    calls: list[ToolCall] = []
    for match in FUNCTIONGEMMA_CALL_PATTERN.finditer(text):
        arguments = _parse_functiongemma_arguments(match.group(2))
        if arguments is not None:
            calls.append(ToolCall(name=match.group(1), arguments=arguments))

    visible_text = FUNCTIONGEMMA_CALL_ENVELOPE_PATTERN.sub("", text)
    visible_text = FUNCTIONGEMMA_BOUNDARY_PATTERN.sub("", visible_text).strip()
    return ModelResponse(text=visible_text, tool_calls=tuple(calls))


def normalize_functiongemma_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style history to FunctionGemma's documented developer/tool format."""
    normalized = deepcopy(messages)
    if normalized and normalized[0].get("role") in {"system", "developer"}:
        original = str(normalized[0].get("content", "")).strip()
        content = FUNCTIONGEMMA_DEVELOPER_PROMPT
        if original and original != FUNCTIONGEMMA_DEVELOPER_PROMPT:
            content = f"{content}\n\n{original}"
        normalized[0] = {"role": "developer", "content": content}
    else:
        normalized.insert(
            0,
            {"role": "developer", "content": FUNCTIONGEMMA_DEVELOPER_PROMPT},
        )

    for message in normalized:
        if message.get("role") == "tool":
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                try:
                    response = json.loads(raw_content)
                except json.JSONDecodeError:
                    response = raw_content
            else:
                response = raw_content
            name = message.get("name")
            if not isinstance(name, str) and isinstance(response, dict):
                name = response.get("name")
            if not isinstance(name, str):
                raise ValueError("FunctionGemma tool responses require a function name")
            message.clear()
            message.update(
                {
                    "role": "tool",
                    "content": {"name": name, "response": response},
                }
            )
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "FunctionGemma tool-call arguments must be valid JSON"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError("FunctionGemma tool-call arguments must decode to an object")
                function["arguments"] = parsed
    return normalized


def uses_functiongemma_protocol(model_id: str) -> bool:
    """Detect official Hub IDs or local checkpoints that declare a FunctionGemma source."""
    normalized = model_id.lower().replace("\\", "/")
    if re.fullmatch(r"(?:google/)?functiongemma-[^/]+", normalized):
        return True

    model_path = Path(model_id)
    if not model_path.is_dir():
        return False
    for filename in ("config.json", "tokenizer_config.json"):
        config_path = model_path / filename
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        declared_protocol = str(config.get("device_agent_protocol", "")).lower()
        declared_source = str(
            config.get("_name_or_path", config.get("name_or_path", ""))
        ).lower().replace("\\", "/")
        chat_template = str(config.get("chat_template", ""))
        if declared_protocol == "functiongemma":
            return True
        if re.search(r"(?:^|/)functiongemma-[^/]+(?:/|$)", declared_source):
            return True
        if "<start_function_call>" in chat_template:
            return True
    template_path = model_path / "chat_template.jinja"
    if template_path.is_file():
        try:
            template = template_path.read_text(encoding="utf-8")
        except OSError:
            template = ""
        if "<start_function_call>" in template and "<end_function_call>" in template:
            return True
    return False


def functiongemma_stop_token_ids(tokenizer: Any) -> list[int]:
    """Return the documented FunctionGemma generation stop-token set."""
    stop_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_ids.append(eos_token_id)
    response_token_id = tokenizer.convert_tokens_to_ids("<start_function_response>")
    if (
        isinstance(response_token_id, int)
        and response_token_id != getattr(tokenizer, "unk_token_id", None)
        and response_token_id not in stop_ids
    ):
        stop_ids.append(response_token_id)
    return stop_ids


class TransformersBackend:
    """Lazy optional backend for model-native local tool calling."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str | None = None,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        generation_timeout_seconds: float = 120.0,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "local model dependencies are missing; run `uv sync --extra train`"
            ) from exc

        self._torch = torch
        self.name = model_id if adapter_path is None else f"{model_id}+{Path(adapter_path).name}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_timeout_seconds = generation_timeout_seconds
        self.model_id = model_id
        self.requested_model_revision = model_revision
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.load_in_4bit = load_in_4bit
        self.is_functiongemma = uses_functiongemma_protocol(model_id)
        local_snapshot = cached_model_snapshot(
            model_id=model_id,
            model_revision=model_revision,
            cache_root=Path(".hf-cache"),
        )
        model_source: str | Path = local_snapshot or model_id
        tokenizer_source: str | Path = model_source
        if (
            not self.is_functiongemma
            and self.adapter_path is not None
            and (self.adapter_path / "tokenizer_config.json").is_file()
        ):
            tokenizer_source = self.adapter_path
        tokenizer_kwargs = (
            {} if isinstance(tokenizer_source, Path) else {"revision": model_revision}
        )
        if self.is_functiongemma:
            self.processor = AutoProcessor.from_pretrained(model_source, **tokenizer_kwargs)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
            self.processor = self.tokenizer

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dtype": "auto",
            "revision": model_revision,
        }
        if load_in_4bit:
            # NF4 is the documented training/inference representation for 4-bit normally
            # distributed model weights.
            # Source: https://huggingface.co/docs/transformers/quantization/bitsandbytes#normal-float-4-nf4
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if isinstance(model_source, Path):
            model_kwargs.pop("revision", None)
            model_kwargs["local_files_only"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or getattr(
            self.tokenizer, "init_kwargs", {}
        ).get("_commit_hash") or (model_revision if local_snapshot is not None else None)

        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load a LoRA adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

    def provenance(self) -> dict[str, Any]:
        packages = {}
        for package in ("torch", "transformers", "peft", "bitsandbytes"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        adapter_model = (
            self.adapter_path / "adapter_model.safetensors" if self.adapter_path else None
        )
        local_model = Path(self.model_id)
        return {
            "model_id": self.model_id,
            "requested_model_revision": self.requested_model_revision,
            "model_revision": self.model_revision,
            "model_artifact_sha256": (
                artifact_tree_sha256(local_model) if local_model.exists() else None
            ),
            "chat_template_sha256": (
                    hashlib.sha256(self.processor.chat_template.encode("utf-8")).hexdigest()
                    if getattr(self.processor, "chat_template", None)
                else None
            ),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "adapter_sha256": (
                hashlib.sha256(adapter_model.read_bytes()).hexdigest()
                if adapter_model is not None and adapter_model.is_file()
                else None
            ),
            "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "generation_timeout_seconds": self.generation_timeout_seconds,
            "gpu_name": (
                self._torch.cuda.get_device_name(0) if self._torch.cuda.is_available() else None
            ),
            "packages": packages,
        }

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        # Use the model's canonical tool-aware chat template instead of a custom ReAct format.
        # Sources:
        # https://huggingface.co/docs/transformers/chat_templating#advanced-tool-use--function-calling
        # https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md
        if self.is_functiongemma:
            # FunctionGemma requires its exact developer trigger and model-specific
            # tool control tokens. Sources:
            # https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
            # https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma
            inputs = self.processor.apply_chat_template(
                normalize_functiongemma_messages(messages),
                tools=tools,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "max_time": self.generation_timeout_seconds,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        if self.is_functiongemma:
            stop_ids = functiongemma_stop_token_ids(self.tokenizer)
            if stop_ids:
                generation_kwargs["eos_token_id"] = stop_ids
                generation_kwargs["pad_token_id"] = stop_ids[0]

        cuda_available = self._torch.cuda.is_available()
        if cuda_available:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        with self._torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        if cuda_available:
            self._torch.cuda.synchronize()
        generation_ms = (perf_counter() - started) * 1000
        generated = output[0, inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        parsed = (
            parse_functiongemma_response(text)
            if self.is_functiongemma
            else parse_qwen_response(text)
        )
        return ModelResponse(
            text=parsed.text,
            tool_calls=parsed.tool_calls,
            raw_text=text,
            usage={
                "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(generated.shape[0]),
                "generation_ms": round(generation_ms, 3),
                "peak_vram_mb": (
                    round(self._torch.cuda.max_memory_allocated() / 1024**2, 3)
                    if cuda_available
                    else 0.0
                ),
            },
        )
