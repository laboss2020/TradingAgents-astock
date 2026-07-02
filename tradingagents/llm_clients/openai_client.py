import os
from typing import Any, Optional

import requests

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling. ``with_structured_output`` defaults
    to function-calling so the Responses-API parse path is avoided
    (langchain-openai's parse path emits noisy
    PydanticSerializationUnexpectedValue warnings per call without
    affecting correctness).

    Provider-specific quirks (e.g. DeepSeek's thinking mode) live in
    purpose-built subclasses below so this base class stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Two quirks that don't apply to other OpenAI-compatible providers:

    1. **Thinking-mode round-trip.** When DeepSeek's thinking models return
       a response with ``reasoning_content``, that field must be echoed
       back as part of the assistant message on the next turn or the API
       fails with HTTP 400. ``_create_chat_result`` captures the field on
       receive and ``_get_request_payload`` re-attaches it on send.

    2. **deepseek-reasoner has no tool_choice.** Structured output via
       function-calling is unavailable, so we raise NotImplementedError
       and let the agent factories fall back to free-text generation
       (see ``tradingagents/agents/utils/structured.py``).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError(
                "deepseek-reasoner does not support tool_choice; structured "
                "output is unavailable. Agent factories fall back to "
                "free-text generation automatically."
            )
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "minimax": ("https://api.minimax.chat/v1", "MINIMAX_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "copilot": ("https://models.inference.ai.azure.com", "GITHUB_TOKEN"),
}


def _preflight_copilot_access(base_url: str, api_key: str, model: str) -> None:
    """Best-effort Copilot preflight check with actionable diagnostics.

    The GitHub Models OpenAI-compatible gateway should expose ``/models``.
    We query it once before first use to fail fast with clear guidance when
    token scope, org policy, or plan limits block access.

    Set ``COPILOT_PREFLIGHT=0`` to skip this check.
    """
    if os.getenv("COPILOT_PREFLIGHT", "1").strip() in {"0", "false", "False"}:
        return

    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=8)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Copilot 预检失败：无法连接 GitHub Models 网关。"
            "请检查网络/代理设置，或设 COPILOT_PREFLIGHT=0 临时跳过预检。"
            f" 原始错误：{exc}"
        ) from exc

    if resp.status_code == 200:
        # Optional model existence hint (do not hard fail if response shape changes).
        try:
            payload = resp.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(models, list):
                ids = {
                    item.get("id")
                    for item in models
                    if isinstance(item, dict) and item.get("id")
                }
                if ids and model not in ids:
                    raise RuntimeError(
                        "Copilot 预检通过，但当前模型不在该账号可用列表中："
                        f"{model}。请切换为 Copilot 计划支持的模型，"
                        "或确认组织是否允许该模型发布方。"
                    )
        except ValueError:
            # Non-JSON response: ignore and proceed.
            pass
        return

    if resp.status_code in {401, 403}:
        raise RuntimeError(
            "Copilot 预检失败：鉴权或权限不足。"
            "请确认 GITHUB_TOKEN 有效且具备 models:read；"
            "若为 Business/Enterprise，请确认组织已启用 GitHub Models 并允许对应模型发布方。"
        )

    if resp.status_code == 429:
        raise RuntimeError(
            "Copilot 预检失败：已触发速率/额度限制（429）。"
            "请稍后重试，或升级/调整 Copilot 计划与配额策略。"
        )

    raise RuntimeError(
        "Copilot 预检失败：GitHub Models 网关返回异常状态码 "
        f"{resp.status_code}。响应摘要：{resp.text[:300]}"
    )


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_CONFIG:
            default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = self.base_url or default_base
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
                elif "api_key" not in self.kwargs:
                    # Without this, ChatOpenAI fails downstream with a confusing
                    # "OPENAI_API_KEY must be set" — but deepseek/qwen/glm/minimax
                    # each need their OWN env var. Name the exact one (#42).
                    raise RuntimeError(
                        f"未找到 {self.provider} 的 API Key。请在 .env 文件或环境变量中设置 "
                        f"`{api_key_env}`（例如 `{api_key_env}=你的key`），设置后重启程序。"
                        f"注意：{self.provider} 用的是 {api_key_env}，不是 OPENAI_API_KEY。"
                    )
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        if self.provider == "copilot":
            _preflight_copilot_access(
                base_url=llm_kwargs["base_url"],
                api_key=llm_kwargs["api_key"],
                model=self.model,
            )

        # DeepSeek's thinking-mode quirks live in their own subclass so the
        # base NormalizedChatOpenAI stays free of provider-specific branches.
        chat_cls = DeepSeekChatOpenAI if self.provider == "deepseek" else NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
