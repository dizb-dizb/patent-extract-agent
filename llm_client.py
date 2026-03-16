"""
统一 LLM 客户端：支持 Gemini、DeepSeek、FAST_MODEL（SiliconFlow 等）。
双模型策略：规则核心用高质量模型，普通分割用高性能模型，保证结果有效且高效。
"""

from __future__ import annotations

import os
from typing import Any, Literal

ModelTier = Literal["quality", "performance"]


def _has_llm_config() -> bool:
    """是否有可用的 LLM 配置。"""
    return bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("FAST_MODEL_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    )


def _resolve_model_and_client(tier: ModelTier) -> tuple[str, str] | None:
    """
    按 tier 解析 (model_name, provider)。
    规则核心 quality：高质量模型（如 gemini-1.5-pro、deepseek-chat）
    普通分割 performance：高性能模型（如 gemini-1.5-flash、FAST_MODEL）
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        if tier == "quality":
            model = os.environ.get("GEMINI_QUALITY_MODEL", "gemini-3.1-pro-preview").strip() or "gemini-3.1-pro-preview"
        else:
            model = os.environ.get("GEMINI_PERFORMANCE_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"
        return (model, "gemini")

    fast_url = os.environ.get("FAST_MODEL_URL", "").strip()
    fast_key = os.environ.get("FAST_MODEL_API_KEY", "").strip()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if tier == "quality":
        if api_key:
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
            return (model, "deepseek")
        if fast_url and fast_key:
            model = os.environ.get("FAST_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct"
            return (model, "fast")
    else:
        if fast_url and fast_key:
            model = os.environ.get("FAST_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct"
            return (model, "fast")
        if api_key:
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
            return (model, "deepseek")
    return None


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0,
    max_tokens: int = 1024,
    tier: ModelTier = "performance",
) -> str | None:
    """
    统一 chat 接口。messages 格式: [{"role": "system"|"user"|"assistant", "content": "..."}]
    tier: "quality" 规则核心（高质量模型），"performance" 普通分割（高性能模型）
    返回最后一条 assistant 回复的 content，失败返回 None。
    """
    resolved = _resolve_model_and_client(tier)
    if not resolved:
        return None

    model_name, provider = resolved
    system_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[模型回复]\n{content}\n[继续]")

    system_content = "\n\n".join(system_parts) if system_parts else ""
    user_content = "\n\n".join(user_parts) if user_parts else ""

    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "").strip())
            kwargs: dict[str, Any] = {}
            if system_content:
                kwargs["system_instruction"] = system_content
            model = genai.GenerativeModel(model_name, **kwargs)
            gen_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            try:
                from google.generativeai.types import GenerationConfig
                gen_cfg = GenerationConfig(**gen_config)
            except (ImportError, TypeError):
                gen_cfg = gen_config
            response = model.generate_content(user_content, generation_config=gen_cfg)
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass
        # Gemini 失败时自动切换 DeepSeek
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
                msgs = []
                if system_content:
                    msgs.append({"role": "system", "content": system_content})
                msgs.append({"role": "user", "content": user_content})
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if resp.choices:
                    c = resp.choices[0].message.content
                    if c:
                        return c.strip()
            except Exception:
                pass
        return None

    if provider == "fast":
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("FAST_MODEL_API_KEY", "").strip(),
                base_url=os.environ.get("FAST_MODEL_URL", "").strip().rstrip("/"),
            )
            msgs = []
            if system_content:
                msgs.append({"role": "system", "content": system_content})
            msgs.append({"role": "user", "content": user_content})
            resp = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if resp.choices:
                c = resp.choices[0].message.content
                if c:
                    return c.strip()
        except Exception:
            pass
        return None

    if provider == "deepseek":
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
                base_url="https://api.deepseek.com",
            )
            msgs = []
            if system_content:
                msgs.append({"role": "system", "content": system_content})
            msgs.append({"role": "user", "content": user_content})
            resp = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if resp.choices:
                c = resp.choices[0].message.content
                if c:
                    return c.strip()
        except Exception:
            pass

    return None


def get_client_and_model_for_openai_style() -> tuple[Any, str] | None:
    """
    返回 (OpenAI client, model_name) 供需要直接调用 client.chat.completions.create 的代码使用。
    若使用 Gemini，则返回 None（调用方应改用 chat_completion）。
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        return None  # 使用 Gemini 时需走 chat_completion

    fast_url = os.environ.get("FAST_MODEL_URL", "").strip()
    fast_key = os.environ.get("FAST_MODEL_API_KEY", "").strip()
    if fast_url and fast_key:
        from openai import OpenAI
        return (
            OpenAI(api_key=fast_key, base_url=fast_url.rstrip("/")),
            os.environ.get("FAST_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct",
        )

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if api_key:
        from openai import OpenAI
        return (
            OpenAI(api_key=api_key, base_url="https://api.deepseek.com"),
            os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
        )
    return None
