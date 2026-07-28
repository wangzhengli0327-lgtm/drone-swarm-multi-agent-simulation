from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class CompatibleAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str = "https://apihub.agnes-ai.com/v1"
    model: str = "agnes-2.0-flash"
    timeout_seconds: int = 45
    requires_api_key: bool = True
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    max_tokens: int = 2600


class OpenAICompatibleClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.config.requires_api_key and not self.config.api_key.strip():
            raise CompatibleAPIError("当前提供商需要 API Key，但尚未配置")
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": 0.2,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DroneSwarm-LLM-Experiment/1.0",
            "Connection": "close",
        }
        if self.config.api_key.strip():
            headers["Authorization"] = f"Bearer {self.config.api_key.strip()}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response_payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = CompatibleAPIError(f"模型 API HTTP {exc.code}: {detail}")
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.config.max_retries:
                    raise last_error from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else self.config.retry_backoff_seconds * (attempt + 1)
                time.sleep(min(delay, 12.0))
            except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionResetError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise CompatibleAPIError(
                        f"模型 API 请求在 {attempt + 1} 次尝试后失败：{exc}"
                    ) from exc
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if response_payload is None:
            raise CompatibleAPIError(f"模型 API 请求失败：{last_error}")

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CompatibleAPIError("模型 API 返回结构中缺少 choices[0].message.content") from exc
        return self._parse_json_content(content)

    @staticmethod
    def _parse_json_content(content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise CompatibleAPIError("模型返回内容不是文本或 JSON 对象")
        cleaned = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as original_error:
            parsed = None
            decoder = json.JSONDecoder()
            for index, character in enumerate(cleaned):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            if parsed is None:
                preview = cleaned[:220].replace("\n", " ")
                raise CompatibleAPIError(f"模型没有返回有效 JSON；响应开头：{preview}") from original_error
        if not isinstance(parsed, dict):
            raise CompatibleAPIError("模型 JSON 顶层必须是对象")
        return parsed


AgnesAPIError = CompatibleAPIError
AgnesConfig = OpenAICompatibleConfig
AgnesClient = OpenAICompatibleClient
