"""API-based model runner using OpenAI-compatible NVIDIA NIM endpoint."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ModelRunner:
    model_name: str
    _client: OpenAI

    @classmethod
    def load(cls, model_name: str, api_key: str, base_url: str) -> "ModelRunner":
        print(f"Connecting to {model_name} via API ...")
        client = OpenAI(base_url=base_url, api_key=api_key)
        return cls(model_name=model_name, _client=client)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content
