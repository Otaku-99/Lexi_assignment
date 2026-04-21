from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests
from openai import OpenAI


@dataclass
class LLMSettings:
    provider: str
    model: str
    temperature: float = 0.1


class LLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.settings.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        if self.settings.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.settings.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": self.settings.temperature},
        }
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://localhost:11434. Start Ollama and pull the selected model first."
            ) from exc

        data = response.json()
        return data.get("response", "").strip()

