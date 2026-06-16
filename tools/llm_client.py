"""
Unified LLM client: Claude (primary) → OpenAI (fallback) → Ollama (offline).
Supports streaming, exponential backoff, and cost tracking.
"""

import os
import time
import json
from typing import Optional, Generator, Any
import logging

COST_PER_1K = {
    "claude-opus-4-8":     {"input": 0.015,  "output": 0.075},
    "claude-sonnet-4-6":   {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "gpt-4o":              {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015,"output": 0.0006},
    "llama3":              {"input": 0.0,    "output": 0.0},
    "mistral":             {"input": 0.0,    "output": 0.0},
}

PROVIDER_PRIORITY = ["claude", "openai", "ollama"]

CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL",  "claude-opus-4-8")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama3")
OLLAMA_BASE   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIVACY_MODE  = os.getenv("PRIVACY_MODE",  "false").lower() == "true"

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """Unified LLM client with automatic provider fallback."""

    def __init__(self):
        self.last_provider_used = "none"
        self._memory: Optional[Any] = None

    def _build_chain(self):
        if PRIVACY_MODE:
            return ["ollama"]
        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append("claude")
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        chain.append("ollama")
        return chain

    def complete(self, prompt: str, max_tokens: int = 1024, system: str = "") -> str:
        """Complete a prompt. Tries providers in order, raises on all failure."""
        chain = self._build_chain()
        last_error = None
        for provider in chain:
            try:
                result = self._call_with_retry(provider, prompt, max_tokens, system)
                self.last_provider_used = provider
                return result
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def complete_sync(self, prompt: str, max_tokens: int = 1024, system: str = "") -> str:
        """Synchronous complete (wraps complete — already sync)."""
        return self.complete(prompt, max_tokens, system)

    def stream(self, prompt: str, max_tokens: int = 1024) -> Generator[str, None, None]:
        """Stream response tokens. Falls back to complete() if streaming unavailable."""
        chain = self._build_chain()
        for provider in chain:
            try:
                yield from self._stream_provider(provider, prompt, max_tokens)
                self.last_provider_used = provider
                return
            except Exception:
                continue
        yield self.complete(prompt, max_tokens)

    def _call_with_retry(
        self, provider: str, prompt: str, max_tokens: int, system: str, retries: int = 3
    ) -> str:
        delay = 1.0
        for attempt in range(retries):
            try:
                if provider == "claude":
                    return self._call_claude(prompt, max_tokens, system)
                elif provider == "openai":
                    return self._call_openai(prompt, max_tokens, system)
                elif provider == "ollama":
                    return self._call_ollama(prompt, max_tokens)
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise
        raise RuntimeError("Retry exhausted")

    def _call_claude(self, prompt: str, max_tokens: int, system: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text = response.content[0].text
        self._log_cost("claude", CLAUDE_MODEL,
                       response.usage.input_tokens, response.usage.output_tokens, prompt)
        return text

    def _call_openai(self, prompt: str, max_tokens: int, system: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, max_tokens=max_tokens
        )
        text = response.choices[0].message.content
        usage = response.usage
        self._log_cost("openai", OPENAI_MODEL,
                       usage.prompt_tokens, usage.completion_tokens, prompt)
        return text

    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        import urllib.request
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    def _stream_provider(self, provider: str, prompt: str, max_tokens: int) -> Generator:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            with client.messages.stream(
                model=CLAUDE_MODEL, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            yield self._call_ollama(prompt, max_tokens)

    def _log_cost(self, provider: str, model: str, in_tok: int, out_tok: int, prompt: str):
        rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
        cost = (in_tok * rates["input"] + out_tok * rates["output"]) / 1000.0
        try:
            if self._memory is None:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from agent.memory.memory_manager import MemoryManager
                self._memory = MemoryManager()
            self._memory.log_llm_cost(
                provider=provider, model=model,
                input_tokens=in_tok, output_tokens=out_tok,
                cost_usd=cost, task="interpret",
            )
        except Exception as exc:
            logger.debug("Failed to log LLM cost: %s", exc)
