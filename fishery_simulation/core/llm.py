"""
LangChain-based LLM adapter supporting Ollama (primary), Anthropic, OpenAI.

Logging:
  logs/llm_calls.jsonl  — one JSON record per call (machine-readable)
  logs/llm_calls.txt    — human-readable prompt/response transcript
"""
import os
import json
import time
import datetime
from langchain_core.messages import HumanMessage


class LLMAdapter:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = None,
        log_dir: str = "logs",
    ):
        self.provider = provider
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # ── Logging setup ────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, "llm_calls.jsonl")
        self._txt_path   = os.path.join(log_dir, "llm_calls.txt")
        self._jsonl_file = open(self._jsonl_path, "a", buffering=1)
        self._txt_file   = open(self._txt_path,   "a", buffering=1)

        # ── LLM client ───────────────────────────────────────────────
        # For Ollama: num_predict is set at construction time.
        # We build a fresh ChatOllama per call in _make_client() so
        # each call gets the right token budget without passing kwargs
        # to invoke() (which the underlying ollama-python Client rejects).
        if provider == "ollama":
            from langchain_ollama import ChatOllama as _ChatOllama
            self._ChatOllama = _ChatOllama
            self.model_name  = model or "gpt-oss:20b"
            # Build a default client; replaced per-call via _make_client()
            self.client = _ChatOllama(model=self.model_name, temperature=0.7)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            self.model_name = model or "claude-sonnet-4-6"
            self.client = ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=api_key,
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            self.model_name = model or "gpt-4o"
            self.client = ChatOpenAI(
                model=self.model_name,
                openai_api_key=api_key,
            )

        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    def _make_client(self, max_tokens: int):
        """Return a client configured for the given max_tokens budget."""
        if self.provider == "ollama":
            # num_predict must be set in the constructor for ChatOllama
            return self._ChatOllama(
                model=self.model_name,
                temperature=0.7,
                num_predict=max_tokens,
            )
        # Anthropic / OpenAI: use bind() to attach max_tokens to this call
        if self.provider == "anthropic":
            return self.client.bind(max_tokens=max_tokens)
        if self.provider == "openai":
            return self.client.bind(max_tokens=max_tokens)
        return self.client

    # ── Public API ───────────────────────────────────────────────────

    def complete(self, prompt: str, max_tokens: int = 600) -> str:
        """
        Single completion with 3-retry exponential backoff.
        Logs every prompt + response to both JSONL and plain-text files.
        """
        last_exc = None
        t_start  = time.time()
        client   = self._make_client(max_tokens)

        for attempt in range(3):
            try:
                response = client.invoke([HumanMessage(content=prompt)])

                self.total_calls += 1
                elapsed = round(time.time() - t_start, 3)

                # ── Token accounting ──────────────────────────────────
                in_tok, out_tok = 0, 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    in_tok  = um.get("input_tokens",  0)
                    out_tok = um.get("output_tokens", 0)
                elif hasattr(response, "response_metadata"):
                    rm    = response.response_metadata or {}
                    usage = rm.get("usage", {}) or rm.get("token_usage", {})
                    in_tok  = usage.get("prompt_tokens",     0)
                    out_tok = usage.get("completion_tokens", 0)
                self.total_input_tokens  += in_tok
                self.total_output_tokens += out_tok

                result_text = response.content

                # ── JSONL log (metadata only — no raw text) ───────────
                self._jsonl_file.write(json.dumps({
                    "ts":            datetime.datetime.utcnow().isoformat(),
                    "call_number":   self.total_calls,
                    "provider":      self.provider,
                    "model":         self.model_name,
                    "max_tokens":    max_tokens,
                    "attempt":       attempt + 1,
                    "elapsed_s":     elapsed,
                    "input_tokens":  in_tok,
                    "output_tokens": out_tok,
                }) + "\n")

                # ── Plain-text log ────────────────────────────────────
                sep = "=" * 72
                self._txt_file.write(
                    f"\n{sep}\n"
                    f"CALL #{self.total_calls}  |  {datetime.datetime.utcnow().isoformat()}  "
                    f"|  {self.provider}/{self.model_name}  |  "
                    f"attempt {attempt+1}  |  {elapsed}s  |  "
                    f"in={in_tok} out={out_tok}\n"
                    f"{sep}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                    f"--- RESPONSE ---\n{result_text}\n"
                )

                return result_text

            except Exception as e:
                last_exc = e
                elapsed  = round(time.time() - t_start, 3)
                err_msg  = f"[ATTEMPT {attempt+1} FAILED after {elapsed}s: {e}]"

                # Log the failure to both files
                try:
                    self._jsonl_file.write(json.dumps({
                        "ts":          datetime.datetime.utcnow().isoformat(),
                        "call_number": self.total_calls,
                        "provider":    self.provider,
                        "model":       self.model_name,
                        "attempt":     attempt + 1,
                        "elapsed_s":   elapsed,
                        "error":       str(e),
                    }) + "\n")
                    self._txt_file.write(
                        f"\n{'='*72}\n"
                        f"CALL (failed)  |  {datetime.datetime.utcnow().isoformat()}  "
                        f"|  {self.provider}/{self.model_name}\n"
                        f"{'='*72}\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- ERROR ---\n{err_msg}\n"
                    )
                except Exception:
                    pass

                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM call failed after 3 attempts: {last_exc}")

    def log_parsed(self, label: str, parsed: dict):
        """
        Append parsed output to the txt log immediately after a complete() call.
        Call this from the agent after _parse_cot() / reflection parsing.
        Not written to JSONL.
        """
        lines = [f"--- PARSED ({label}) ---"]
        for k, v in parsed.items():
            # Truncate long strings for readability
            val = str(v)
            if len(val) > 120:
                val = val[:117] + "..."
            lines.append(f"  {k}: {val}")
        self._txt_file.write("\n".join(lines) + "\n")

    def close(self):
        """Flush and close both log files."""
        for f in (self._jsonl_file, self._txt_file):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def usage_summary(self) -> dict:
        return {
            "total_calls":         self.total_calls,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
