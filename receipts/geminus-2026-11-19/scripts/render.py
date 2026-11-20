"""Render the GGUF-embedded chat template the way HF `apply_chat_template` does (sandboxed jinja2)."""
from __future__ import annotations
import datetime
from gguf import GGUFReader
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _field_str(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")


class ChatTemplate:
    def __init__(self, gguf_path: str):
        r = GGUFReader(gguf_path, "r")
        self.source = _field_str(r, "tokenizer.chat_template")
        if not self.source:
            raise RuntimeError("GGUF has no tokenizer.chat_template")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise RuntimeError(f"chat template raised: {msg}")

        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        self._tpl = env.from_string(self.source)

    def render_user(self, user_prompt: str, enable_thinking: bool = False) -> str:
        return self._tpl.render(
            messages=[{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=None,
        )
