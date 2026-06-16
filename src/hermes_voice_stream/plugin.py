"""VoiceStreamPlugin: hermes-voice-stream para hermes-agent."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("hermes-voice-stream")


class VoiceStreamPlugin:
    """Plugin standalone."""
    name = "hermes-voice-stream"
    kind = "standalone"
    version = "1.0.0"

    def register(self, ctx) -> None:
        """Hook de registro."""
        # Tools
        ctx.register_tool("hermes_voice_stream_status", self._tool_status)

        # Skills
        skill_path = self._skill_path()
        if skill_path.exists():
            ctx.register_skill("hermes-voice-stream", skill_path)

        log.info("hermes-voice-stream v%s registrado", self.version)

    def _skill_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "skills" / "voice-stream"

    def _tool_status(self, **_):
        return {"status": "ready", "version": self.version}
