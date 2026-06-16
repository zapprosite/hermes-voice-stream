# agent/voice_state.py
"""Voice state machine for the Hermes LiveKit agent.

State contract:
  - Before wake:       IDLE or IDLE (armed_waiting_for_wake in heartbeat)
  - During listening:  LISTENING
  - During STT:        FINAL_TRANSCRIBING
  - During LLM:        ROUTING / THINKING_LOCAL / THINKING_FALLBACK
  - During TTS:        SPEAKING
  - After turn done:   IDLE  ← ALWAYS, on every exit path
  - Barge-in:         INTERRUPTED → IDLE (via finalize_turn)
  - Cancelled:        CANCELLED → IDLE (via finalize_turn)
  - Error recovery:   ERROR_RECOVERY → IDLE (via finalize_turn)

Rule: finalize_turn() MUST be called at the end of every turn exit path.
"""
from __future__ import annotations

import time
import logging
from enum import Enum

_logger = logging.getLogger("hermes.voice_state")


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PARTIAL_TRANSCRIBING = "partial_transcribing"
    FINAL_TRANSCRIBING = "final_transcribing"
    ROUTING = "routing"
    THINKING_LOCAL = "thinking_local"
    THINKING_FALLBACK = "thinking_fallback"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    ERROR_RECOVERY = "error_recovery"
    RESPONDING = "responding"
    PROCESSING = "processing"


class VoiceStateMachine:
    def __init__(self, agent):
        self.agent = agent
        self.current_state = VoiceState.IDLE
        self.state_history = []
        self.current_turn_id = None
        self.current_wake_turn_id = None
        self.cancel_token = None
        self.bargein_start_time = None
        self._redis = None

        # Fila de fala com cancelamento (contrato Jarvis runtime)
        self.speaking = False
        self.current_speech_id = None
        self.cancel_speech = False

        # Deep analysis flag used by TTS pipeline
        self.deep_analysis = False

        try:
            import redis
            self._redis = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
            self._redis.hset("hermes:voice:state", mapping={
                "state": "idle",
                "turn_id": "",
                "cancel_token": "",
                "speaking": "false",
                "playback_state": "idle",
                "current_speech_id": "",
                "cancel_speech": "0",
                "last_transition": str(time.time()),
                "last_error": "",
            })
            self._redis.publish("hermes:voice:state:events", "idle")
        except Exception as e:
            _logger.debug(f"Failed to connect/initialize Redis in VoiceStateMachine: {e}")

    def get_current_wake_turn_id(self) -> str:
        if self._redis is not None:
            try:
                val = self._redis.hget("hermes:voice:state", "wake_turn_id")
                if val:
                    self.current_wake_turn_id = str(val)
                    return self.current_wake_turn_id
            except Exception as e:
                _logger.debug(f"Failed to read wake_turn_id from Redis: {e}")
        return self.current_wake_turn_id or "unknown_wake_turn_id"

    def start_new_turn(self) -> str:
        import uuid
        self.current_turn_id = f"trn_{uuid.uuid4().hex[:8]}"
        _logger.info(f"Starting new conversational turn: {self.current_turn_id}")
        return self.current_turn_id

    def cancel_current_turn(self) -> None:
        if self.current_turn_id:
            _logger.info(f"Cancelling turn: {self.current_turn_id}")
            self.cancel_token = self.current_turn_id
            self.current_turn_id = None

    def finalize_turn(self, reason: str = "", error: str = "") -> None:
        """Finalize the current turn and always return to IDLE.

        This MUST be called at the end of every turn exit path:
          - Normal TTS completion
          - Silence/timeout (no TTS)
          - Barge-in/interruption
          - Cancellation
          - Error recovery (recoverable errors)

        For real failures (GPU gone, STT crash), do NOT call finalize_turn —
        let the error propagate so it's detected as FAIL in status.py.
        """
        wake_turn_id = self.get_current_wake_turn_id()
        self.current_turn_id = None
        self.speaking = False
        self.deep_analysis = False
        self.cancel_speech = False

        now = time.monotonic()
        wall_now = time.time()
        if self._redis is not None:
            try:
                mapping = {
                    "state": VoiceState.IDLE.value,
                    "turn_id": "",
                    "cancel_token": self.cancel_token or "",
                    "speaking": "false",
                    "playback_state": "idle",
                    "cancel_speech": "0",
                    "current_speech_id": "",
                    "last_transition": str(wall_now),
                    "last_error": error,
                }
                self._redis.hset("hermes:voice:state", mapping=mapping)
                self._redis.publish("hermes:voice:state:events", "idle")
                if wake_turn_id:
                    self._redis.hset(
                        f"hermes:voice:turn:{wake_turn_id}",
                        mapping={
                            "turn_terminal_reason": reason or "completed",
                            "terminal_reason": reason or "completed",
                        }
                    )
                _logger.info(
                    "finalize_turn: state=IDLE reason=%r error=%r ts=%.3f",
                    reason,
                    error,
                    wall_now,
                )
            except Exception as e:
                _logger.debug(f"Failed to finalize turn in Redis: {e}")

        # Update in-memory state
        old_state = self.current_state
        if old_state != VoiceState.IDLE:
            self.current_state = VoiceState.IDLE
            self.state_history.append((now, old_state, VoiceState.IDLE, reason))
            _logger.info(
                "VoiceState Transition: %s -> IDLE | Reason: %s",
                old_state.value.upper(),
                reason,
            )

    def transition_to(self, new_state: VoiceState, reason: str = "") -> None:
        old_state = self.current_state
        if old_state == new_state:
            return
        _logger.info(
            f"VoiceState Transition: {old_state.value.upper()} -> {new_state.value.upper()} | Reason: {reason}"
        )
        self.current_state = new_state
        now = time.time()
        self.state_history.append((now, old_state, new_state, reason))
        if self._redis is not None:
            try:
                redis_state = new_state.value
                if new_state == VoiceState.FINAL_TRANSCRIBING:
                    redis_state = "processing"
                mapping = {
                    "state": redis_state,
                    "turn_id": self.current_turn_id or "",
                    "cancel_token": self.cancel_token or "",
                    "last_transition": str(now),
                }
                self._redis.hset("hermes:voice:state", mapping=mapping)

                import json
                event_data = {
                    "state": new_state.value,
                    "turn_id": self.current_turn_id,
                    "cancel_token": self.cancel_token,
                    "reason": reason,
                    "timestamp": now,
                }
                self._redis.publish("hermes:voice:state:events", json.dumps(event_data))
            except Exception as e:
                _logger.debug(f"Failed to publish state {new_state} to Redis: {e}")
