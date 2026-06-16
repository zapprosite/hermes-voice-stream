"""agent/voice_heartbeat.py — Periodic heartbeat writer for the Hermes voice daemon.

Writes /run/user/$UID/hermes/voice_readiness.json every HEARTBEAT_INTERVAL_S seconds
while the agent is alive. This allows `hermes` CLI to read readiness state fast (<30ms)
without running `hermes status` (which takes ~900ms for a full Python process startup).

Schema mirrors what hermes_voice/status.py writes so that status.py can read either.

The heartbeat is a lightweight writer — it does NOT replicate the full status.py logic.
It writes what the agent knows directly:
  - state (from Redis / state machine)
  - wake_listening (from Redis)
  - gpu_ready (cached from startup check, refreshed every 60s)
  - service_state: always "active" (if this task is running, the service is active)
  - audio_input_device: from Redis
  - tts_ready / stt_ready / oww_ready: from startup health flags (set once)
  - last_transition: from Redis
  - last_error: from Redis

For full diagnostics, `hermes doctor` runs status.py which has the complete logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

_logger = logging.getLogger("hermes.voice.heartbeat")

HEARTBEAT_INTERVAL_S = 3.0  # Write heartbeat every 3 seconds
_GPU_RECHECK_INTERVAL_S = 60.0  # Re-check GPU every minute

# Runtime directory (tmpfs — survives restart but not reboot)
_UID = os.getuid()
_RUNTIME_DIR = Path(f"/run/user/{_UID}/hermes")
_HEARTBEAT_PATH = _RUNTIME_DIR / "voice_readiness.json"

# Atomic write: write to .tmp then rename
_HEARTBEAT_TMP = _RUNTIME_DIR / "voice_readiness.json.tmp"


def _check_gpu() -> bool:
    """Check if nvidia-smi reports a GPU. Cached by caller."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _read_redis_snapshot(redis_client: Any) -> dict:
    """Read the current voice state from Redis. Returns empty dict on failure."""
    try:
        data = redis_client.hgetall("hermes:voice:state") or {}
        return data
    except Exception as e:
        _logger.debug("heartbeat: failed to read Redis: %s", e)
        return {}


def _classify_readiness(
    state: str,
    last_transition_ts: float,
    all_healthy: bool,
    service_state: str,
    gpu_ready: bool,
) -> tuple[str, str]:
    """Lightweight readiness classifier for the heartbeat writer.

    Returns (readiness_level, reason) — mirrors status.py semantics
    but with only the info available to the daemon directly.
    """
    if service_state not in ("active", "running"):
        return "BOOTING", f"service_state={service_state}"

    if not gpu_ready:
        return "FAIL", "gpu_unavailable"

    active_states = {
        "listening", "partial_transcribing", "final_transcribing",
        "routing", "thinking_local", "thinking_fallback", "speaking",
    }

    if state in ("idle", "ready", "armed_waiting_for_wake"):
        if all_healthy:
            return "READY", "armed_waiting_for_wake"
        return "MODEL_WARMING", "signals_not_all_ready"

    if state == "listening":
        return "BUSY_AUDIO", f"agent_listening"

    if state in ("speaking", "routing", "thinking_local", "thinking_fallback"):
        return "BUSY_AUDIO", f"agent_{state}"

    if state in ("final_transcribing", "partial_transcribing"):
        now = time.time()
        age = now - last_transition_ts if last_transition_ts > 0 else 0
        if age < 120:
            return "BUSY_AUDIO", f"agent_{state}_recent_{int(age)}s"
        # Stale transcribing state
        if all_healthy:
            return "STALE_STATE", f"redis_state_stale (state='{state}' for {int(age)}s, all_healthy=True)"
        return "DEGRADED", f"state_stuck_{state}_{int(age)}s"

    if state in ("interrupted", "cancelled", "error_recovery"):
        # These should be transient; if stuck, show as STALE
        now = time.time()
        age = now - last_transition_ts if last_transition_ts > 0 else 0
        if age < 30:
            return "BUSY_AUDIO", f"agent_{state}_transitioning"
        if all_healthy:
            return "STALE_STATE", f"redis_state_stale (state='{state}' for {int(age)}s)"
        return "DEGRADED", f"state_stuck_{state}_{int(age)}s"

    return "READY", "armed_waiting_for_wake"


def _write_heartbeat_atomic(payload: dict) -> bool:
    """Write heartbeat JSON atomically to tmpfs. Returns True on success."""
    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _HEARTBEAT_TMP
        tmp.write_text(json.dumps(payload, indent=None), encoding="utf-8")
        tmp.rename(_HEARTBEAT_PATH)
        return True
    except Exception as e:
        _logger.debug("heartbeat: write failed: %s", e)
        return False


async def run_heartbeat_daemon(
    state_machine: Any,
    *,
    stt_ready: bool = True,
    tts_ready: bool = True,
    oww_ready: bool = True,
    audio_input_device: str = "pipewire",
) -> None:
    """Asyncio background task: writes heartbeat every HEARTBEAT_INTERVAL_S.

    Should be launched as:
        asyncio.create_task(run_heartbeat_daemon(state_machine, ...))

    The task runs indefinitely until cancelled (on agent shutdown).
    """
    _logger.info(
        "heartbeat_daemon: starting (interval=%.1fs path=%s)",
        HEARTBEAT_INTERVAL_S,
        _HEARTBEAT_PATH,
    )

    redis_client = state_machine._redis
    gpu_ready = _check_gpu()
    last_gpu_check = time.time()

    consecutive_failures = 0

    while True:
        try:
            now = time.time()

            # Re-check GPU periodically
            if now - last_gpu_check > _GPU_RECHECK_INTERVAL_S:
                gpu_ready = _check_gpu()
                last_gpu_check = now

            # Read current state from Redis (or fall back to state machine memory)
            if redis_client is not None:
                snapshot = _read_redis_snapshot(redis_client)
            else:
                snapshot = {}

            state = snapshot.get("state", state_machine.current_state.value)
            wake_listening = snapshot.get("wake_listening", "0")
            wake_model_loaded = snapshot.get("wake_model_loaded", "0")
            last_transition_raw = snapshot.get("last_transition", "0")
            last_error = snapshot.get("last_error", "")
            audio_dev = snapshot.get("audio_input_device", audio_input_device)

            try:
                last_transition_ts = float(last_transition_raw) if last_transition_raw else 0.0
            except (ValueError, TypeError):
                last_transition_ts = 0.0

            # Determine all_healthy flag
            all_healthy = (
                stt_ready
                and tts_ready
                and oww_ready
                and gpu_ready
                and bool(audio_dev)
                and wake_listening in ("1", "yes", "true")
            )

            readiness, reason = _classify_readiness(
                state=state,
                last_transition_ts=last_transition_ts,
                all_healthy=all_healthy,
                service_state="active",  # If this task is running, service is active
                gpu_ready=gpu_ready,
            )

            payload = {
                "timestamp": now,
                "readiness": readiness,
                "reason": reason,
                "service_state": "active",
                "service_substate": "running",
                "audio_input_ready": bool(audio_dev),
                "audio_input_device": audio_dev or "",
                "oww_ready": oww_ready,
                "stt_ready": stt_ready,
                "tts_ready": tts_ready,
                "gpu_ready": gpu_ready,
                "headless_client_alive": True,  # This task lives in the same process
                "wake_listening": wake_listening in ("1", "yes", "true"),
                "wake_model_loaded": wake_model_loaded in ("1", "yes", "true"),
                "state": state,
                "provider_info": f"omnivoice={'ready' if tts_ready else 'unready'},cuda={gpu_ready}",
                "last_error": last_error,
                "last_transition": str(int(last_transition_ts)) if last_transition_ts else "",
                # Daemon marker: allows status.py to distinguish daemon-written heartbeats
                "writer": "daemon",
            }

            success = _write_heartbeat_atomic(payload)
            if success:
                consecutive_failures = 0
                _logger.debug(
                    "heartbeat_daemon: wrote readiness=%s state=%s", readiness, state
                )
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    _logger.warning(
                        "heartbeat_daemon: %d consecutive write failures", consecutive_failures
                    )

        except asyncio.CancelledError:
            _logger.info("heartbeat_daemon: cancelled, stopping")
            raise
        except Exception as e:
            _logger.debug("heartbeat_daemon: unexpected error: %s", e)

        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
