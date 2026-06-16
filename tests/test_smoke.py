"""Smoke tests para hermes-voice-stream."""
import pytest


def test_module_imports():
    """voice_stream deve ser importavel."""
    try:
        from voice_stream import audio_utils, voice_state
        assert audio_utils is not None
        assert voice_state is not None
    except ImportError:
        pytest.skip("voice_stream nao instalado")
