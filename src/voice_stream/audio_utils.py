# agent/audio_utils.py
import re
import io
import wave
from typing import AsyncIterator
from livekit import rtc
from livekit.agents.utils import AudioBuffer

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.?!])(?:\s+|$)")
_STREAM_BOUNDARY_RE = re.compile(r"(?<=[.?!,;:])(?:\s+|$)")
_MD_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL)
_MD_FENCE_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_URL_RE = re.compile(r"https?://\S+")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"\*(.+?)\*")
_MD_CODE_RE = re.compile(r"`(.+?)`")
_MD_HEADER_RE = re.compile(r"^#+\s*", flags=re.MULTILINE)
_MD_LIST_RE = re.compile(r"^\s*[-*]\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"---+")
_MD_EXCESS_NL_RE = re.compile(r"\n{3,}")

def _strip_markdown_for_tts(text: str) -> str:
    text = _MD_THINK_BLOCK_RE.sub("", text)
    text = _MD_FENCE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_URL_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_LIST_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    text = _MD_EXCESS_NL_RE.sub("\n\n", text)
    return text.strip()

def _buffer_to_wav_bytes(buffer: AudioBuffer) -> bytes:
    frames = list(buffer) if isinstance(buffer, list) else [buffer]
    if not frames:
        raise RuntimeError("LiveKit STT buffer was empty")
    first = frames[0]
    sample_rate = first.sample_rate
    num_channels = first.num_channels
    with io.BytesIO() as wav_file:
        with wave.open(wav_file, "wb") as wav_writer:
            wav_writer.setnchannels(num_channels)
            wav_writer.setsampwidth(2)
            wav_writer.setframerate(sample_rate)
            for frame in frames:
                if frame.sample_rate != sample_rate or frame.num_channels != num_channels:
                    raise RuntimeError(
                        f"LiveKit STT buffer changed audio shape mid-turn "
                        f"(expected {sample_rate}Hz/{num_channels}ch, "
                        f"got {frame.sample_rate}Hz/{frame.num_channels}ch)"
                    )
                wav_writer.writeframes(bytes(frame.data))
        return wav_file.getvalue()

def _normalize_whisper_language(language: str) -> str:
    value = str(language or "").strip()
    if not value:
        return "pt"
    return value.split("-", 1)[0].split("_", 1)[0]

def _wav_bytes_to_frames(wav_bytes: bytes) -> AsyncIterator[rtc.AudioFrame]:
    async def _iterate() -> AsyncIterator[rtc.AudioFrame]:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                if sample_width != 2:
                    raise RuntimeError(f"OmniVoice returned unsupported sample width: {sample_width}")
                samples_per_chunk = max(sample_rate // 50, 1)
                while True:
                    data = wf.readframes(samples_per_chunk)
                    if not data:
                        break
                    samples_per_channel = len(data) // (num_channels * sample_width)
                    if samples_per_channel <= 0:
                        continue
                    yield rtc.AudioFrame(
                        data=data,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=samples_per_channel,
                    )
        except wave.Error as exc:
            raise RuntimeError(f"OmniVoice returned invalid WAV audio: {exc}") from exc
    return _iterate()
