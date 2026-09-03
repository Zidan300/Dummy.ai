"""Persistent microphone capture, WebRTC VAD, and Whisper transcription."""

from __future__ import annotations

from collections import deque
import logging
import math
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
MODEL_SIZE = "tiny"
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 2

# These values deliberately use consecutive frames. A single quiet frame never
# ends an utterance, and the pre-roll preserves the start of a word.
PREBUFFER_MS = 300
PREBUFFER_FRAMES = PREBUFFER_MS // FRAME_MS
SPEECH_START_FRAMES = 3
SILENCE_DURATION = 0.72
SILENCE_FRAMES = math.ceil(SILENCE_DURATION * 1000 / FRAME_MS)
MAX_UTTERANCE_SECONDS = 20.0

# Barge-in uses the same VAD and persistent capture stream, but it is allowed
# to hand short rolling segments to Whisper before normal utterance silence.
BARGE_MIN_AUDIO_SECONDS = 0.36
BARGE_RECOGNITION_INTERVAL_SECONDS = 0.18
BARGE_MAX_SEGMENT_SECONDS = 1.20
BARGE_MAX_SEGMENT_FRAMES = math.ceil(BARGE_MAX_SEGMENT_SECONDS * 1000 / FRAME_MS)
BARGE_CONFIRM_SILENCE_FRAMES = math.ceil(0.12 * 1000 / FRAME_MS)


class AudioError(RuntimeError):
    """A microphone or audio-stream failure that may be recoverable."""


class VadError(RuntimeError):
    """A WebRTC VAD failure."""


class TranscriptionError(RuntimeError):
    """A Whisper failure."""


class AudioCapture:
    """Keep one callback-based PortAudio stream open for the app lifetime."""

    def __init__(self, on_level=None) -> None:
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=160)
        self._events: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=16)
        self._stream: sd.InputStream | None = None
        self._lock = threading.RLock()
        self._accepting = False
        self._last_overflow_report = 0.0
        self._last_frame_at = 0.0
        self._on_level = on_level

    def start(self) -> None:
        with self._lock:
            if self._stream is not None and self._is_active_locked():
                return

            stale_stream = self._stream
            self._stream = None
            self._accepting = False

        if stale_stream is not None:
            try:
                stale_stream.stop()
            except Exception:
                logger.debug("Stale microphone stop reported an error", exc_info=True)
            try:
                stale_stream.close()
            except Exception:
                logger.debug("Stale microphone close reported an error", exc_info=True)

        with self._lock:
            self._clear_frames()
            self._accepting = True
            self._last_frame_at = time.monotonic()
            stream: sd.InputStream | None = None
            try:
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=FRAME_SIZE,
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream
            except Exception as exc:
                self._accepting = False
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug("Could not close failed microphone stream", exc_info=True)
                raise AudioError(f"microphone start failed: {exc}") from exc

    def stop(self) -> None:
        """Stop and release the current stream; safe to call repeatedly."""
        with self._lock:
            self._accepting = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                logger.debug("Microphone stop reported an error", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("Microphone close reported an error", exc_info=True)

        self._clear_frames()

    def is_active(self) -> bool:
        with self._lock:
            return self._stream is not None and self._is_active_locked()

    def is_healthy(self, max_callback_gap: float = 2.0) -> bool:
        with self._lock:
            return (
                self._stream is not None
                and self._is_active_locked()
                and time.monotonic() - self._last_frame_at <= max_callback_gap
            )

    def read_frame(self, timeout: float = 0.2) -> np.ndarray | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def _is_active_locked(self) -> bool:
        try:
            return bool(self._stream and self._stream.active)
        except Exception:
            return False

    def _clear_frames(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def clear_pending_frames(self) -> None:
        """Discard audio captured during processing or speaker playback."""
        self._clear_frames()

    def _report_event(self, kind: str, message: str) -> None:
        try:
            self._events.put_nowait((kind, message))
        except queue.Full:
            # A later health check will notice an inactive stream. Do not let
            # an error-reporting queue take down the PortAudio callback.
            pass

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            now = time.monotonic()
            status_text = str(status)
            is_overflow = bool(getattr(status, "input_overflow", False))
            if is_overflow:
                if now - self._last_overflow_report >= 2.0:
                    self._last_overflow_report = now
                    self._report_event("overflow", status_text)
            else:
                self._report_event("stream", status_text)

        if not self._accepting:
            return

        try:
            frame = np.asarray(indata[:, 0], dtype=np.int16).copy()
            if len(frame) != FRAME_SIZE:
                self._report_event("stream", f"unexpected audio frame size: {len(frame)}")
                return
            self._last_frame_at = time.monotonic()
            if self._on_level is not None:
                try:
                    # A normalized RMS estimate is cheap enough for the existing
                    # PortAudio callback. The Qt signal receiver performs all UI
                    # work on the GUI thread.
                    normalized = frame.astype(np.float32) / 32768.0
                    level = float(min(1.0, math.sqrt(float(np.mean(normalized * normalized))) * 4.0))
                    self._on_level(level)
                except Exception as exc:
                    self._report_event("stream", f"audio level callback failed: {exc}")
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                # Preserve the newest audio and report the condition at a low
                # rate instead of allowing callback backpressure to grow.
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    pass
                now = time.monotonic()
                if now - self._last_overflow_report >= 2.0:
                    self._last_overflow_report = now
                    self._report_event("overflow", "audio frame queue full")
        except Exception as exc:
            self._report_event("stream", f"microphone callback failed: {exc}")


class UtteranceDetector:
    """Collect one utterance from a persistent capture stream."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS) -> None:
        try:
            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception as exc:
            raise VadError(f"VAD initialization failed: {exc}") from exc

    def listen(
        self,
        capture: AudioCapture,
        stop_event: threading.Event,
        on_speech_detected=None,
        ignore_until: float = 0.0,
        max_utterance_seconds: float = MAX_UTTERANCE_SECONDS,
    ) -> np.ndarray | None:
        prebuffer: deque[np.ndarray] = deque(maxlen=PREBUFFER_FRAMES)
        utterance: list[np.ndarray] = []
        speech_run = 0
        silence_run = 0
        speaking = False
        started_at = 0.0

        while not stop_event.is_set():
            if time.monotonic() < ignore_until:
                # Consume the short speaker-start guard without allowing
                # Dummy's own audio into the interruption pre-buffer.
                capture.read_frame(timeout=0.1)
                continue

            for kind, message in capture.drain_events():
                if kind == "overflow":
                    logger.warning("Audio input overflow; continuing")
                else:
                    raise AudioError(message)

            if not capture.is_active():
                raise AudioError("microphone stream is no longer active")
            if hasattr(capture, "is_healthy") and not capture.is_healthy():
                raise AudioError("microphone stream stopped delivering audio")

            frame = capture.read_frame()
            if frame is None:
                continue

            try:
                is_speech = self._vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            except Exception as exc:
                raise VadError(f"VAD processing failed: {exc}") from exc

            if not speaking:
                prebuffer.append(frame)
                if is_speech:
                    speech_run += 1
                    if speech_run >= SPEECH_START_FRAMES:
                        speaking = True
                        started_at = time.monotonic()
                        utterance = list(prebuffer)
                        if on_speech_detected:
                            on_speech_detected()
                else:
                    speech_run = 0
                continue

            utterance.append(frame)
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1

            if silence_run >= SILENCE_FRAMES:
                return self._to_float_audio(utterance)

            if time.monotonic() - started_at >= max_utterance_seconds:
                logger.info("Maximum utterance duration reached")
                return self._to_float_audio(utterance)

        return None

    @staticmethod
    def _to_float_audio(frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames).astype(np.float32) / 32768.0


class BargeInDetector:
    """Collect short rolling speech segments for low-latency control words."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS) -> None:
        try:
            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception as exc:
            raise VadError(f"Barge-in VAD initialization failed: {exc}") from exc

    def listen_for_segments(
        self,
        capture: AudioCapture,
        stop_event,
        on_segment,
        on_speech_detected=None,
        ignore_until=0.0,
    ) -> None:
        """Run until the current speech ends or ``on_segment`` confirms control.

        The callback is invoked on bounded rolling audio windows. Returning
        True stops the collector immediately; False continues collecting the
        same utterance. Normal questions therefore remain harmless while a
        short, explicit STOP can be recognized before the silence timeout.
        """
        prebuffer: deque[np.ndarray] = deque(maxlen=PREBUFFER_FRAMES)
        speech_frames: list[np.ndarray] = []
        speech_duration_frames = 0
        speech_run = 0
        silence_run = 0
        speaking = False
        last_attempt_at = 0.0

        def current_ignore_until() -> float:
            value = ignore_until() if callable(ignore_until) else ignore_until
            return float(value)

        def reset_segment() -> None:
            nonlocal speech_frames, speech_duration_frames, speech_run
            nonlocal silence_run, speaking, last_attempt_at
            prebuffer.clear()
            speech_frames = []
            speech_duration_frames = 0
            speech_run = 0
            silence_run = 0
            speaking = False
            last_attempt_at = 0.0

        while not stop_event.is_set():
            if time.monotonic() < current_ignore_until():
                reset_segment()
                capture.read_frame(timeout=0.1)
                continue

            for kind, message in capture.drain_events():
                if kind == "overflow":
                    logger.warning("Audio input overflow during barge-in; continuing")
                else:
                    raise AudioError(message)

            if not capture.is_active():
                raise AudioError("microphone stream is no longer active")
            if hasattr(capture, "is_healthy") and not capture.is_healthy():
                raise AudioError("microphone stream stopped delivering audio")

            frame = capture.read_frame()
            if frame is None:
                continue

            try:
                is_speech = self._vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            except Exception as exc:
                raise VadError(f"Barge-in VAD processing failed: {exc}") from exc

            if not speaking:
                prebuffer.append(frame)
                if is_speech:
                    speech_run += 1
                    if speech_run >= SPEECH_START_FRAMES:
                        speaking = True
                        speech_frames = list(prebuffer)
                        speech_duration_frames = len(speech_frames)
                        silence_run = 0
                        last_attempt_at = 0.0
                        if on_speech_detected:
                            on_speech_detected()
                else:
                    speech_run = 0
                continue

            speech_frames.append(frame)
            speech_duration_frames += 1
            if len(speech_frames) > BARGE_MAX_SEGMENT_FRAMES:
                speech_frames = speech_frames[-BARGE_MAX_SEGMENT_FRAMES:]
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1

            now = time.monotonic()
            audio_seconds = speech_duration_frames * FRAME_MS / 1000.0
            ready = audio_seconds >= BARGE_MIN_AUDIO_SECONDS
            due = now - last_attempt_at >= BARGE_RECOGNITION_INTERVAL_SECONDS
            if ready and due:
                # Playback can begin while Whisper is being prepared. Drop
                # the in-flight segment if the speaker guard was raised.
                if now < current_ignore_until():
                    reset_segment()
                    continue
                segment = speech_frames[-BARGE_MAX_SEGMENT_FRAMES:]
                last_attempt_at = now
                if on_segment(
                    UtteranceDetector._to_float_audio(segment),
                    silence_run >= BARGE_CONFIRM_SILENCE_FRAMES,
                ):
                    return

            # Preserve the normal, conservative end-of-utterance behavior for
            # non-control speech. Only the recognition start is accelerated.
            if silence_run >= SILENCE_FRAMES:
                return


class WhisperTranscriber:
    """Lazy, one-time CPU int8 Whisper model owner."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                logger.info("Loading Whisper (%s, CPU int8)", MODEL_SIZE)
                try:
                    self._model = WhisperModel(
                        MODEL_SIZE,
                        device="cpu",
                        compute_type="int8",
                    )
                except Exception as exc:
                    raise TranscriptionError(f"Whisper initialization failed: {exc}") from exc
                logger.info("Whisper ready")

    def transcribe(
        self,
        audio: np.ndarray,
        stop_event: threading.Event | None = None,
    ) -> str:
        if audio.size == 0:
            return ""
        self.load()
        if stop_event is not None and stop_event.is_set():
            return ""

        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text_parts: list[str] = []
            for segment in segments:
                if stop_event is not None and stop_event.is_set():
                    return ""
                text_parts.append(segment.text)
            return " ".join(text_parts).strip()
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc


# Backwards-compatible helpers for callers that used the old module directly.
_default_capture: AudioCapture | None = None
_default_transcriber: WhisperTranscriber | None = None


def record_audio() -> np.ndarray:
    """Record one utterance using a temporary capture for legacy callers."""
    global _default_capture
    if _default_capture is None:
        _default_capture = AudioCapture()
    stop_event = threading.Event()
    detector = UtteranceDetector()
    _default_capture.start()
    try:
        audio = detector.listen(_default_capture, stop_event)
        return audio if audio is not None else np.array([], dtype=np.float32)
    finally:
        _default_capture.stop()


def transcribe(audio: np.ndarray) -> str:
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = WhisperTranscriber()
    return _default_transcriber.transcribe(audio)
