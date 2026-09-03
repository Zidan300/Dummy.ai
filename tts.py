"""Piper synthesis and ffplay playback with safe cancellation."""

from __future__ import annotations

import logging
from array import array
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave


logger = logging.getLogger(__name__)
MODEL = str(Path(__file__).resolve().with_name("en_US-lessac-medium.onnx"))
PIPER = os.environ.get("DUMMY_PIPER", "") or shutil.which("piper") or "piper"
FFPLAY = os.environ.get("DUMMY_FFPLAY", "") or shutil.which("ffplay") or "ffplay"


class TTSError(RuntimeError):
    """Piper synthesis or ffplay playback failed."""


class SpeechPlayer:
    """Owns at most one Piper/ffplay process and can stop both safely."""

    def __init__(self) -> None:
        self._cancel_event = threading.Event()
        self._lock = threading.RLock()
        self._processes: set[subprocess.Popen] = set()
        self._ffplay_process: subprocess.Popen | None = None

    def reset_cancellation(self) -> None:
        """Prepare the player for a new response after an interruption."""
        with self._lock:
            if self._processes:
                raise TTSError("cannot reset TTS while a speech process is active")
            self._cancel_event.clear()

    def cancel(self) -> bool:
        """Cancel active synthesis/playback and report whether ffplay was active."""
        self._cancel_event.set()
        with self._lock:
            processes = list(self._processes)
            ffplay_active = self._ffplay_process is not None and self._ffplay_process.poll() is None
        for process in processes:
            self._terminate(process)
        # The OS processes are stopped synchronously above. Remove them from
        # the ownership set immediately so a new post-interruption response
        # does not lose a race with the old worker's final cleanup.
        with self._lock:
            self._processes.difference_update(processes)
            if self._ffplay_process in processes:
                self._ffplay_process = None
        return ffplay_active

    def speak(
        self,
        text: str,
        cancel_event: threading.Event | None = None,
        on_piper_start=None,
        on_audio_ready=None,
        on_playback_start=None,
        on_playback_level=None,
    ) -> bool:
        text = text.strip()
        if not text or self._cancelled(cancel_event):
            return False

        self._cancel_event.clear()
        wav_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="dummy-tts-",
                suffix=".wav",
                delete=False,
            ) as wav_file:
                wav_path = wav_file.name

            if on_piper_start:
                on_piper_start()
            piper = self._start(
                [PIPER, "-m", MODEL, "-f", wav_path],
                stdin=subprocess.PIPE,
            )
            try:
                if piper.stdin is not None:
                    piper.stdin.write(text + "\n")
                    piper.stdin.close()
                if not self._wait(piper, cancel_event):
                    return False
                if piper.returncode != 0:
                    raise TTSError(self._failure_message("Piper", piper))
            finally:
                if piper.poll() is None:
                    self._terminate(piper)
                self._forget(piper)

            if self._cancelled(cancel_event):
                return False

            if on_audio_ready:
                on_audio_ready()
            player = self._start(
                [
                    FFPLAY,
                    "-autoexit",
                    "-nodisp",
                    wav_path,
                ],
                stdin=subprocess.DEVNULL,
            )
            with self._lock:
                self._ffplay_process = player
            try:
                if on_playback_start:
                    on_playback_start()
                if not self._wait(
                    player,
                    cancel_event,
                    wav_path=wav_path,
                    on_playback_level=on_playback_level,
                ):
                    return False
                if player.returncode != 0:
                    raise TTSError(self._failure_message("ffplay", player))
            finally:
                with self._lock:
                    if self._ffplay_process is player:
                        self._ffplay_process = None
                if player.poll() is None:
                    self._terminate(player)
                self._forget(player)
            return True
        except TTSError:
            raise
        except (OSError, ValueError, BrokenPipeError) as exc:
            raise TTSError(f"speech playback failed: {exc}") from exc
        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary speech file: %s", wav_path)

    def _cancelled(self, external: threading.Event | None) -> bool:
        return self._cancel_event.is_set() or bool(external and external.is_set())

    def _start(self, args: list[str], stdin) -> subprocess.Popen:
        process = subprocess.Popen(
            args,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._processes.add(process)
        return process

    def _wait(
        self,
        process: subprocess.Popen,
        external: threading.Event | None,
        wav_path: str | None = None,
        on_playback_level=None,
    ) -> bool:
        meter = None
        if wav_path and on_playback_level:
            try:
                meter = _WaveLevelMeter(wav_path)
            except (OSError, ValueError, wave.Error):
                # Telemetry is optional; a malformed/unavailable WAV must not
                # change the outcome of otherwise valid playback.
                logger.debug("Could not open WAV for playback level", exc_info=True)
        started_at = time.monotonic()
        try:
            while process.poll() is None:
                if self._cancelled(external):
                    self._terminate(process)
                    return False
                if meter is not None:
                    try:
                        on_playback_level(meter.level_at(time.monotonic() - started_at))
                    except (OSError, ValueError, wave.Error):
                        logger.debug("Could not measure TTS playback level", exc_info=True)
                        meter.close()
                        meter = None
                time.sleep(0.05)
            if on_playback_level:
                on_playback_level(0.0)
            return not self._cancelled(external)
        finally:
            if meter is not None:
                meter.close()

    def _forget(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    @staticmethod
    def _failure_message(name: str, process: subprocess.Popen) -> str:
        details = ""
        if process.stderr is not None:
            try:
                details = process.stderr.read().strip()
            except OSError:
                details = ""
        suffix = f": {details}" if details else ""
        return f"{name} exited with status {process.returncode}{suffix}"

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.75)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=0.75)
            except (OSError, subprocess.TimeoutExpired):
                logger.error("Speech process did not stop cleanly")


class _WaveLevelMeter:
    """Read small sequential WAV windows for UI-only output amplitude."""

    def __init__(self, path: str) -> None:
        self._wave = wave.open(path, "rb")
        self._rate = self._wave.getframerate()
        self._width = self._wave.getsampwidth()
        self._channels = self._wave.getnchannels()
        self._read_frames = 0

    def close(self) -> None:
        self._wave.close()

    def level_at(self, elapsed: float) -> float:
        target_frames = max(self._read_frames, int(max(0.0, elapsed) * self._rate))
        frames_to_read = min(target_frames - self._read_frames, max(1, self._rate // 20))
        if frames_to_read <= 0:
            return 0.0
        data = self._wave.readframes(frames_to_read)
        actual_frames = len(data) // max(1, self._width * self._channels)
        self._read_frames += actual_frames
        if not data or actual_frames <= 0:
            return 0.0
        values = self._samples(data)
        if not values:
            return 0.0
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        peak = float((1 << (8 * self._width - 1)) - 1)
        return min(1.0, rms / max(1.0, peak) * 4.0)

    def _samples(self, data: bytes) -> list[int]:
        if self._width == 1:
            return [sample - 128 for sample in data]
        if self._width == 2:
            values = array("h")
        elif self._width == 4:
            values = array("i")
        else:
            return []
        values.frombytes(data[: len(data) - (len(data) % values.itemsize)])
        if getattr(values, "itemsize", 0) and sys.byteorder != "little":
            values.byteswap()
        return list(values)


_default_player = SpeechPlayer()


def speak(text: str) -> bool:
    return _default_player.speak(text)
