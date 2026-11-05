"""Speech detection from a microphone, using plain signal processing (no model, no recording).

Each 30 ms frame gets three numbers: loudness against a running noise floor, spectral flatness
(speech has strong harmonics, so its spectrum is "peaky"; fans and HVAC are flat), and the share
of energy in the 300-3400 Hz speech band. A frame counts as speech when all three agree, with a
short hangover so pauses between words don't flicker. The noise floor is a low percentile of the
last few seconds of levels, so it follows whatever steady background the room has.
Audio is analysed in memory and discarded.
"""

from __future__ import annotations

import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np


class SpeechDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        margin_db: float = 8.0,
        max_flatness: float = 0.35,
        min_band_ratio: float = 0.4,  # low male voices put real energy under 300 Hz; fans sit near 0.2
        hangover_frames: int = 6,
        noise_window_s: float = 4.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.margin_db = margin_db
        self.max_flatness = max_flatness
        self.min_band_ratio = min_band_ratio
        self.hangover_frames = hangover_frames
        self.window = np.hanning(self.frame_len)
        freqs = np.fft.rfftfreq(self.frame_len, 1.0 / sample_rate)
        self._shape_bins = (freqs >= 100) & (freqs <= 4000)
        self._speech_bins = (freqs >= 300) & (freqs <= 3400)
        self._all_bins = (freqs >= 60) & (freqs <= min(8000, sample_rate / 2))
        # Noise floor = a low percentile of recent frame levels ("minimum statistics"). Speech always has
        # pauses, so the floor tracks the background even while people talk, and steady noise sets it.
        self._levels: deque[float] = deque(maxlen=max(5, int(noise_window_s * 1000 / frame_ms)))
        self.noise_db: float | None = None
        self._hang = 0
        self._buffer = np.zeros(0, dtype=np.float32)

    def features(self, frame: np.ndarray) -> tuple[float, float, float]:
        """Return (level in dBFS, spectral flatness 0..1, speech-band energy ratio 0..1) for one frame."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        level_db = 20.0 * np.log10(rms + 1e-10)
        power = np.abs(np.fft.rfft(frame * self.window)) ** 2 + 1e-12
        shape = power[self._shape_bins]
        flatness = float(np.exp(np.mean(np.log(shape))) / np.mean(shape))
        band_ratio = float(power[self._speech_bins].sum() / power[self._all_bins].sum())
        return level_db, flatness, band_ratio

    def process(self, samples: np.ndarray) -> list[bool]:
        """Feed mono float samples in [-1, 1]; returns one speech flag per complete frame."""
        self._buffer = np.concatenate([self._buffer, np.asarray(samples, dtype=np.float32).ravel()])
        flags = []
        while len(self._buffer) >= self.frame_len:
            frame, self._buffer = self._buffer[: self.frame_len], self._buffer[self.frame_len :]
            level_db, flatness, band_ratio = self.features(frame)
            self._levels.append(level_db)
            self.noise_db = float(np.percentile(self._levels, 10))
            candidate = (
                level_db > self.noise_db + self.margin_db
                and flatness < self.max_flatness
                and band_ratio > self.min_band_ratio
            )
            if candidate:
                self._hang = self.hangover_frames
                flags.append(True)
            elif self._hang > 0:
                self._hang -= 1
                flags.append(True)
            else:
                flags.append(False)
        return flags


class SpeechActivity:
    """Thread-safe rolling share of speech frames over the last `window_s` seconds."""

    def __init__(self, window_s: float = 10.0) -> None:
        self.window_s = window_s
        self._frames: deque[tuple[float, bool]] = deque()
        self._lock = threading.Lock()

    def add(self, t: float, flags: list[bool], frame_s: float) -> None:
        with self._lock:
            for i, f in enumerate(flags):
                self._frames.append((t + i * frame_s, f))
            self._trim(t + len(flags) * frame_s)

    def _trim(self, now: float) -> None:
        while self._frames and self._frames[0][0] < now - self.window_s:
            self._frames.popleft()

    def ratio(self, now: float) -> float | None:
        """Share of speech in the window ending at `now`, or None if there is no audio yet."""
        with self._lock:
            self._trim(now)
            if not self._frames:
                return None
            return sum(f for _, f in self._frames) / len(self._frames)


def read_wav(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Read a 16-bit PCM WAV as mono float32, resampled (linearly) to `target_rate`."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and len(data):
        n = int(len(data) * target_rate / rate)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


class WavFeed:
    """Plays a WAV file into the detector in step with a video clock."""

    def __init__(self, path: str | Path, detector: SpeechDetector, activity: SpeechActivity) -> None:
        self.samples = read_wav(path, detector.sample_rate)
        self.detector, self.activity = detector, activity
        self.pos = 0

    def advance_to(self, t: float) -> None:
        end = min(len(self.samples), int(t * self.detector.sample_rate))
        if end <= self.pos:
            return
        start_t = self.pos / self.detector.sample_rate
        flags = self.detector.process(self.samples[self.pos : end])
        self.pos = end
        self.activity.add(start_t, flags, self.detector.frame_len / self.detector.sample_rate)


class MicFeed:
    """Live microphone input via sounddevice (optional dependency: pip install 'roomsense[audio]')."""

    def __init__(self, detector: SpeechDetector, activity: SpeechActivity, clock) -> None:
        import sounddevice as sd

        self.detector, self.activity, self.clock = detector, activity, clock
        self._lock = threading.Lock()
        self.stream = sd.InputStream(
            samplerate=detector.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=detector.frame_len * 4,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG002 (sounddevice signature)
        with self._lock:
            flags = self.detector.process(indata[:, 0].copy())
        self.activity.add(
            self.clock() - frames / self.detector.sample_rate,
            flags,
            self.detector.frame_len / self.detector.sample_rate,
        )

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()
