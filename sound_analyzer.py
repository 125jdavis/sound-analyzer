#!/usr/bin/env python3
"""
Sound Analyzer - desktop spectrogram + spectrum analyzer for audio files.

Built for automotive NVH work: load a recording, inspect the time-frequency
picture (spectrogram), and read exact frequency / level values from a linked
spectrum chart.

Run
    python sound_analyzer.py                  # empty window, use File > Open or drag & drop
    python sound_analyzer.py recording.wav    # open a file directly
    python sound_analyzer.py --demo           # synthetic engine run-up to try things out

Mouse (both charts)
    Wheel ................. zoom both axes around the pointer
    Wheel over an axis .... zoom only that axis
    Ctrl + wheel .......... zoom X only        Shift + wheel ... zoom Y only
    Left-drag ............. pan   (draws a zoom box when "Box zoom" is on)
    Right-drag ............ stretch / squeeze X and Y independently
    Right-click ........... view menu (view all, export PNG / CSV, ...)

Cursor
    Hover the spectrogram . spectrum chart shows that moment in time
    Left-click spectrogram  move the playhead there and pin the cursor frequency
    Hover / click spectrum  move / pin the cursor frequency
    Drag the dashed line .. seek

Keys
    Space play/pause - Home go to start - Left/Right step one frame
    Shift+Left/Right step 1 s - F fit view - Ctrl+O open
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional, falls back to the stdlib wave module
    sf = None

try:
    import sounddevice as sd
    _SD_ERROR = None
except Exception as _exc:  # PortAudio missing, no audio subsystem, ...
    sd = None
    _SD_ERROR = _exc


APP_NAME = "Sound Analyzer"
DB_FLOOR = -200.0          # level used for digital silence
SNAP_RADIUS_PX = 10        # peak-snap search radius, in screen pixels
CELL_BUDGET = 40_000_000   # max spectrogram cells kept in memory (~160 MB float32)
N_HARMONICS = 10

Qt = QtCore.Qt

AUDIO_FILTER = (
    "Audio files (*.wav *.flac *.ogg *.oga *.opus *.mp3 *.aif *.aiff *.aifc "
    "*.w64 *.rf64 *.caf *.au *.snd);;All files (*)"
)

FFT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
OVERLAPS = [("0 %", 0.0), ("50 %", 0.5), ("75 %", 0.75), ("87.5 %", 0.875), ("93.75 %", 0.9375)]
ZERO_PADS = [1, 2, 4, 8]
MAX_FREQS = [("Full (Nyquist)", None), ("20 kHz", 20000), ("10 kHz", 10000), ("5 kHz", 5000),
             ("2 kHz", 2000), ("1 kHz", 1000), ("500 Hz", 500)]
AVERAGES = [1, 2, 4, 8, 16, 32]
WEIGHTINGS = [("Z (none)", "Z"), ("A", "A"), ("C", "C")]
FREQ_PRESETS = [("Full range", None), ("0 - 100 Hz", 100), ("0 - 200 Hz", 200), ("0 - 500 Hz", 500),
                ("0 - 1 kHz", 1000), ("0 - 2 kHz", 2000), ("0 - 5 kHz", 5000), ("0 - 10 kHz", 10000)]

# Generalised-cosine window coefficients (periodic / "DFT-even" form).
WINDOWS = {
    "Hann": (0.5, 0.5),
    "Hamming": (0.54, 0.46),
    "Blackman": (0.42, 0.5, 0.08),
    "Blackman-Harris": (0.35875, 0.48829, 0.14128, 0.01168),
    "Flat top": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
    "Rectangular": (1.0,),
}


# --------------------------------------------------------------------------------------
# Audio I/O
# --------------------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 samples shaped (n_samples, n_channels)."""
    if sf is not None:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(data), int(sr)
    return _load_wav_stdlib(path)


def _load_wav_stdlib(path: str) -> tuple[np.ndarray, int]:
    """Fallback PCM WAV reader used only when the soundfile package is missing."""
    import wave

    with wave.open(path, "rb") as w:
        ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        i = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        i = np.where(i >= 1 << 23, i - (1 << 24), i)
        a = i.astype(np.float32) / float(1 << 23)
    elif width == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {width} bytes")
    return a.reshape(-1, ch), int(sr)


def make_demo_signal(sr: int = 44100, duration: float = 30.0) -> np.ndarray:
    """Synthetic 4-cylinder engine run-up with a cabin boom, tyre cavity tone and road noise."""
    rng = np.random.default_rng(7)
    n = int(sr * duration)
    t = np.arange(n) / sr
    # RPM profile: idle, 1st gear pull, shift, 2nd gear pull, coast down, cruise, light accel.
    kt = [0, 2.0, 9.0, 9.5, 16.0, 19.0, 21.0, 26.0, 30.0]
    kr = [800, 800, 5000, 3000, 4800, 1500, 2300, 2100, 3200]
    rpm = np.interp(t, kt, kr)
    rot_hz = rpm / 60.0
    phase = 2 * np.pi * np.cumsum(rot_hz) / sr

    def boom(f):  # body-cavity resonance around 125 Hz
        return 1.0 + 4.0 * np.exp(-((f - 125.0) / 10.0) ** 2)

    x = np.zeros(n)
    for order, amp in [(0.5, 0.01), (1, 0.02), (2, 0.20), (4, 0.06), (6, 0.03), (8, 0.015), (27, 0.004)]:
        x += amp * boom(order * rot_hz) * np.sin(order * phase + rng.uniform(0, 2 * np.pi))
    speed = np.clip((rpm - 700) / 4300, 0.1, 1.0)
    x += 0.01 * speed * np.sin(2 * np.pi * 215.0 * t)                      # tyre cavity
    noise = np.fft.rfft(rng.standard_normal(n))                              # road noise
    f = np.fft.rfftfreq(n, 1 / sr)
    noise *= 1.0 / np.sqrt(1.0 + (f / 60.0) ** 2) / np.sqrt(1.0 + (f / 2000.0) ** 4)
    road = np.fft.irfft(noise, n)
    x += 0.15 * road / (np.abs(road).max() + 1e-12) * (0.4 + 0.6 * speed)
    x *= 0.5 / np.abs(x).max()
    return x.astype(np.float32)[:, None]


# --------------------------------------------------------------------------------------
# DSP
# --------------------------------------------------------------------------------------

def make_window(name: str, n: int) -> np.ndarray:
    coeffs = WINDOWS[name]
    k = np.arange(n) * (2.0 * np.pi / n)
    w = np.zeros(n)
    for i, a in enumerate(coeffs):
        w += ((-1) ** i) * a * np.cos(i * k)
    return w


def weighting_db(freqs: np.ndarray, kind: str) -> np.ndarray:
    """IEC 61672 A / C frequency weighting in dB (Z = flat)."""
    f = np.asarray(freqs, dtype=np.float64)
    if kind not in ("A", "C"):
        return np.zeros_like(f)
    f2 = f * f
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "A":
            r = (12194.0 ** 2 * f2 ** 2) / (
                (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
            w = 20.0 * np.log10(r) + 2.00
        else:
            r = (12194.0 ** 2 * f2) / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
            w = 20.0 * np.log10(r) + 0.06
    w = np.nan_to_num(w, nan=-150.0, neginf=-150.0)
    return np.maximum(w, -150.0)


class Cancelled(Exception):
    pass


@dataclass
class Spectrogram:
    db: np.ndarray          # (n_bins, n_frames) float32 levels, weighting + calibration applied in place
    sr: int
    nfft: int               # window length in samples
    zp: int                 # zero-padding factor
    hop: int                # samples between frames
    window: str
    enbw: float             # equivalent noise bandwidth of the window, in bins
    offset: np.ndarray      # per-bin dB offset currently baked into ``db``

    @property
    def df(self) -> float:
        return self.sr / (self.nfft * self.zp)

    @property
    def dt(self) -> float:
        return self.hop / self.sr

    @property
    def n_bins(self) -> int:
        return self.db.shape[0]

    @property
    def n_frames(self) -> int:
        return self.db.shape[1]

    @property
    def freqs(self) -> np.ndarray:
        return np.arange(self.n_bins) * self.df

    @property
    def f_max(self) -> float:
        return (self.n_bins - 1) * self.df

    @property
    def t_max(self) -> float:
        return (self.n_frames - 1) * self.dt


def compute_spectrogram(x, sr, nfft=8192, window="Hann", overlap=0.875, zp=1, fmax=None,
                        progress=None, cancelled=None, cell_budget=CELL_BUDGET) -> Spectrogram:
    """Short-time FFT magnitude in dBFS.

    Scaling: a full-scale sine reads 0 dB at its peak bin (amplitude spectrum,
    single-sided). Frames are centred on t = k * hop / sr (the signal is padded by
    half a window at both ends), so the spectrogram covers the full file.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.float32(x.mean())                       # remove DC offset (matters for low-freq work)
    half = nfft // 2
    xp = np.concatenate([np.zeros(half, np.float32), x, np.zeros(nfft - half, np.float32)])
    n_fft = nfft * zp
    df = sr / n_fft
    n_bins_full = n_fft // 2 + 1
    n_bins = n_bins_full if not fmax else min(n_bins_full, int(math.ceil(fmax / df)) + 2)

    hop = max(1, int(round(nfft * (1.0 - overlap))))
    n_frames = 1 + (len(xp) - nfft) // hop
    max_frames = max(256, int(cell_budget // n_bins))
    if n_frames > max_frames:  # very long file: widen the hop so memory stays bounded
        hop = int(math.ceil((len(xp) - nfft) / (max_frames - 1)))
        n_frames = 1 + (len(xp) - nfft) // hop

    win = make_window(window, nfft)
    enbw = float(nfft * np.sum(win ** 2) / np.sum(win) ** 2)
    scale = np.float32(2.0 / win.sum())
    win = win.astype(np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(xp, nfft)[::hop]
    out = np.empty((n_bins, n_frames), np.float32)
    chunk = max(1, (1 << 21) // n_fft)
    for i0 in range(0, n_frames, chunk):
        if cancelled is not None and cancelled():
            raise Cancelled()
        seg = frames[i0:i0 + chunk] * win
        spec = np.fft.rfft(seg, n=n_fft, axis=1)[:, :n_bins]
        mag = np.abs(spec).astype(np.float32, copy=False)
        mag *= scale
        mag[:, 0] *= 0.5                                    # DC is not doubled
        np.maximum(mag, np.float32(10 ** (DB_FLOOR / 20)), out=mag)
        out[:, i0:i0 + chunk] = (20.0 * np.log10(mag)).T
        if progress is not None:
            progress(int(100 * min(n_frames, i0 + chunk) / n_frames))

    return Spectrogram(db=out, sr=sr, nfft=nfft, zp=zp, hop=hop, window=window, enbw=enbw,
                       offset=np.zeros(n_bins, np.float32))


def maxpool2d(a: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Block-max downsampling. Keeps narrow tonal lines visible when zoomed out."""
    if fy == 1 and fx == 1:
        return a
    h, w = a.shape
    H, W = -(-h // fy), -(-w // fx)
    if H * fy != h or W * fx != w:
        p = np.full((H * fy, W * fx), -1e4, np.float32)
        p[:h, :w] = a
        a = p
    return a.reshape(H, fy, W, fx).max(axis=(1, 3))


# --------------------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------------------

class Player:
    """Callback-driven audio output. ``pos`` (in samples) is shared with the audio thread."""

    def __init__(self):
        self.buf: np.ndarray | None = None
        self.sr = 0
        self.pos = 0
        self.gain = 1.0
        self.loop: tuple[int, int] | None = None
        self.stream = None

    def set_buffer(self, buf: np.ndarray, sr: int):
        playing = self.is_playing()
        if playing:
            self.stop()
        self.buf = buf
        self.sr = sr
        self.pos = min(self.pos, len(buf))
        if playing:
            self.play()

    def is_playing(self) -> bool:
        return self.stream is not None and self.stream.active

    def play(self):
        if self.buf is None:
            return
        if sd is None:
            raise RuntimeError(f"The sounddevice / PortAudio library is not available:\n{_SD_ERROR}")
        self.stop()
        if self.loop is not None:
            s, e = self.loop
            if not (s <= self.pos < e):
                self.pos = s
        elif self.pos >= len(self.buf) - 1:
            self.pos = 0
        try:
            max_out = int(sd.query_devices(kind="output")["max_output_channels"])
        except Exception:
            max_out = 2
        channels = 2 if max_out >= 2 else 1
        self.stream = sd.OutputStream(samplerate=self.sr, channels=channels, dtype="float32",
                                      callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def position_samples(self) -> int:
        """Best estimate of the sample currently coming out of the speakers."""
        p = self.pos
        if self.is_playing():
            try:
                p -= int(self.stream.latency * self.sr)
            except Exception:
                pass
            if self.loop is not None and p < self.loop[0]:
                p += self.loop[1] - self.loop[0]
        return int(max(0, p))

    def _callback(self, outdata, frames, time_info, status):
        buf, pos, g, loop = self.buf, self.pos, self.gain, self.loop
        n_total = len(buf)
        start, end = loop if loop is not None else (0, n_total)
        written = 0
        while written < frames:
            if pos >= end or pos < start:
                if loop is None:
                    outdata[written:] = 0
                    self.pos = n_total
                    raise sd.CallbackStop()
                pos = start
            n = min(frames - written, end - pos)
            src = buf[pos:pos + n]
            out = outdata[written:written + n]
            if src.shape[1] == out.shape[1]:
                np.multiply(src, g, out=out)
            elif src.shape[1] == 1:
                out[:] = src * g                         # mono -> both speakers
            else:
                out[:, 0] = src.mean(axis=1) * g         # stereo -> mono device
            pos += n
            written += n
        np.clip(outdata, -1.0, 1.0, out=outdata)
        self.pos = pos


# --------------------------------------------------------------------------------------
# Qt helpers
# --------------------------------------------------------------------------------------

class NVHViewBox(pg.ViewBox):
    """ViewBox with modifier-key single-axis zoom and app-defined "view all" bounds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds_fn = None

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            mods = ev.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                axis = 0
            elif mods & Qt.KeyboardModifier.ShiftModifier:
                axis = 1
        super().wheelEvent(ev, axis)

    def childrenBounds(self, frac=None, orthoRange=(None, None), items=None):
        if self.bounds_fn is not None:
            b = self.bounds_fn()
            if b is not None:
                return b
        return super().childrenBounds(frac=frac, orthoRange=orthoRange, items=items)


class SpectrogramWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    done = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, gen, x, sr, params):
        super().__init__()
        self.gen, self.x, self.sr, self.params = gen, x, sr, params

    def run(self):
        try:
            res = compute_spectrogram(self.x, self.sr, progress=self.progress.emit,
                                      cancelled=self.isInterruptionRequested, **self.params)
        except Cancelled:
            return
        except Exception as exc:
            self.failed.emit(self.gen, f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(self.gen, res)


def make_colormap(name: str) -> pg.ColorMap:
    if name == "Analyzer (blue-red)":
        pos = [0.0, 0.25, 0.45, 0.6, 0.75, 0.88, 1.0]
        col = [(0, 20, 170), (0, 80, 255), (0, 220, 255), (40, 235, 90), (235, 240, 0), (255, 140, 0), (225, 0, 0)]
        return pg.ColorMap(pos, col)
    if name == "Jet":
        return pg.ColorMap([0, 0.125, 0.375, 0.625, 0.875, 1.0],
                           [(0, 0, 143), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)])
    if name == "Grayscale":
        return pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)])
    return pg.colormap.get(name.lower())


COLORMAPS = ["Analyzer (blue-red)", "Jet", "Turbo", "Viridis", "Inferno", "Magma", "Grayscale"]


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:06.3f}"


def fmt_freq(f: float) -> str:
    return f"{f:.1f} Hz" if f < 10000 else f"{f / 1000:.3f} kHz"


def combo(items, current=None) -> QtWidgets.QComboBox:
    c = QtWidgets.QComboBox()
    for label, data in items:
        c.addItem(label, data)
    if current is not None:
        i = c.findData(current)
        if i >= 0:
            c.setCurrentIndex(i)
    return c


# --------------------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.qs = QtCore.QSettings("SoundAnalyzer", "SoundAnalyzer")
        self.audio: np.ndarray | None = None
        self.sr = 0
        self.path: str | None = None
        self.spec: Spectrogram | None = None
        self.player = Player()

        self.hover_sg: tuple[float, float] | None = None   # (t, f) while the mouse is over the spectrogram
        self.hover_sp_f: float | None = None               # f while the mouse is over the spectrum
        self.pin_f = 100.0                                 # cursor frequency when not hovering

        self._syncing = False
        self._gen = 0
        self._worker = None
        self._workers: set[SpectrogramWorker] = set()
        self._first_view = False
        self._data_version = 0
        self._slice_key = None
        self._slice = None
        self._analysis_cache = None
        self._was_playing = False

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(15)
        self._render_timer.timeout.connect(self._render_image)

        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(150)
        self._recompute_timer.timeout.connect(self._recompute)

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(30)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._restore_settings()
        self._connect_settings()
        self._update_units()
        self._update_info()

    # ---------------------------------------------------------------- UI construction
    def _build_ui(self):
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)
        self.setAcceptDrops(True)

        # Toolbar --------------------------------------------------------------
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(18, 18))
        self.open_btn = self._make_open_button("Open audio file...", font_px=14, pad="6px 16px")
        tb.addWidget(self.open_btn)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("  Frequency view: "))
        self.preset_combo = combo(FREQ_PRESETS)
        self.preset_combo.setToolTip("Quick-zoom the frequency axis of both charts")
        self.preset_combo.activated.connect(self._apply_preset)
        tb.addWidget(self.preset_combo)
        a = tb.addAction("Fit all")
        a.setShortcut("F")
        a.setToolTip("Show the whole file and full frequency range (F)")
        a.triggered.connect(self.fit_view)
        self.box_action = tb.addAction("Box zoom")
        self.box_action.setCheckable(True)
        self.box_action.setToolTip("When on, left-drag draws a rectangle to zoom into")
        self.box_action.toggled.connect(self._set_box_zoom)
        a = tb.addAction("Auto levels")
        a.setToolTip("Fit the colour range (and spectrum dB axis) to what's currently visible")
        a.triggered.connect(self.auto_levels)
        tb.addSeparator()
        a = tb.addAction("Help")
        a.triggered.connect(self.show_help)

        # Plots ----------------------------------------------------------------
        self.sp_vb = NVHViewBox()
        self.spectrum_plot = pg.PlotWidget(viewBox=self.sp_vb)
        self.sg_vb = NVHViewBox()
        self.sgram_plot = pg.PlotWidget(viewBox=self.sg_vb)
        self._build_spectrum_plot()
        self._build_spectrogram_plot()

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.spectrum_plot)
        splitter.addWidget(self.sgram_plot)
        splitter.setSizes([380, 560])
        splitter.setChildrenCollapsible(False)

        # Transport bar ----------------------------------------------------------
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setMinimumWidth(170)
        self.play_btn.setMinimumHeight(40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setMinimumWidth(130)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setToolTip("Stop and return to the start (or the start of the loop)")
        self.stop_btn.clicked.connect(self.stop)
        self._style_play_button(False)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#3a3d41;color:#eee;font-size:15px;font-weight:bold;border-radius:4px}"
            "QPushButton:hover{background:#4a4d52}")

        self.time_label = QtWidgets.QLabel("00:00.000 / 00:00.000")
        f = QtGui.QFont("DejaVu Sans Mono")
        f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        f.setPointSize(13)
        self.time_label.setFont(f)
        self.loop_chk = QtWidgets.QCheckBox("Loop visible time range")
        self.loop_chk.setToolTip("Playback loops over the time span currently shown in the spectrogram")
        self.loop_chk.toggled.connect(self._update_loop)
        self.follow_chk = QtWidgets.QCheckBox("Auto-scroll")
        self.follow_chk.setChecked(True)
        self.follow_chk.setToolTip("Page the spectrogram view to keep the playhead visible during playback")
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(-30, 40)
        self.gain_spin.setSingleStep(3)
        self.gain_spin.setDecimals(0)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setToolTip("Playback gain only - does not affect the analysis (output is clipped at full scale)")
        self.gain_spin.valueChanged.connect(lambda v: setattr(self.player, "gain", float(10 ** (v / 20))))

        transport = QtWidgets.QHBoxLayout()
        transport.setContentsMargins(8, 6, 8, 6)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addSpacing(16)
        transport.addWidget(self.time_label)
        transport.addSpacing(16)
        transport.addWidget(self.loop_chk)
        transport.addWidget(self.follow_chk)
        transport.addStretch(1)
        transport.addWidget(QtWidgets.QLabel("Playback gain:"))
        transport.addWidget(self.gain_spin)

        # Start screen shown until a file is loaded: one big "open" button.
        welcome = QtWidgets.QWidget()
        welcome.setStyleSheet("background:#000;")
        wl = QtWidgets.QVBoxLayout(welcome)
        wl.addStretch(1)
        big_open = self._make_open_button("Open audio file...", font_px=20, pad="16px 40px")
        wl.addWidget(big_open, 0, Qt.AlignmentFlag.AlignHCenter)
        hint = QtWidgets.QLabel("or drag and drop a file onto this window\n\nWAV  -  FLAC  -  OGG  -  MP3  -  AIFF")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color:#9a9a9a; font-size:14px; margin-top:14px;")
        wl.addWidget(hint)
        wl.addStretch(1)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(welcome)     # page 0: nothing loaded yet
        self.stack.addWidget(splitter)    # page 1: the charts

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.stack, 1)
        v.addLayout(transport)
        self.setCentralWidget(central)

        # Settings dock ------------------------------------------------------------
        self._build_settings_dock()

        # Status bar ------------------------------------------------------------------
        self.file_label = QtWidgets.QLabel("Open a file (Ctrl+O) or drop one on the window")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setMaximumHeight(14)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.file_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

        # Shortcuts -------------------------------------------------------------------
        def sc(key, fn):
            s = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            s.setContext(Qt.ShortcutContext.WindowShortcut)
            s.activated.connect(fn)
        sc(QtGui.QKeySequence.StandardKey.Open, self.open_dialog)
        sc("Space", self.toggle_play)
        sc("Home", lambda: self.seek(self._loop_start()))
        sc("Left", lambda: self.step(-1))
        sc("Right", lambda: self.step(+1))
        sc("Shift+Left", lambda: self.step_seconds(-1.0))
        sc("Shift+Right", lambda: self.step_seconds(+1.0))

    def _make_open_button(self, text: str, font_px: int, pad: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon), "  " + text)
        btn.setIconSize(QtCore.QSize(font_px + 4, font_px + 4))
        btn.setToolTip("Choose an audio file to analyse (Ctrl+O)")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton{{background:#1f6fd1;color:white;font-size:{font_px}px;font-weight:bold;"
            f"padding:{pad};border-radius:5px}}"
            "QPushButton:hover{background:#2f82ea} QPushButton:pressed{background:#1859a8}")
        btn.clicked.connect(self.open_dialog)
        return btn

    def _style_play_button(self, playing: bool):
        if playing:
            self.play_btn.setText("Pause")
            bg, hov = "#b26a00", "#c87a10"
        else:
            self.play_btn.setText("Play")
            bg, hov = "#1e7d32", "#2b9a42"
        self.play_btn.setStyleSheet(
            f"QPushButton{{background:{bg};color:white;font-size:15px;font-weight:bold;border-radius:4px}}"
            f"QPushButton:hover{{background:{hov}}}")

    def _make_label(self, fill) -> pg.TextItem:
        t = pg.TextItem("", color=(255, 255, 255), fill=pg.mkBrush(fill), border=pg.mkPen(None), anchor=(0, 1))
        f = QtGui.QFont()
        f.setPointSize(12)
        t.setFont(f)
        t.setZValue(100)
        return t

    def _build_spectrum_plot(self):
        pi = self.spectrum_plot.getPlotItem()
        pi.hideButtons()
        pi.disableAutoRange()
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setLabel("bottom", "Frequency", units="Hz")
        pi.setLabel("left", "Level (dBFS)")
        pi.setTitle("")
        self.spectrum_plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.spec_curve = pi.plot(pen=pg.mkPen("#3cff3c", width=1.5))
        self.spec_curve.setDownsampling(auto=True, method="peak")
        self.spec_curve.setClipToView(True)

        self.sp_vline = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 80, 80, 130), width=1, style=Qt.PenStyle.DashLine))
        self.sp_marker = pg.ScatterPlotItem(size=22, symbol="+", pen=pg.mkPen(None), brush=pg.mkBrush("w"))
        self.sp_label = self._make_label((215, 25, 25, 235))
        for it in (self.sp_vline, self.sp_marker, self.sp_label):
            pi.addItem(it, ignoreBounds=True)
            it.setVisible(False)

        self.sp_harm = []
        for n in range(2, N_HARMONICS + 2):
            ln = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 170, 0, 140), width=1, style=Qt.PenStyle.DotLine),
                                 label=f"{n}x", labelOpts={"position": 0.82, "color": (255, 170, 0)})
            ln.setVisible(False)
            pi.addItem(ln, ignoreBounds=True)
            self.sp_harm.append(ln)

        # Band RMS box, pinned to the top-right corner of the plot area.
        self.rms_label = pg.TextItem("", color=(255, 255, 255), fill=pg.mkBrush(110, 110, 110, 220), anchor=(1, 0))
        f = QtGui.QFont()
        f.setPointSize(11)
        self.rms_label.setFont(f)
        self.rms_label.setParentItem(self.sp_vb)
        self.rms_label.setZValue(100)
        self.sp_vb.sigResized.connect(self._place_rms_label)

        self.sp_vb.bounds_fn = self._spectrum_bounds
        self.sp_vb.sigXRangeChanged.connect(self._sp_freq_range_changed)
        self._proxy_sp = pg.SignalProxy(self.spectrum_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_sp_mouse)
        self.spectrum_plot.scene().sigMouseClicked.connect(self._on_sp_click)
        self.spectrum_plot.viewport().installEventFilter(self)

    def _build_spectrogram_plot(self):
        pi = self.sgram_plot.getPlotItem()
        pi.hideButtons()
        pi.disableAutoRange()
        pi.setLabel("bottom", "Time", units="s")
        pi.setLabel("left", "Frequency", units="Hz")
        pi.setTitle("")
        self.sgram_plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.img = pg.ImageItem()
        pi.addItem(self.img)
        self.cbar = pg.ColorBarItem(values=(-100, -20), colorMap=make_colormap(COLORMAPS[0]),
                                    interactive=True, width=16, rounding=0.5, colorMapMenu=False)
        self.cbar.setImageItem(self.img, insert_in=pi)
        self.cbar.sigLevelsChanged.connect(self._on_levels_changed)

        self.playhead = pg.InfiniteLine(pos=0, angle=90, movable=True,
                                        pen=pg.mkPen((255, 255, 255, 220), width=1.5, style=Qt.PenStyle.DashLine),
                                        hoverPen=pg.mkPen((255, 255, 0), width=2.5))
        self.playhead.setZValue(50)
        self.playhead.sigDragged.connect(self._on_playhead_dragged)
        pi.addItem(self.playhead, ignoreBounds=True)

        cross_pen = pg.mkPen((255, 255, 255, 110), width=1, style=Qt.PenStyle.DotLine)
        self.sg_vline = pg.InfiniteLine(angle=90, pen=cross_pen)
        self.sg_hline = pg.InfiniteLine(angle=0, pen=cross_pen)
        self.sg_marker = pg.ScatterPlotItem(size=22, symbol="+", pen=pg.mkPen(None), brush=pg.mkBrush("w"))
        self.sg_label = self._make_label((95, 95, 95, 225))
        for it in (self.sg_vline, self.sg_hline, self.sg_marker, self.sg_label):
            pi.addItem(it, ignoreBounds=True)
            it.setVisible(False)

        self.sg_harm = []
        for n in range(2, N_HARMONICS + 2):
            ln = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 170, 0, 150), width=1, style=Qt.PenStyle.DotLine),
                                 label=f"{n}x", labelOpts={"position": 0.02, "color": (255, 170, 0)})
            ln.setVisible(False)
            pi.addItem(ln, ignoreBounds=True)
            self.sg_harm.append(ln)

        self.sg_vb.bounds_fn = self._spectrogram_bounds
        self.sg_vb.sigRangeChanged.connect(lambda *_: self._render_timer.start())
        self.sg_vb.sigResized.connect(lambda *_: self._render_timer.start())
        self.sg_vb.sigYRangeChanged.connect(self._sg_freq_range_changed)
        self.sg_vb.sigXRangeChanged.connect(lambda *_: self._update_loop())
        self._proxy_sg = pg.SignalProxy(self.sgram_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_sg_mouse)
        self.sgram_plot.scene().sigMouseClicked.connect(self._on_sg_click)
        self.sgram_plot.viewport().installEventFilter(self)

    def _build_settings_dock(self):
        dock = QtWidgets.QDockWidget("Settings", self)
        dock.setObjectName("settings_dock")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("Analysis")
        f1 = QtWidgets.QFormLayout(g1)
        self.channel_combo = combo([("Mono", 0)])
        self.fft_combo = combo([(str(n), n) for n in FFT_SIZES], 8192)
        self.window_combo = combo([(k, k) for k in WINDOWS], "Hann")
        self.overlap_combo = combo(OVERLAPS, 0.875)
        self.zp_combo = combo([(f"{z}x", z) for z in ZERO_PADS], 1)
        self.fmax_combo = combo(MAX_FREQS, None)
        self.fft_combo.setToolTip("Longer FFT = finer frequency resolution, coarser time resolution")
        self.zp_combo.setToolTip("Zero padding interpolates the spectrum (smoother peaks, more precise\n"
                                 "peak frequency readout). It does not add real resolution.")
        self.fmax_combo.setToolTip("Only keep frequencies up to this limit. Lower = less memory and\n"
                                   "finer time steps on long recordings.")
        self.channel_combo.setToolTip("Channel to analyse (playback follows this choice)")
        f1.addRow("Channel", self.channel_combo)
        f1.addRow("FFT size", self.fft_combo)
        f1.addRow("Window", self.window_combo)
        f1.addRow("Overlap", self.overlap_combo)
        f1.addRow("Zero padding", self.zp_combo)
        f1.addRow("Max frequency", self.fmax_combo)
        self.res_label = QtWidgets.QLabel("")
        self.res_label.setWordWrap(True)
        self.res_label.setStyleSheet("color:#9ab;")
        f1.addRow(self.res_label)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("Levels")
        f2 = QtWidgets.QFormLayout(g2)
        self.avg_combo = combo([(f"{n} frame{'s' if n > 1 else ''}", n) for n in AVERAGES], 1)
        self.avg_combo.setToolTip("Linear (power) averaging of adjacent spectrogram frames for the spectrum chart")
        self.weight_combo = combo(WEIGHTINGS, "Z")
        self.cal_spin = QtWidgets.QDoubleSpinBox()
        self.cal_spin.setRange(-200, 200)
        self.cal_spin.setDecimals(1)
        self.cal_spin.setSuffix(" dB")
        self.cal_spin.setToolTip("Added to every level. If you know your mic/recorder calibration\n"
                                 "(dB SPL at 0 dBFS), enter it here to read dB SPL directly.")
        self.lo_spin = QtWidgets.QDoubleSpinBox()
        self.hi_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.lo_spin, self.hi_spin):
            s.setRange(-300, 300)
            s.setDecimals(1)
            s.setSingleStep(5)
            s.setSuffix(" dB")
            s.valueChanged.connect(self._on_level_spins)
        f2.addRow("Spectrum averaging", self.avg_combo)
        f2.addRow("Weighting", self.weight_combo)
        f2.addRow("Calibration offset", self.cal_spin)
        f2.addRow("Colour range min", self.lo_spin)
        f2.addRow("Colour range max", self.hi_spin)
        lay.addWidget(g2)

        g3 = QtWidgets.QGroupBox("Display / cursor")
        f3 = QtWidgets.QFormLayout(g3)
        self.cmap_combo = combo([(c, c) for c in COLORMAPS], COLORMAPS[0])
        self.snap_chk = QtWidgets.QCheckBox("Snap cursor to nearest peak")
        self.snap_chk.setChecked(True)
        self.harm_chk = QtWidgets.QCheckBox("Show harmonics of cursor (2x, 3x, ...)")
        self.link_chk = QtWidgets.QCheckBox("Link frequency axes of both charts")
        self.link_chk.setChecked(True)
        f3.addRow("Colour map", self.cmap_combo)
        f3.addRow(self.snap_chk)
        f3.addRow(self.harm_chk)
        f3.addRow(self.link_chk)
        lay.addWidget(g3)
        lay.addStretch(1)

        dock.setWidget(w)
        dock.setMinimumWidth(270)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_settings(self):
        for c in (self.fft_combo, self.window_combo, self.overlap_combo, self.zp_combo, self.fmax_combo):
            c.currentIndexChanged.connect(self._schedule_recompute)
            c.currentIndexChanged.connect(self._update_info)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.avg_combo.currentIndexChanged.connect(self._on_display_setting)
        self.weight_combo.currentIndexChanged.connect(self._on_offsets_changed)
        self.cal_spin.valueChanged.connect(self._on_offsets_changed)
        self.cmap_combo.currentIndexChanged.connect(
            lambda *_: self.cbar.setColorMap(make_colormap(self.cmap_combo.currentData())))
        self.snap_chk.toggled.connect(self._on_display_setting)
        self.harm_chk.toggled.connect(self._on_display_setting)
        self.link_chk.toggled.connect(self._on_link_toggled)

    # ---------------------------------------------------------------- settings persistence
    def _restore_settings(self):
        def set_combo(c, key, cast):
            v = self.qs.value(key)
            if v is None:
                return
            try:
                v = None if v in ("None", "") else cast(v)
            except (TypeError, ValueError):
                return
            i = c.findData(v)
            if i >= 0:
                c.setCurrentIndex(i)
        set_combo(self.fft_combo, "fft", int)
        set_combo(self.window_combo, "window", str)
        set_combo(self.overlap_combo, "overlap", float)
        set_combo(self.zp_combo, "zp", int)
        set_combo(self.fmax_combo, "fmax", int)
        set_combo(self.avg_combo, "avg", int)
        set_combo(self.weight_combo, "weighting", str)
        set_combo(self.cmap_combo, "cmap", str)
        set_combo(self.preset_combo, "preset", int)
        self.cbar.setColorMap(make_colormap(self.cmap_combo.currentData()))
        for chk, key in ((self.snap_chk, "snap"), (self.harm_chk, "harmonics"), (self.link_chk, "link"),
                         (self.follow_chk, "follow")):
            v = self.qs.value(key)
            if v is not None:
                chk.setChecked(str(v).lower() in ("true", "1"))
        try:
            self.cal_spin.setValue(float(self.qs.value("cal", 0.0)))
            self.gain_spin.setValue(float(self.qs.value("gain", 0.0)))
        except (TypeError, ValueError):
            pass
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def _save_settings(self):
        qs = self.qs
        qs.setValue("fft", self.fft_combo.currentData())
        qs.setValue("window", self.window_combo.currentData())
        qs.setValue("overlap", self.overlap_combo.currentData())
        qs.setValue("zp", self.zp_combo.currentData())
        qs.setValue("fmax", str(self.fmax_combo.currentData()))
        qs.setValue("avg", self.avg_combo.currentData())
        qs.setValue("weighting", self.weight_combo.currentData())
        qs.setValue("cmap", self.cmap_combo.currentData())
        qs.setValue("preset", str(self.preset_combo.currentData()))
        qs.setValue("snap", self.snap_chk.isChecked())
        qs.setValue("harmonics", self.harm_chk.isChecked())
        qs.setValue("link", self.link_chk.isChecked())
        qs.setValue("follow", self.follow_chk.isChecked())
        qs.setValue("cal", self.cal_spin.value())
        qs.setValue("gain", self.gain_spin.value())
        qs.setValue("geometry", self.saveGeometry())

    # ---------------------------------------------------------------- file loading
    def open_dialog(self):
        start = self.qs.value("last_dir", os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open audio file", str(start), AUDIO_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            data, sr = load_audio(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open file",
                                           f"{os.path.basename(path)}\n\n{type(exc).__name__}: {exc}")
            return
        if data.size == 0:
            QtWidgets.QMessageBox.warning(self, "Empty file", "The file contains no audio samples.")
            return
        self.qs.setValue("last_dir", os.path.dirname(os.path.abspath(path)))
        self.set_audio(data, sr, os.path.basename(path))
        self.path = path

    def set_audio(self, data: np.ndarray, sr: int, name: str):
        self.player.stop()
        self._play_timer.stop()
        self._style_play_button(False)
        self.stack.setCurrentIndex(1)
        self.audio = np.ascontiguousarray(data, dtype=np.float32)
        self.sr = int(sr)
        self._analysis_cache = None
        self.spec = None
        self._data_version += 1
        self.player.pos = 0
        self.playhead.setValue(0)

        n_ch = self.audio.shape[1]
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if n_ch == 1:
            self.channel_combo.addItem("Mono", 0)
        else:
            self.channel_combo.addItem(f"Mix of all {n_ch} channels", -1)
            for c in range(n_ch):
                self.channel_combo.addItem(f"Channel {c + 1}", c)
        self.channel_combo.blockSignals(False)

        dur = len(self.audio) / self.sr
        self.setWindowTitle(f"{name} - {APP_NAME}")
        self.file_label.setText(f"{name}   |   {self.sr} Hz   |   {n_ch} ch   |   {fmt_time(dur)}  ({dur:.2f} s)")
        self.playhead.setBounds([0, dur])
        self.player.set_buffer(self._playback_buffer(), self.sr)
        self._update_time_label()
        self._first_view = True
        self._update_info()
        self._recompute()

    def _channel(self) -> int:
        c = self.channel_combo.currentData()
        return 0 if c is None else int(c)

    def _analysis_signal(self) -> np.ndarray:
        ch = self._channel()
        if self._analysis_cache is not None and self._analysis_cache[0] == ch:
            return self._analysis_cache[1]
        a = self.audio
        x = a[:, 0] if a.shape[1] == 1 else (a.mean(axis=1) if ch < 0 else a[:, ch])
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._analysis_cache = (ch, x)
        return x

    def _playback_buffer(self) -> np.ndarray:
        a = self.audio
        ch = self._channel()
        if a.shape[1] == 1:
            return a
        if ch < 0:
            return a if a.shape[1] == 2 else a.mean(axis=1, keepdims=True).astype(np.float32)
        return a[:, ch:ch + 1]

    def _on_channel_changed(self):
        if self.audio is None:
            return
        self.player.set_buffer(self._playback_buffer(), self.sr)
        self._schedule_recompute()

    # ---------------------------------------------------------------- spectrogram computation
    def _params(self) -> dict:
        return dict(nfft=int(self.fft_combo.currentData()), window=self.window_combo.currentData(),
                    overlap=float(self.overlap_combo.currentData()), zp=int(self.zp_combo.currentData()),
                    fmax=self.fmax_combo.currentData())

    def _schedule_recompute(self, *_):
        if self.audio is not None:
            self._recompute_timer.start()

    def _recompute(self):
        if self.audio is None:
            return
        self._gen += 1
        if self._worker is not None:
            self._worker.requestInterruption()
        w = SpectrogramWorker(self._gen, self._analysis_signal(), self.sr, self._params())
        w.progress.connect(self.progress.setValue)
        w.done.connect(self._on_spec_done)
        w.failed.connect(self._on_spec_failed)
        w.finished.connect(lambda w=w: self._workers.discard(w))
        self._workers.add(w)
        self._worker = w
        self.progress.setValue(0)
        self.progress.setVisible(True)
        w.start()

    def _on_spec_failed(self, gen, msg):
        if gen != self._gen:
            return
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Analysis failed", msg)

    def _on_spec_done(self, gen, spec: Spectrogram):
        if gen != self._gen:
            return
        self.progress.setVisible(False)
        self._apply_offsets(spec)
        self.spec = spec
        self._data_version += 1

        dur = len(self.audio) / self.sr
        self.sg_vb.setLimits(xMin=-0.02 * dur - spec.dt, xMax=dur * 1.02 + spec.dt,
                             yMin=-spec.df, yMax=spec.f_max + spec.df,
                             minXRange=spec.dt * 2, minYRange=spec.df * 2)
        self.sp_vb.setLimits(xMin=-spec.df, xMax=spec.f_max + spec.df, yMin=-300, yMax=300,
                             minXRange=spec.df * 2, minYRange=0.5)
        if self._first_view:
            self._first_view = False
            self.pin_f = min(100.0, spec.f_max / 2)
            self.sg_vb.setRange(xRange=(0, dur), yRange=(0, spec.f_max), padding=0)
            self._apply_preset()
            self.auto_levels()
        else:
            # keep the user's view, but clamp it to the new data extent
            (x0, x1), (y0, y1) = self.sg_vb.viewRange()
            if y0 > spec.f_max or y1 > spec.f_max + spec.df:
                self.sg_vb.setYRange(max(0.0, min(y0, spec.f_max * 0.9)), spec.f_max, padding=0)
        self._update_info()
        self._render_image()
        self.refresh_cursor()

    def _offset_vector(self, freqs) -> np.ndarray:
        w = weighting_db(freqs, self.weight_combo.currentData())
        return (w + self.cal_spin.value()).astype(np.float32)

    def _apply_offsets(self, spec: Spectrogram):
        new = self._offset_vector(spec.freqs)
        delta = new - spec.offset
        if np.any(delta != 0):
            spec.db += delta[:, None]
            spec.offset = new

    def _on_offsets_changed(self, *_):
        self._update_units()
        if self.spec is None:
            return
        self._apply_offsets(self.spec)
        self._data_version += 1
        self._render_image()
        self.refresh_cursor()

    def _on_display_setting(self, *_):
        self._update_info()
        self.refresh_cursor()

    def _update_units(self):
        w = self.weight_combo.currentData()
        base = "dBFS" if self.cal_spin.value() == 0 else "dB"
        unit = base if w == "Z" else f"{base}({w})"
        self._unit = unit
        self.cbar.axis.setLabel(unit)
        self.spectrum_plot.getPlotItem().setLabel("left", f"Level ({unit})")

    def _update_info(self, *_):
        p = self._params()
        sr = self.sr or 44100
        df = sr / (p["nfft"] * p["zp"])
        win_ms = 1000 * p["nfft"] / sr
        hop = max(1, int(round(p["nfft"] * (1 - p["overlap"]))))
        txt = (f"Resolution: {df:.2f} Hz per bin\nWindow length: {win_ms:.0f} ms\n"
               f"Time step: {1000 * hop / sr:.1f} ms")
        if self.spec is not None and self.spec.hop != hop:
            txt += f"\n(long file: time step widened to {1000 * self.spec.dt:.1f} ms\n to stay within memory)"
        self.res_label.setText(txt)
        if self.spec is not None:
            s = self.spec
            avg = self.avg_combo.currentData()
            w = self.weight_combo.currentData()
            self.spectrum_plot.getPlotItem().setTitle(
                f"{s.sr} Hz;  {s.window};  FFT size: {s.nfft};  zero padding: {s.zp};  lin. averaging: {avg};  "
                f"df = {s.df:.2f} Hz;  weighting: {w};  cal: {self.cal_spin.value():+.1f} dB",
                color="#cccccc", size="9pt")
            self.sgram_plot.getPlotItem().setTitle(
                f"{s.sr} Hz;  {s.window};  FFT size: {s.nfft};  overlap: {100 * (1 - s.hop / s.nfft):.1f} %;  "
                f"dt = {1000 * s.dt:.1f} ms;  {s.n_frames} frames x {s.n_bins} bins",
                color="#cccccc", size="9pt")

    # ---------------------------------------------------------------- spectrogram image (viewport rendering)
    def _render_image(self):
        """Render only the visible part of the spectrogram, max-pooled to screen resolution.

        This keeps panning/zooming fast for big files and makes sure narrow tonal
        lines stay visible when zoomed out (a plain resize would skip them).
        """
        s = self.spec
        if s is None:
            self.img.clear()
            return
        (x0, x1), (y0, y1) = self.sg_vb.viewRange()
        dt, df = s.dt, s.df
        nb, nf = s.db.shape
        mx, my = 0.5 * (x1 - x0), 0.5 * (y1 - y0)       # render a margin so short pans look seamless
        c0 = int(np.clip(math.floor((x0 - mx) / dt + 0.5), 0, nf))
        c1 = int(np.clip(math.ceil((x1 + mx) / dt + 0.5), 0, nf))
        r0 = int(np.clip(math.floor((y0 - my) / df + 0.5), 0, nb))
        r1 = int(np.clip(math.ceil((y1 + my) / df + 0.5), 0, nb))
        if c1 <= c0 or r1 <= r0:
            self.img.clear()
            return
        dpr = self.sgram_plot.devicePixelRatioF()
        pw = max(1.0, self.sg_vb.width() * dpr)
        ph = max(1.0, self.sg_vb.height() * dpr)
        cpx = (x1 - x0) / dt / pw                        # data cells per screen pixel
        rpx = (y1 - y0) / df / ph
        fx = max(1, int(math.ceil(cpx))) if cpx > 1.0 else 1
        fy = max(1, int(math.ceil(rpx))) if rpx > 1.0 else 1
        sub = maxpool2d(s.db[r0:r1, c0:c1], fy, fx)
        self.img.setImage(sub, autoLevels=False, levels=self.cbar.levels())
        self.img.setRect(QtCore.QRectF(c0 * dt - dt / 2, r0 * df - df / 2,
                                       sub.shape[1] * fx * dt, sub.shape[0] * fy * df))

    # ---------------------------------------------------------------- levels
    def auto_levels(self):
        if self.spec is None:
            return
        s = self.spec
        (x0, x1), (y0, y1) = self.sg_vb.viewRange()
        c0, c1 = int(max(0, x0 / s.dt)), int(min(s.n_frames, x1 / s.dt + 1))
        r0, r1 = int(max(1, y0 / s.df)), int(min(s.n_bins, y1 / s.df + 1))   # skip the DC bin
        sub = s.db[r0:r1, c0:c1] if (c1 > c0 and r1 > r0) else s.db
        step = max(1, int(math.sqrt(sub.size / 400_000)))
        sample = sub[::step, ::step]
        hi = float(np.max(sub))
        lo = float(np.percentile(sample, 20))
        hi = math.ceil(hi / 5.0) * 5.0
        lo = max(math.floor(lo / 5.0) * 5.0, hi - 90.0)
        if hi - lo < 20:
            lo = hi - 20
        self.cbar.setLevels((lo, hi))
        self._on_levels_changed()
        self.sp_vb.setYRange(lo, hi + 5, padding=0)

    def _on_levels_changed(self, *_):
        lo, hi = self.cbar.levels()
        for s, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)

    def _on_level_spins(self, *_):
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if hi > lo:
            self.cbar.setLevels((lo, hi))

    # ---------------------------------------------------------------- view helpers
    def _spectrogram_bounds(self):
        if self.spec is None or self.audio is None:
            return None
        return [[0.0, len(self.audio) / self.sr], [0.0, self.spec.f_max]]

    def _spectrum_bounds(self):
        if self.spec is None:
            return None
        lo, hi = self.cbar.levels()
        return [[0.0, self.spec.f_max], [lo, hi + 5]]

    def fit_view(self):
        if self.spec is None:
            return
        self.sg_vb.setRange(xRange=(0, len(self.audio) / self.sr), yRange=(0, self.spec.f_max), padding=0)
        self.sp_vb.setXRange(0, self.spec.f_max, padding=0)
        lo, hi = self.cbar.levels()
        self.sp_vb.setYRange(lo, hi + 5, padding=0)

    def _apply_preset(self, *_):
        if self.spec is None:
            return
        f = self.preset_combo.currentData()
        f1 = self.spec.f_max if f is None else min(float(f), self.spec.f_max)
        self.sg_vb.setYRange(0, f1, padding=0)
        if not self.link_chk.isChecked():
            self.sp_vb.setXRange(0, f1, padding=0)

    def _set_box_zoom(self, on: bool):
        mode = pg.ViewBox.RectMode if on else pg.ViewBox.PanMode
        self.sg_vb.setMouseMode(mode)
        self.sp_vb.setMouseMode(mode)

    def _sg_freq_range_changed(self, vb, rng):
        if self._syncing or not self.link_chk.isChecked():
            return
        self._syncing = True
        try:
            self.sp_vb.setXRange(rng[0], rng[1], padding=0)
        finally:
            self._syncing = False

    def _sp_freq_range_changed(self, vb, rng):
        self._update_rms()
        if self._syncing or not self.link_chk.isChecked():
            return
        self._syncing = True
        try:
            self.sg_vb.setYRange(rng[0], rng[1], padding=0)
        finally:
            self._syncing = False

    def _on_link_toggled(self, on):
        if on:
            self._sg_freq_range_changed(self.sg_vb, self.sg_vb.viewRange()[1])

    # ---------------------------------------------------------------- mouse handling
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Leave:
            if obj is self.sgram_plot.viewport() and self.hover_sg is not None:
                self.hover_sg = None
                self.refresh_cursor()
            elif obj is self.spectrum_plot.viewport() and self.hover_sp_f is not None:
                self.hover_sp_f = None
                self.refresh_cursor()
        return super().eventFilter(obj, ev)

    def _on_sg_mouse(self, evt):
        pos = evt[0]
        inside = (self.spec is not None and self.sgram_plot.underMouse()
                  and self.sg_vb.sceneBoundingRect().contains(pos))
        if not inside:
            if self.hover_sg is not None:
                self.hover_sg = None
                self.refresh_cursor()
            return
        p = self.sg_vb.mapSceneToView(pos)
        dur = len(self.audio) / self.sr
        self.hover_sg = (float(np.clip(p.x(), 0, dur)), float(np.clip(p.y(), 0, self.spec.f_max)))
        self.refresh_cursor()

    def _on_sp_mouse(self, evt):
        pos = evt[0]
        inside = (self.spec is not None and self.spectrum_plot.underMouse()
                  and self.sp_vb.sceneBoundingRect().contains(pos))
        if not inside:
            if self.hover_sp_f is not None:
                self.hover_sp_f = None
                self.refresh_cursor()
            return
        p = self.sp_vb.mapSceneToView(pos)
        self.hover_sp_f = float(np.clip(p.x(), 0, self.spec.f_max))
        self.refresh_cursor()

    def _on_sg_click(self, ev):
        if self.spec is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        if not self.sg_vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.sg_vb.mapSceneToView(ev.scenePos())
        dur = len(self.audio) / self.sr
        t = float(np.clip(p.x(), 0, dur))
        f = float(np.clip(p.y(), 0, self.spec.f_max))
        j = self._frame_index(t)
        k = self._snap(self.spec.db[:, j], self._bin_index(f), self.sg_vb, axis=1)
        self.pin_f = k * self.spec.df
        self.seek(t)

    def _on_sp_click(self, ev):
        if self.spec is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        if not self.sp_vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.sp_vb.mapSceneToView(ev.scenePos())
        f = float(np.clip(p.x(), 0, self.spec.f_max))
        spectrum = self._spectrum_at(self._frame_index(self._playhead_time()))
        k = self._snap(spectrum, self._bin_index(f), self.sp_vb, axis=0)
        self.pin_f = k * self.spec.df
        self.refresh_cursor()

    def _on_playhead_dragged(self, line):
        if self.audio is None:
            return
        self.player.pos = int(line.value() * self.sr)
        self._update_time_label()
        self.refresh_cursor()

    # ---------------------------------------------------------------- cursor / spectrum
    def _frame_index(self, t: float) -> int:
        return int(np.clip(round(t / self.spec.dt), 0, self.spec.n_frames - 1))

    def _bin_index(self, f: float) -> int:
        return int(np.clip(round(f / self.spec.df), 0, self.spec.n_bins - 1))

    def _snap(self, arr: np.ndarray, k: int, vb: pg.ViewBox, axis: int) -> int:
        if not self.snap_chk.isChecked():
            return k
        rng = vb.viewRange()[axis]
        px = vb.width() if axis == 0 else vb.height()
        bins_per_px = (rng[1] - rng[0]) / self.spec.df / max(px, 1.0)
        w = max(1, int(round(SNAP_RADIUS_PX * bins_per_px)))
        lo, hi = max(0, k - w), min(len(arr), k + w + 1)
        return lo + int(np.argmax(arr[lo:hi]))

    def _spectrum_at(self, j: int) -> np.ndarray:
        n = int(self.avg_combo.currentData())
        key = (j, n, self._data_version)
        if key == self._slice_key:
            return self._slice
        db = self.spec.db
        if n <= 1:
            out = db[:, j]
        else:
            j0 = int(np.clip(j - n // 2, 0, max(0, self.spec.n_frames - n)))
            p = np.power(10.0, db[:, j0:j0 + n].astype(np.float64) / 10.0).mean(axis=1)
            out = (10.0 * np.log10(np.maximum(p, 1e-30))).astype(np.float32)
        self._slice_key, self._slice = key, out
        return out

    def _playhead_time(self) -> float:
        return self.player.position_samples() / self.sr if self.sr else 0.0

    def refresh_cursor(self):
        """Single place that updates both cursors, both labels and the spectrum curve."""
        s = self.spec
        visible = s is not None
        for it in (self.sp_vline, self.sp_marker, self.sp_label, self.sg_hline, self.sg_marker, self.sg_label):
            it.setVisible(visible)
        if not visible:
            self.sg_vline.setVisible(False)
            self.spec_curve.setData([], [])
            return

        if self.hover_sg is not None:
            t, f = self.hover_sg
            source = "sg"
        else:
            t = self._playhead_time()
            f = self.hover_sp_f if self.hover_sp_f is not None else self.pin_f
            source = "sp" if self.hover_sp_f is not None else "pin"

        j = self._frame_index(t)
        spectrum = self._spectrum_at(j)
        self.spec_curve.setData(s.freqs, spectrum)

        k = self._bin_index(f)
        if source == "sg":
            k = self._snap(s.db[:, j], k, self.sg_vb, axis=1)
        elif source == "sp":
            k = self._snap(spectrum, k, self.sp_vb, axis=0)
        fk = k * s.df

        # Spectrogram crosshair + label
        self.sg_vline.setVisible(source == "sg")
        self.sg_vline.setValue(t)
        self.sg_hline.setValue(fk)
        self.sg_marker.setData([t], [fk])
        self.sg_label.setText(f"{fmt_freq(fk)}, {t:.2f} s\n{s.db[k, j]:.1f} {self._unit}")
        self._place_label(self.sg_label, self.sg_vb, t, fk)

        # Spectrum crosshair + label
        val = float(spectrum[k])
        self.sp_vline.setValue(fk)
        self.sp_marker.setData([fk], [val])
        self.sp_label.setText(f"{fmt_freq(fk)}   {val:.1f} {self._unit}")
        self._place_label(self.sp_label, self.sp_vb, fk, val)

        # Harmonic markers
        show_h = self.harm_chk.isChecked() and fk > 0
        for n, (a, b) in enumerate(zip(self.sp_harm, self.sg_harm), start=2):
            on = show_h and n * fk <= s.f_max
            a.setVisible(on)
            b.setVisible(on)
            if on:
                a.setValue(n * fk)
                b.setValue(n * fk)

        self._update_rms()

    def _place_label(self, label: pg.TextItem, vb: pg.ViewBox, x: float, y: float):
        """Put the label above-right of the point, flipping near the plot edges."""
        (x0, x1), (y0, y1) = vb.viewRange()
        ax = 1.06 if (x - x0) > 0.65 * (x1 - x0) else -0.06
        ay = -0.2 if (y - y0) > 0.8 * (y1 - y0) else 1.2
        label.setAnchor((ax, ay))
        label.setPos(x, y)

    def _update_rms(self):
        if self.spec is None or self._slice is None:
            self.rms_label.setText("")
            return
        s = self.spec
        (f0, f1), _ = self.sp_vb.viewRange()
        k0 = int(np.clip(math.ceil(f0 / s.df), 0, s.n_bins))
        k1 = int(np.clip(math.floor(f1 / s.df) + 1, 0, s.n_bins))
        if k1 <= k0:
            self.rms_label.setText("")
            return
        p = np.power(10.0, self._slice[k0:k1].astype(np.float64) / 10.0).sum()
        val = 10.0 * math.log10(max(p / (s.enbw * s.zp), 1e-30))
        self.rms_label.setText(f"RMS {max(f0, 0):.0f}-{min(f1, s.f_max):.0f} Hz = {val:.1f} {self._unit}")
        self._place_rms_label()

    def _place_rms_label(self, *_):
        self.rms_label.setPos(self.sp_vb.width() - 6, 6)

    # ---------------------------------------------------------------- playback
    def toggle_play(self):
        if self.audio is None:
            return
        if self.player.is_playing():
            self.pause()
        else:
            self.play()

    def play(self):
        self._update_loop()
        try:
            self.player.play()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Playback", f"Could not start audio playback.\n\n{exc}")
            return
        self._was_playing = True
        self._style_play_button(True)
        self._play_timer.start()

    def pause(self):
        pos = self.player.position_samples()
        self.player.stop()
        self.player.pos = pos
        self._was_playing = False
        self._play_timer.stop()
        self._style_play_button(False)
        self._sync_playhead()

    def stop(self):
        if self.audio is None:
            return
        self.pause()
        self.seek(self._loop_start())

    def seek(self, t: float):
        if self.audio is None:
            return
        dur = len(self.audio) / self.sr
        t = float(np.clip(t, 0, dur))
        self.player.pos = int(t * self.sr)
        self._sync_playhead(t)

    def step(self, frames: int):
        if self.spec is not None:
            self.seek(self._playhead_time() + frames * self.spec.dt)

    def step_seconds(self, sec: float):
        if self.audio is not None:
            self.seek(self._playhead_time() + sec)

    def _loop_range(self):
        if not self.loop_chk.isChecked() or self.audio is None:
            return None
        (x0, x1), _ = self.sg_vb.viewRange()
        n = len(self.audio)
        s, e = int(np.clip(x0 * self.sr, 0, n)), int(np.clip(x1 * self.sr, 0, n))
        return (s, e) if e - s > 0.05 * self.sr else None

    def _loop_start(self) -> float:
        r = self._loop_range()
        return r[0] / self.sr if r else 0.0

    def _update_loop(self, *_):
        self.player.loop = self._loop_range()

    def _sync_playhead(self, t: float | None = None):
        if t is None:
            t = self._playhead_time()
        if not self.playhead.moving:
            self.playhead.setValue(t)
        self._update_time_label(t)
        self.refresh_cursor()

    def _update_time_label(self, t: float | None = None):
        if self.audio is None:
            return
        if t is None:
            t = self._playhead_time()
        self.time_label.setText(f"{fmt_time(t)} / {fmt_time(len(self.audio) / self.sr)}")

    def _on_play_tick(self):
        if not self.player.is_playing():               # reached the end of the file
            if self._was_playing:
                self.player.stop()
                self._was_playing = False
                self._play_timer.stop()
                self._style_play_button(False)
                self.player.pos = len(self.audio)
                self._sync_playhead()
            return
        t = self._playhead_time()
        if self.follow_chk.isChecked() and not self.loop_chk.isChecked() and self.hover_sg is None:
            (x0, x1), _ = self.sg_vb.viewRange()
            width = x1 - x0
            if t > x1 or t < x0:
                self.sg_vb.setXRange(t - 0.02 * width, t + 0.98 * width, padding=0)
        self._sync_playhead(t)

    # ---------------------------------------------------------------- misc
    def show_help(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"{APP_NAME} - help")
        box.setText(f"<pre>{_HELP_TEXT}</pre>")
        box.exec()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            if url.isLocalFile():
                self.load_file(url.toLocalFile())
                break

    def closeEvent(self, ev):
        self.player.stop()
        self._save_settings()
        for w in list(self._workers):
            w.requestInterruption()
            w.wait(3000)
        super().closeEvent(ev)


_HELP_TEXT = """\
MOUSE (both charts)
  Wheel - zoom both axes around the pointer
  Wheel over an axis - zoom only that axis
  Ctrl + wheel - zoom X only;  Shift + wheel - zoom Y only
  Left-drag - pan  (or draw a zoom box when "Box zoom" is on)
  Right-drag - stretch / squeeze X and Y independently
  Right-click - view menu (view all, export to PNG / CSV / SVG ...)

CURSOR
  Hover the spectrogram - the spectrum chart shows that moment
  Click the spectrogram - move the playhead there and pin the cursor frequency
  Hover / click the spectrum - move / pin the cursor frequency
  Drag the dashed white line - seek
  The colour bar on the right can be dragged to change the colour range

KEYS
  Space  play / pause          Home  go to start
  Left / Right  step one frame  Shift + Left / Right  step 1 s
  F  fit view                  Ctrl + O  open file

LEVELS
  0 dBFS = a full-scale sine wave. Enter a calibration offset to read dB SPL.
  The RMS box (top right of the spectrum) is the level of the visible
  frequency band at the cursor time - zoom the spectrum to choose the band.
"""


def apply_dark_theme(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    window, base, text = QtGui.QColor(37, 37, 38), QtGui.QColor(28, 28, 30), QtGui.QColor(225, 225, 225)
    pal.setColor(R.Window, window)
    pal.setColor(R.WindowText, text)
    pal.setColor(R.Base, base)
    pal.setColor(R.AlternateBase, window)
    pal.setColor(R.ToolTipBase, QtGui.QColor(50, 50, 52))
    pal.setColor(R.ToolTipText, text)
    pal.setColor(R.Text, text)
    pal.setColor(R.Button, QtGui.QColor(50, 50, 53))
    pal.setColor(R.ButtonText, text)
    pal.setColor(R.Highlight, QtGui.QColor(42, 130, 218))
    pal.setColor(R.HighlightedText, QtGui.QColor(255, 255, 255))
    pal.setColor(QtGui.QPalette.ColorGroup.Disabled, R.Text, QtGui.QColor(120, 120, 120))
    pal.setColor(QtGui.QPalette.ColorGroup.Disabled, R.ButtonText, QtGui.QColor(120, 120, 120))
    app.setPalette(pal)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spectrogram / spectrum analyzer for audio files")
    parser.add_argument("file", nargs="?", help="audio file to open")
    parser.add_argument("--demo", action="store_true", help="open a synthetic engine run-up recording")
    args = parser.parse_args(argv)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="k", foreground="#c8c8c8")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    apply_dark_theme(app)

    win = MainWindow()
    win.show()
    if args.demo:
        win.set_audio(make_demo_signal(), 44100, "demo engine run-up (synthetic)")
    elif args.file:
        win.load_file(args.file)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
