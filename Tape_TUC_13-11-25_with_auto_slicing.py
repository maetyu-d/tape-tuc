"""
Tape TUC - Clean Playback + Anti-Click
--------------------------------------
Stable, clear playback with varispeed, reverse per section, inertia,
target-time fit, zoom, and undo — plus an option to soften pops/clicks
at marked segment boundaries.

Features:
- Any WAV/FLAC/OGG/AIFF via soundfile
- Resampled once to device sample rate
- Looping
- Markers (add/remove/drag)
- Per-section speed + reverse
- Inertia (smooth transitions between section speeds)
- Target total time scaling
- Zoom in/out on waveform
- Undo for edits
- Anti-click toggle to apply soft gain shaping near marker boundaries
"""

import sys
import threading
import bisect
import math
import numpy as np
import soundfile as sf

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QIODevice
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
import pyqtgraph as pg


class TapeIODevice(QIODevice):
    def __init__(self, looper):
        super().__init__()
        self.looper = looper

    def readData(self, maxlen: int) -> bytes:
        frames = maxlen // 2  # mono int16 → 2 bytes
        if frames <= 0:
            return bytes()
        data = self.looper.provide_samples(frames)
        return data.tobytes()

    def writeData(self, data: bytes) -> int:
        return 0

    def bytesAvailable(self) -> int:
        return 48000 * 2 + super().bytesAvailable()


class TapeLooper(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tape TUC")

        # Audio state
        self.audio_data = None
        self.sample_rate = 48000
        self.num_samples = 0

        # Global playhead (forward in "tape space")
        self.play_pos = 0.0

        # Markers / sections
        self.markers = []
        self.boundary_samples = set()
        self.section_speeds = [1.0]
        self.section_reverse = [False]
        self.section_starts = []
        self.section_ends = []

        # Tape Age (placeholder for future FX)
        self.tape_age = 50
        self.enable_splice_fx = True

        # Inertia
        self.inertia_enabled = False
        self.inertia_amount = 50
        self.current_speed = 1.0

        # Boundary smoothing radius (for anti-click)
        self.boundary_smooth_len = 400  # ~8–9ms at 48k
        self.anticlick_enabled = True
        self.anticlick_amount = 50  # 0..100

        # Wow / flutter phases
        self.wow_phase = 0.0
        self.flutter_phase = 0.0

        # Splice FX envelope (for thumps at boundaries)
        self.splice_env_len = 256
        x = np.linspace(0, 1, self.splice_env_len, dtype=np.float32)
        # Simple thump: starts >1, decays toward 1
        self.splice_env = 1.0 + 0.8 * np.exp(-5.0 * x)
        self.splice_remaining = 0
        self.splice_index = 0

        # Undo stack
        self.undo_stack = []
        self._suppress_undo = False

        # Mutex
        self.lock = threading.RLock()

        # Qt audio
        self.audio_output = None
        self.audio_device = None
        self.is_playing = False

        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Audio")
        self.btn_detect = QtWidgets.QPushButton("Detect Beats")
        self.beat_sens = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.beat_sens.setRange(1, 100)
        self.beat_sens.setValue(50)
        self.beat_sens_lbl = QtWidgets.QLabel("50")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_rewind = QtWidgets.QPushButton("Rewind")
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom +")
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom -")
        self.btn_undo = QtWidgets.QPushButton("Undo")

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_detect)
        top.addWidget(self.beat_sens)
        top.addWidget(self.beat_sens_lbl)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_pause)
        top.addWidget(self.btn_stop)
        top.addWidget(self.btn_rewind)
        top.addWidget(self.btn_zoom_in)
        top.addWidget(self.btn_zoom_out)
        top.addWidget(self.btn_undo)
        top.addStretch()

        lbl = QtWidgets.QLabel("Tape Age:")
        self.age = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.age.setRange(0, 100)
        self.age.setValue(50)
        self.age_lbl = QtWidgets.QLabel("50")

        self.chk_splice = QtWidgets.QCheckBox("Splice FX ")
        self.chk_splice.setChecked(True)

        # Anti-click toggle
        self.chk_anticlick = QtWidgets.QCheckBox("Anti-click")
        self.chk_anticlick.setChecked(True)
        self.anticlick_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.anticlick_slider.setRange(0, 100)
        self.anticlick_slider.setValue(50)
        self.anticlick_lbl = QtWidgets.QLabel("50")


        # Inertia controls
        self.chk_inertia = QtWidgets.QCheckBox("Inertia")
        self.chk_inertia.setChecked(False)
        self.inertia_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.inertia_slider.setRange(0, 100)
        self.inertia_slider.setValue(50)
        self.inertia_lbl = QtWidgets.QLabel("50")

        top.addWidget(lbl)
        top.addWidget(self.age)
        top.addWidget(self.age_lbl)
        top.addWidget(self.chk_splice)
        top.addWidget(self.chk_anticlick)
        top.addWidget(self.anticlick_slider)
        top.addWidget(self.anticlick_lbl)
        top.addWidget(self.chk_inertia)
        top.addWidget(self.inertia_slider)
        top.addWidget(self.inertia_lbl)

        layout.addLayout(top)

        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter, 1)

        # Left: waveform
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot([], [])
        splitter.addWidget(self.plot)

        # Right: sections + target time
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Speed", "Reverse"])
        rlay.addWidget(self.table)

        # Target time controls
        target_layout = QtWidgets.QHBoxLayout()
        target_layout.addWidget(QtWidgets.QLabel("Target total time (s):"))
        self.target_time_edit = QtWidgets.QLineEdit()
        self.target_time_edit.setPlaceholderText("e.g. 60 for 1 minute")
        target_layout.addWidget(self.target_time_edit)
        self.btn_apply_target = QtWidgets.QPushButton("Fit to Target Time")
        target_layout.addWidget(self.btn_apply_target)
        rlay.addLayout(target_layout)

        splitter.addWidget(right)

        # Signals
        self.btn_load.clicked.connect(self.load_audio)
        self.btn_detect.clicked.connect(self.detect_beats)
        self.beat_sens.valueChanged.connect(self.on_sens_change)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_rewind.clicked.connect(self.rewind)
        self.age.valueChanged.connect(self.on_age)
        self.chk_splice.toggled.connect(self.on_splice_toggle)
        self.chk_anticlick.toggled.connect(self.on_anticlick_toggle)
        self.anticlick_slider.valueChanged.connect(self.on_anticlick_change)
        self.chk_inertia.toggled.connect(self.on_inertia_toggle)
        self.inertia_slider.valueChanged.connect(self.on_inertia_change)
        self.plot.scene().sigMouseClicked.connect(self.on_click)
        self.btn_apply_target.clicked.connect(self.on_apply_target_time)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_undo.clicked.connect(self.undo)

        # Playhead timer
        self.playhead = pg.InfiniteLine(angle=90, pen="y")
        self.plot.addItem(self.playhead)

        t = QtCore.QTimer(self)
        t.timeout.connect(self.update_playhead)
        t.start(50)
        self._playhead_timer = t

        self.marker_lines = []

    # ---------- Resampling helper ----------

    @staticmethod
    def _resample_to_rate(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
        if in_sr == out_sr or len(audio) == 0:
            return audio.astype(np.float32, copy=False)
        ratio = float(out_sr) / float(in_sr)
        new_len = int(round(len(audio) * ratio))
        if new_len <= 1:
            return audio.astype(np.float32, copy=False)
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
        resampled = np.interp(new_x, old_x, audio.astype(np.float64))
        return resampled.astype(np.float32)

    # ---------- Load audio ----------

    def load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open",
            "",
            "Audio (*.wav *.flac *.ogg *.aiff *.aif *.aifc *.au)"
        )
        if not path:
            return
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        if data.shape[1] > 1:
            mono = data.mean(axis=1)
        else:
            mono = data[:, 0]

        dev = QMediaDevices.defaultAudioOutput()
        fmt = dev.preferredFormat()
        out_sr = fmt.sampleRate()
        if out_sr <= 0:
            out_sr = sr

        processed = self._resample_to_rate(mono, int(sr), int(out_sr))

        with self.lock:
            self.audio_data = processed
            self.sample_rate = int(out_sr)
            self.num_samples = len(processed)
            self.play_pos = 0.0
            self.markers = []
            self.section_speeds = [1.0]
            self.section_reverse = [False]
            self.current_speed = 1.0
            self.recompute_boundaries_and_sections()
            self.undo_stack.clear()
            self.splice_remaining = 0
            self.splice_index = 0

        t = np.arange(self.num_samples) / self.sample_rate
        self.curve.setData(t, processed)

        for m in self.marker_lines:
            self.plot.removeItem(m)
        self.marker_lines = []

        self.rebuild_table()
        self.teardown_audio()
    def on_sens_change(self, v):
        self.beat_sens_lbl.setText(str(v))

    def detect_beats(self):
        """Automatic beat/onset detection and marker placement."""
        with self.lock:
            data = self.audio_data
            sr = self.sample_rate
            N = self.num_samples

        if data is None or N is None or N <= 0 or sr <= 0:
            QtWidgets.QMessageBox.warning(self, "No audio", "Load an audio file before detecting beats.")
            return

        x = np.array(data, dtype=np.float32)
        max_abs = np.max(np.abs(x)) if x.size > 0 else 0.0
        if max_abs > 0:
            x = x / max_abs

        frame_size = 1024
        hop = 512
        if N < frame_size + 1:
            QtWidgets.QMessageBox.information(self, "Too short", "Audio is too short for beat detection.")
            return

        num_frames = 1 + (N - frame_size) // hop
        if num_frames <= 1:
            QtWidgets.QMessageBox.information(self, "Too short", "Audio is too short for beat detection.")
            return

        energies = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop
            end = start + frame_size
            frame = x[start:end]
            energies[i] = np.sum(frame * frame)

        if num_frames >= 3:
            kernel = np.ones(3, dtype=np.float32) / 3.0
            e_smooth = np.convolve(energies, kernel, mode="same")
        else:
            e_smooth = energies

        diff = np.maximum(e_smooth[1:] - e_smooth[:-1], 0.0)
        if diff.size == 0:
            QtWidgets.QMessageBox.information(self, "No beats", "Could not detect any clear beats.")
            return

        mean = float(np.mean(diff))
        std = float(np.std(diff))
        thresh = mean + (self.beat_sens.value() / 100.0) * std

        min_interval_sec = 0.2
        min_gap = max(1, int(min_interval_sec * sr / hop))

        peaks = []
        last_peak = -min_gap
        for j in range(1, diff.size - 1):
            v = diff[j]
            if v < thresh:
                continue
            if not (v >= diff[j - 1] and v >= diff[j + 1]):
                continue
            if j - last_peak < min_gap:
                continue
            peaks.append(j)
            last_peak = j

        if not peaks:
            QtWidgets.QMessageBox.information(self, "No beats", "Could not detect any clear beats.")
            return

        beat_samples = [int(p * hop) for p in peaks]
        beat_samples = [b for b in beat_samples if 0 < b < N]
        if not beat_samples:
            QtWidgets.QMessageBox.information(self, "No beats", "Could not detect any clear beats.")
            return

        self.push_undo()
        with self.lock:
            self.markers = sorted(set(beat_samples))
            self.recompute_boundaries_and_sections()

        self.update_markers()
        self.rebuild_table()
    # ---------- Setup / teardown Qt audio ----------

    def setup_audio(self):
        self.teardown_audio()
        if self.audio_data is None:
            return

        dev = QMediaDevices.defaultAudioOutput()
        fmt = dev.preferredFormat()
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        fmt.setSampleRate(self.sample_rate)

        self.audio_output = QAudioSink(dev, fmt, self)
        self.audio_output.setVolume(1.0)
        self.audio_device = TapeIODevice(self)
        self.audio_device.open(QIODevice.OpenModeFlag.ReadOnly)

    def teardown_audio(self):
        if self.audio_output is not None:
            self.audio_output.stop()
        if self.audio_device is not None and self.audio_device.isOpen():
            self.audio_device.close()
        self.audio_output = None
        self.audio_device = None
        self.is_playing = False

    # ---------- Marker logic ----------

    def on_click(self, ev):
        if self.audio_data is None:
            return
        p = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x = p.x()
        if x < 0:
            return

        idx = int(x * self.sample_rate)
        idx = max(0, min(idx, self.num_samples - 1))

        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            changed = False
            self.push_undo()
            with self.lock:
                if idx not in self.markers:
                    bisect.insort(self.markers, idx)
                    self.recompute_boundaries_and_sections()
                    changed = True
            if changed:
                self.update_markers()
                self.rebuild_table()

        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            if not self.markers:
                return
            thr = int(0.02 * self.sample_rate)
            self.push_undo()
            with self.lock:
                nearest = min(self.markers, key=lambda m: abs(m - idx))
                if abs(nearest - idx) <= thr:
                    self.markers.remove(nearest)
                    self.recompute_boundaries_and_sections()
                    changed = True
                else:
                    changed = False
            if changed:
                self.update_markers()
                self.rebuild_table()

    def update_markers(self):
        for m in self.marker_lines:
            self.plot.removeItem(m)
        self.marker_lines = []

        with self.lock:
            ms = list(self.markers)

        for s in ms:
            t = s / self.sample_rate
            line = pg.InfiniteLine(pos=t, angle=90, movable=True, pen="r")
            line._s = s
            line.sigPositionChanged.connect(self.marker_moved)
            self.plot.addItem(line)
            self.marker_lines.append(line)

    def marker_moved(self):
        line = self.sender()
        t = float(line.value())
        s = int(t * self.sample_rate)
        s = max(0, min(s, self.num_samples - 1))

        changed = False
        self.push_undo()
        with self.lock:
            old = line._s
            if old in self.markers:
                self.markers.remove(old)
            if s not in self.markers:
                bisect.insort(self.markers, s)
                line._s = s
                changed = True
            else:
                bisect.insort(self.markers, old)
                line._s = old

            self.recompute_boundaries_and_sections()

        if changed:
            self.rebuild_table()

    def recompute_boundaries_and_sections(self):
        s = set(self.markers)
        s.add(0)
        if self.num_samples > 0:
            s.add(self.num_samples - 1)
        self.boundary_samples = s

        m_sorted = sorted(self.markers)
        starts = [0] + m_sorted
        ends = m_sorted + [self.num_samples]
        self.section_starts = starts
        self.section_ends = ends

    # ---------- Sections / table ----------

    def get_sections(self):
        with self.lock:
            starts = list(self.section_starts)
            ends = list(self.section_ends)
        return list(zip(starts, ends))

    def rebuild_table(self):
        secs = self.get_sections()
        with self.lock:
            oldS = self.section_speeds
            oldR = self.section_reverse
            self.section_speeds = [
                oldS[i] if i < len(oldS) else 1.0 for i in range(len(secs))
            ]
            self.section_reverse = [
                oldR[i] if i < len(oldR) else False for i in range(len(secs))
            ]

        self.table.blockSignals(True)
        self.table.setRowCount(len(secs))

        for i, (s, e) in enumerate(secs):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s / self.sample_rate:.3f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{e / self.sample_rate:.3f}"))

            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.25, 4.0)
            sp.setSingleStep(0.05)
            sp.setValue(self.section_speeds[i])
            sp.setProperty("i", i)
            sp.valueChanged.connect(self.speed_changed)
            self.table.setCellWidget(i, 3, sp)

            ck = QtWidgets.QCheckBox()
            ck.setChecked(self.section_reverse[i])
            ck.setProperty("i", i)
            ck.toggled.connect(self.reverse_changed)
            self.table.setCellWidget(i, 4, ck)

        self.table.blockSignals(False)

    def speed_changed(self, val):
        i = self.sender().property("i")
        self.push_undo()
        with self.lock:
            self.section_speeds[i] = float(val)

    def reverse_changed(self, val):
        i = self.sender().property("i")
        self.push_undo()
        with self.lock:
            self.section_reverse[i] = bool(val)

    # ---------- Transport ----------

    def play(self):
        if self.audio_data is None:
            QtWidgets.QMessageBox.warning(self, "No audio", "Load an audio file first.")
            return
        self.setup_audio()
        with self.lock:
            if self.play_pos >= self.num_samples:
                self.play_pos = 0.0
        if self.audio_output is not None and self.audio_device is not None:
            self.audio_output.start(self.audio_device)
            self.is_playing = True

    def pause(self):
        self.teardown_audio()

    def stop(self):
        self.teardown_audio()
        with self.lock:
            self.play_pos = 0.0

    def rewind(self):
        with self.lock:
            self.play_pos = 0.0

    # ---------- UI updates ----------

    def update_playhead(self):
        if self.audio_data is None:
            return
        with self.lock:
            t = self.play_pos / self.sample_rate if self.sample_rate > 0 else 0.0
        self.playhead.setPos(t)

    def zoom_in(self):
        self._zoom(0.5)

    def zoom_out(self):
        self._zoom(2.0)

    def _zoom(self, factor: float):
        if self.audio_data is None or self.sample_rate <= 0:
            return
        vb = self.plot.getViewBox()
        xr, yr = vb.viewRange()
        full_dur = self.num_samples / float(self.sample_rate) if self.num_samples > 0 else 1.0
        if not math.isfinite(full_dur) or full_dur <= 0:
            return

        cur_start, cur_end = xr
        if not (math.isfinite(cur_start) and math.isfinite(cur_end)) or cur_end <= cur_start:
            cur_start, cur_end = 0.0, full_dur
        center = 0.5 * (cur_start + cur_end)
        half = 0.5 * (cur_end - cur_start)
        if half <= 0:
            half = full_dur * 0.5

        new_half = half * factor
        min_half = full_dur / 1000.0
        max_half = full_dur * 0.5
        if new_half < min_half:
            new_half = min_half
        if new_half > max_half:
            new_half = max_half

        start = center - new_half
        end = center + new_half

        if start < 0.0:
            end -= start
            start = 0.0
        if end > full_dur:
            shift = end - full_dur
            start -= shift
            end = full_dur
            if start < 0.0:
                start = 0.0

        vb.setXRange(start, end, padding=0.0)

    # ---------- Handlers ----------

    def on_age(self, v):
        self.push_undo()
        with self.lock:
            self.tape_age = int(v)
        self.age_lbl.setText(str(v))

    def on_splice_toggle(self, v):
        self.push_undo()
        with self.lock:
            self.enable_splice_fx = bool(v)

    def on_anticlick_toggle(self, v):
        self.push_undo()
        with self.lock:
            self.anticlick_enabled = bool(v)

    def on_anticlick_change(self, val):
        self.push_undo()
        with self.lock:
            self.anticlick_amount = int(val)
        self.anticlick_lbl.setText(str(val))

    def on_inertia_toggle(self, v):
        self.push_undo()
        with self.lock:
            self.inertia_enabled = bool(v)

    def on_inertia_change(self, val):
        self.push_undo()
        with self.lock:
            self.inertia_amount = int(val)
        self.inertia_lbl.setText(str(val))

    def on_apply_target_time(self):
        self.push_undo()
        text = self.target_time_edit.text().strip()
        if not text:
            return
        try:
            target_seconds = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid time", "Enter target time in seconds (e.g. 60).")
            return
        if target_seconds <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid time", "Target time must be greater than zero.")
            return

        with self.lock:
            if self.audio_data is None or self.num_samples <= 0:
                return
            sr = float(self.sample_rate)
            secs = self.get_sections()
            speeds = list(self.section_speeds)
            if len(secs) == 0:
                return

            total_time = 0.0
            for i, (start, end) in enumerate(secs):
                length_samples = max(1, end - start)
                v = speeds[i] if i < len(speeds) and speeds[i] > 0 else 1.0
                total_time += length_samples / (v * sr)

            if total_time <= 0:
                return

            k = total_time / target_seconds
            new_speeds = []
            for v in speeds:
                nv = v * k
                if nv < 0.25:
                    nv = 0.25
                if nv > 4.0:
                    nv = 4.0
                new_speeds.append(nv)
            self.section_speeds = new_speeds

        self.rebuild_table()

    # ---------- Undo ----------

    def push_undo(self):
        if self._suppress_undo:
            return
        with self.lock:
            state = {
                'markers': list(self.markers),
                'section_speeds': list(self.section_speeds),
                'section_reverse': list(self.section_reverse),
                'tape_age': self.tape_age,
                'enable_splice_fx': self.enable_splice_fx,
                'anticlick_enabled': self.anticlick_enabled,
                'inertia_enabled': self.inertia_enabled,
                'inertia_amount': self.inertia_amount,
                'current_speed': self.current_speed,
                'play_pos': self.play_pos,
            }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _apply_state(self, state):
        with self.lock:
            self.markers = list(state.get('markers', []))
            self.section_speeds = list(state.get('section_speeds', []))
            self.section_reverse = list(state.get('section_reverse', []))
            self.tape_age = int(state.get('tape_age', self.tape_age))
            self.enable_splice_fx = bool(state.get('enable_splice_fx', self.enable_splice_fx))
            self.anticlick_enabled = bool(state.get('anticlick_enabled', self.anticlick_enabled))
            self.inertia_enabled = bool(state.get('inertia_enabled', self.inertia_enabled))
            self.inertia_amount = int(state.get('inertia_amount', self.inertia_amount))
            self.current_speed = float(state.get('current_speed', self.current_speed))
            self.play_pos = float(state.get('play_pos', self.play_pos))
            self.recompute_boundaries_and_sections()
        self._suppress_undo = True
        try:
            self.age.setValue(self.tape_age)
            self.chk_splice.setChecked(self.enable_splice_fx)
            self.chk_anticlick.setChecked(self.anticlick_enabled)
            self.chk_inertia.setChecked(self.inertia_enabled)
            self.inertia_slider.setValue(self.inertia_amount)
            self.update_markers()
            self.rebuild_table()
        finally:
            self._suppress_undo = False

    def undo(self):
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        self._apply_state(state)

    # ---------- Audio callback ----------

    def get_section_index_for_pos(self, pos, section_starts):
        sec = bisect.bisect_right(section_starts, pos) - 1
        if sec < 0:
            sec = 0
        return sec

    def provide_samples(self, frames: int) -> np.ndarray:
        with self.lock:
            if self.audio_data is None or self.num_samples == 0:
                return np.zeros(frames, np.int16)

            data = self.audio_data
            N = self.num_samples
            pos = self.play_pos
            sr = float(self.sample_rate)
            speeds = list(self.section_speeds)
            revs = list(self.section_reverse)
            section_starts = list(self.section_starts)
            section_ends = list(self.section_ends)
            boundary_samples = list(self.boundary_samples)
            anticlick = self.anticlick_enabled
            smooth_len = self.boundary_smooth_len
            anticlick_amount = self.anticlick_amount
            wow_p = self.wow_phase
            flt_p = self.flutter_phase
            age = self.tape_age
            splice_on = self.enable_splice_fx
            splice_remaining = self.splice_remaining
            splice_index = self.splice_index

        out = np.zeros(frames, np.int16)
        dt = 1.0 / sr if sr > 0 else 0.0
        num_secs = len(section_starts)

        # Wow & flutter parameters (scaled by Tape Age)
        a = max(0.0, min(1.0, age / 100.0))
        wow_depth = 0.001 + 0.006 * a   # slow, gentle
        flutter_depth = 0.0005 + 0.003 * a  # faster, subtle
        wow_freq = 0.4   # Hz
        flutter_freq = 7.0  # Hz

        for i in range(frames):
            # Wrap playhead into tape length
            if N > 0:
                if pos >= N:
                    pos -= N * int(pos // N)
                if pos < 0:
                    pos += N * (1 + int(-pos // N))

            if N <= 1:
                s = 0.0
                idx0 = 0
            else:
                # Determine section and local position within that section
                if num_secs == 0:
                    sec = 0
                    sec_start = 0
                    sec_end = N
                else:
                    sec = self.get_section_index_for_pos(pos, section_starts)
                    if sec >= num_secs:
                        sec = num_secs - 1
                    sec_start = section_starts[sec]
                    sec_end = section_ends[sec]
                    if sec_end <= sec_start:
                        sec_end = sec_start + 1

                sec_len = sec_end - sec_start
                local = pos - sec_start
                local = local % sec_len

                # Reverse or forward mapping for this section
                if sec < len(revs) and revs[sec]:
                    read_pos = (sec_end - 1) - local
                else:
                    read_pos = sec_start + local

                idx0 = int(read_pos)
                frac = read_pos - idx0
                idx0 = max(0, min(idx0, N - 1))
                idx1 = min(idx0 + 1, N - 1)
                s0 = data[idx0]
                s1 = data[idx1]
                s = (1.0 - frac) * s0 + frac * s1

            # Anti-click boundary smoothing (strength via slider)
            if anticlick and smooth_len > 0 and boundary_samples:
                dmin = min(abs(idx0 - b) for b in boundary_samples)
                if dmin < smooth_len:
                    x = (smooth_len - dmin) / smooth_len
                    # anticlick_amount 0 → gentle (min attenuation)
                    # anticlick_amount 100 → strong (more attenuation than before)
                    amt = max(0.0, min(1.0, anticlick_amount / 100.0))
                    base_strength = 0.3  # was ~0.6 before
                    extra = 0.5 * amt    # up to +0.5
                    strength = base_strength + extra  # 0.3..0.8
                    gain = 1.0 - strength * x
                    if gain < 0.0:
                        gain = 0.0
                    s *= gain

            # Splice FX: short thump at exact boundary crossings
            if splice_on and boundary_samples and idx0 in boundary_samples and splice_remaining <= 0:
                splice_remaining = self.splice_env_len
                splice_index = 0

            if splice_on and splice_remaining > 0 and splice_index < self.splice_env_len:
                s *= self.splice_env[splice_index]
                splice_remaining -= 1
                splice_index += 1

            # Clamp, convert to int16
            s = max(-1.0, min(1.0, s))
            out[i] = int(s * 32767)

            # Section speed + inertia
            if num_secs == 0:
                target = 1.0
            else:
                sec_for_speed = self.get_section_index_for_pos(pos, section_starts)
                if sec_for_speed < len(speeds):
                    target = speeds[sec_for_speed]
                else:
                    target = 1.0
            if target < 0:
                target = abs(target)

            if self.inertia_enabled and dt > 0.0 and self.inertia_amount > 0:
                tau_ms = 20.0 + 480.0 * (self.inertia_amount / 100.0)
                tau = tau_ms / 1000.0
                alpha = dt / tau if tau > 0 else 1.0
                if alpha > 1.0:
                    alpha = 1.0
                speed = self.current_speed + (target - self.current_speed) * alpha
            else:
                speed = target

            # Wow/flutter modulation: gently vary effective speed
            wow = math.sin(wow_p)
            flt = math.sin(flt_p)
            mod = 1.0 + wow_depth * wow + flutter_depth * flt
            # Clamp modulation so it never fully kills motion or explodes
            if mod < 0.1:
                mod = 0.1
            if mod > 3.0:
                mod = 3.0
            step = speed * mod
            pos += step
            self.current_speed = speed

            # Advance wow/flutter phases
            wow_p += 2.0 * math.pi * wow_freq * dt
            flt_p += 2.0 * math.pi * flutter_freq * dt

        two_pi = 2.0 * math.pi
        wow_p = wow_p % two_pi
        flt_p = flt_p % two_pi

        with self.lock:
            self.play_pos = pos
            self.wow_phase = wow_p
            self.flutter_phase = flt_p
            self.splice_remaining = splice_remaining
            self.splice_index = splice_index

        return out


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = TapeLooper()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
