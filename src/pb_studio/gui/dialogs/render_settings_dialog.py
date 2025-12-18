from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from pb_studio.video.video_renderer import RenderSettings


class RenderSettingsDialog(QDialog):
    """
    Dialog for configuring video render settings.
    Allows adjusting Quality (CRF), GPU usage, and Presets.
    """

    def __init__(self, parent=None, initial_settings: RenderSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Render Settings")
        self.setMinimumWidth(400)

        self.settings = initial_settings or RenderSettings()

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Form Layout for controls
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 1. GPU Toggle
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setChecked(self.settings.use_gpu)
        self.gpu_check.setToolTip(
            "Use hardware acceleration (NVIDIA NVENC, AMD AMF, Intel QSV) if available."
        )
        form.addRow("Hardware:", self.gpu_check)

        # 2. CRF Slider (Quality)
        # Range: 18 (High) - 28 (Low), Default: 23
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(18, 28)
        self.crf_slider.setValue(self.settings.crf)
        self.crf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.crf_slider.setTickInterval(1)

        self.crf_label = QLabel(str(self.settings.crf))
        self.crf_label.setFixedWidth(30)
        self.crf_slider.valueChanged.connect(lambda v: self.crf_label.setText(str(v)))

        crf_container = QHBoxLayout()
        crf_container.addWidget(QLabel("High Quality (18)"))
        crf_container.addWidget(self.crf_slider)
        crf_container.addWidget(QLabel("Fast/Low (28)"))
        crf_container.addWidget(self.crf_label)

        form.addRow("Quality (CRF):", crf_container)

        # 3. Preset Combo
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"]
        )
        self.preset_combo.setCurrentText(self.settings.preset)
        self.preset_combo.setToolTip("Encoding speed vs compression efficiency.")
        form.addRow("Preset:", self.preset_combo)

        # 4. Encoder Combo (Advanced)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["auto", "h264_nvenc", "h264_amf", "h264_qsv", "libx264"])
        self.encoder_combo.setCurrentText(self.settings.gpu_encoder)
        self.encoder_combo.setToolTip("Specific encoder to use. 'auto' detects best available.")
        form.addRow("Encoder:", self.encoder_combo)

        layout.addLayout(form)

        # Info Label
        info_label = QLabel(
            "<i>Note: GPU acceleration requires compatible drivers.<br>"
            "CRF 23 is recommended for balanced quality and file size.</i>"
        )
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> RenderSettings:
        """Return the configured RenderSettings."""
        return RenderSettings(
            use_gpu=self.gpu_check.isChecked(),
            gpu_encoder=self.encoder_combo.currentText(),
            crf=self.crf_slider.value(),
            preset=self.preset_combo.currentText(),
        )
