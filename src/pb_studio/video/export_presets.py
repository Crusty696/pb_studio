"""
Export Presets für Video-Rendering

Vordefinierte Preset-Profile für gängige Export-Szenarien:
- YouTube (1080p, 4K)
- Instagram (Story, Post, Reel)
- TikTok
- Twitter
- Custom

Jedes Preset enthält:
- Auflösung (width x height)
- Frame-Rate (fps)
- Video-Codec (h264, h265, vp9)
- Audio-Codec (aac, opus)
- Bitrate (video + audio)
- Container-Format (mp4, webm, mov)

Author: PB_studio Development Team
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs."""

    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    PRORES = "prores_ks"


class AudioCodec(Enum):
    """Supported audio codecs."""

    AAC = "aac"
    OPUS = "libopus"
    MP3 = "libmp3lame"


class ContainerFormat(Enum):
    """Supported container formats."""

    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    MKV = "mkv"


@dataclass
class ExportPreset:
    """
    Video-Export-Preset Definition.

    Attributes:
        name: Preset-Name (z.B. "YouTube 1080p")
        description: Beschreibung des Presets
        width: Video-Breite in Pixeln
        height: Video-Höhe in Pixeln
        fps: Frame-Rate (Frames pro Sekunde)
        video_codec: Video-Codec (H264, H265, VP9, etc.)
        audio_codec: Audio-Codec (AAC, Opus, MP3)
        container: Container-Format (MP4, WebM, MOV)
        video_bitrate: Video-Bitrate in kbps (z.B. 5000 = 5 Mbps)
        audio_bitrate: Audio-Bitrate in kbps (z.B. 192)
        pixel_format: Pixel-Format (yuv420p, yuv444p, etc.)
        preset_speed: FFmpeg-Preset für Encoding-Speed (ultrafast, fast, medium, slow)
        crf: Constant Rate Factor für quality-based encoding (0-51, lower = better)
        extra_args: Zusätzliche FFmpeg-Argumente
    """

    name: str
    description: str
    width: int
    height: int
    fps: int = 30
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    container: ContainerFormat = ContainerFormat.MP4
    video_bitrate: int = 5000  # kbps
    audio_bitrate: int = 192  # kbps
    pixel_format: str = "yuv420p"
    preset_speed: str = "medium"  # ultrafast, fast, medium, slow, slower
    crf: int | None = None  # 0-51, None = use bitrate
    extra_args: list[str] = field(default_factory=list)

    @property
    def resolution_str(self) -> str:
        """Returns resolution as string (e.g., '1920x1080')."""
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        """Returns aspect ratio (e.g., 1.777 for 16:9)."""
        return self.width / self.height

    def to_ffmpeg_args(self) -> list[str]:
        """
        Converts preset to FFmpeg command-line arguments.

        Returns:
            List of FFmpeg arguments
        """
        args = []

        # Video codec + preset
        args.extend(["-c:v", self.video_codec.value])
        args.extend(["-preset", self.preset_speed])

        # CRF oder Bitrate
        if self.crf is not None:
            args.extend(["-crf", str(self.crf)])
        else:
            args.extend(["-b:v", f"{self.video_bitrate}k"])

        # Pixel-Format
        args.extend(["-pix_fmt", self.pixel_format])

        # Audio codec + bitrate
        args.extend(["-c:a", self.audio_codec.value])
        args.extend(["-b:a", f"{self.audio_bitrate}k"])

        # Frame-Rate
        args.extend(["-r", str(self.fps)])

        # Resolution (scale filter)
        args.extend(["-vf", f"scale={self.width}:{self.height}"])

        # Extra args
        args.extend(self.extra_args)

        return args


class ExportPresetManager:
    """
    Verwaltet Export-Presets für Video-Rendering.

    Beinhaltet Standard-Presets für gängige Plattformen
    und ermöglicht das Hinzufügen eigener Presets.
    """

    def __init__(self):
        self.presets: dict[str, ExportPreset] = {}
        self._load_default_presets()

        logger.info(f"ExportPresetManager initialisiert: {len(self.presets)} Presets")

    def _load_default_presets(self):
        """Lädt Standard-Presets für gängige Plattformen."""

        # ========================================================================
        # YouTube
        # ========================================================================
        self.presets["youtube_1080p"] = ExportPreset(
            name="YouTube 1080p",
            description="Optimiert für YouTube 1080p (16:9, H.264, AAC)",
            width=1920,
            height=1080,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=8000,  # 8 Mbps
            audio_bitrate=192,
            pixel_format="yuv420p",
            preset_speed="fast",
            extra_args=["-movflags", "+faststart"],  # Web-optimiert
        )

        self.presets["youtube_4k"] = ExportPreset(
            name="YouTube 4K",
            description="Optimiert für YouTube 4K (16:9, H.265, AAC)",
            width=3840,
            height=2160,
            fps=30,
            video_codec=VideoCodec.H265,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=35000,  # 35 Mbps
            audio_bitrate=256,
            pixel_format="yuv420p",
            preset_speed="medium",
            extra_args=["-movflags", "+faststart"],
        )

        self.presets["youtube_60fps"] = ExportPreset(
            name="YouTube 1080p 60fps",
            description="Optimiert für YouTube 1080p 60fps (Gaming, Sports)",
            width=1920,
            height=1080,
            fps=60,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=12000,  # 12 Mbps
            audio_bitrate=192,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        # ========================================================================
        # Instagram
        # ========================================================================
        self.presets["instagram_story"] = ExportPreset(
            name="Instagram Story",
            description="Instagram Story (9:16, max 15s)",
            width=1080,
            height=1920,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=3500,
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        self.presets["instagram_post"] = ExportPreset(
            name="Instagram Post",
            description="Instagram Feed Post (1:1, max 60s)",
            width=1080,
            height=1080,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=3500,
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        self.presets["instagram_reel"] = ExportPreset(
            name="Instagram Reel",
            description="Instagram Reel (9:16, max 90s)",
            width=1080,
            height=1920,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=5000,
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        # ========================================================================
        # TikTok
        # ========================================================================
        self.presets["tiktok"] = ExportPreset(
            name="TikTok",
            description="TikTok (9:16, max 10min)",
            width=1080,
            height=1920,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=5000,
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        # ========================================================================
        # Twitter
        # ========================================================================
        self.presets["twitter"] = ExportPreset(
            name="Twitter",
            description="Twitter/X (16:9, max 2:20)",
            width=1280,
            height=720,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            video_bitrate=5000,
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="fast",
        )

        # ========================================================================
        # High Quality
        # ========================================================================
        self.presets["hq_prores"] = ExportPreset(
            name="ProRes 422 HQ",
            description="High Quality für Editing (ProRes 422 HQ)",
            width=1920,
            height=1080,
            fps=30,
            video_codec=VideoCodec.PRORES,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MOV,
            video_bitrate=0,  # ProRes nutzt Quality-Stufen
            audio_bitrate=256,
            pixel_format="yuv422p10le",
            preset_speed="medium",
            extra_args=["-profile:v", "3"],  # ProRes 422 HQ
        )

        # ========================================================================
        # Web Optimized
        # ========================================================================
        self.presets["web_720p"] = ExportPreset(
            name="Web 720p",
            description="Web-optimiert (720p, kleiner Filesize)",
            width=1280,
            height=720,
            fps=30,
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            container=ContainerFormat.MP4,
            crf=23,  # Quality-based encoding
            audio_bitrate=128,
            pixel_format="yuv420p",
            preset_speed="medium",
            extra_args=["-movflags", "+faststart"],
        )

        logger.debug(f"Loaded {len(self.presets)} default presets")

    def get_preset(self, preset_id: str) -> ExportPreset | None:
        """
        Gibt Preset anhand der ID zurück.

        Args:
            preset_id: Preset-ID (z.B. "youtube_1080p")

        Returns:
            ExportPreset oder None
        """
        return self.presets.get(preset_id)

    def list_presets(self) -> list[str]:
        """
        Gibt Liste aller verfügbaren Preset-IDs zurück.

        Returns:
            Liste von Preset-IDs
        """
        return list(self.presets.keys())

    def list_presets_by_category(self) -> dict[str, list[str]]:
        """
        Gruppiert Presets nach Kategorie (Platform).

        Returns:
            Dict[Kategorie, Liste von Preset-IDs]
        """
        categories = {
            "YouTube": [],
            "Instagram": [],
            "TikTok": [],
            "Twitter": [],
            "High Quality": [],
            "Web": [],
            "Custom": [],
        }

        for preset_id, preset in self.presets.items():
            if preset_id.startswith("youtube_"):
                categories["YouTube"].append(preset_id)
            elif preset_id.startswith("instagram_"):
                categories["Instagram"].append(preset_id)
            elif preset_id.startswith("tiktok"):
                categories["TikTok"].append(preset_id)
            elif preset_id.startswith("twitter"):
                categories["Twitter"].append(preset_id)
            elif preset_id.startswith("hq_"):
                categories["High Quality"].append(preset_id)
            elif preset_id.startswith("web_"):
                categories["Web"].append(preset_id)
            else:
                categories["Custom"].append(preset_id)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def add_preset(self, preset_id: str, preset: ExportPreset):
        """
        Fügt ein neues Preset hinzu oder überschreibt bestehendes.

        Args:
            preset_id: Eindeutige Preset-ID
            preset: ExportPreset-Objekt
        """
        self.presets[preset_id] = preset
        logger.info(f"Preset hinzugefügt/aktualisiert: {preset_id}")

    def remove_preset(self, preset_id: str) -> bool:
        """
        Entfernt ein Preset.

        Args:
            preset_id: Preset-ID zum Entfernen

        Returns:
            True wenn erfolgreich, False wenn nicht gefunden
        """
        if preset_id in self.presets:
            del self.presets[preset_id]
            logger.info(f"Preset entfernt: {preset_id}")
            return True
        return False

    def get_preset_info(self, preset_id: str) -> dict | None:
        """
        Gibt detaillierte Info über ein Preset zurück.

        Args:
            preset_id: Preset-ID

        Returns:
            Dict mit Preset-Info oder None
        """
        preset = self.get_preset(preset_id)
        if not preset:
            return None

        return {
            "name": preset.name,
            "description": preset.description,
            "resolution": preset.resolution_str,
            "fps": preset.fps,
            "video_codec": preset.video_codec.name,
            "audio_codec": preset.audio_codec.name,
            "container": preset.container.value,
            "video_bitrate": f"{preset.video_bitrate} kbps"
            if preset.video_bitrate
            else f"CRF {preset.crf}",
            "audio_bitrate": f"{preset.audio_bitrate} kbps",
            "aspect_ratio": f"{preset.aspect_ratio:.3f}",
        }
