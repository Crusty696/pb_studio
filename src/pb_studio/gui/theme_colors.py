"""
Theme Color Definitions for PB_studio

Defines color palettes for Dark and Light themes.
Used by ThemeManager to generate stylesheets from templates.

Modernized 2025 Design:
- VS Code inspired Dark Mode (#1e1e1e)
- Flat Design
- Rounded Corners
- Improved Spacing
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeColors:
    """
    Color palette for a theme.
    All colors are hex strings.
    """

    # Base colors
    background: str  # Main window background
    background_alt: str  # Panels, Docks, Menus
    background_input: str  # Input fields, lists
    foreground: str  # Main text
    foreground_muted: str  # Secondary text / labels

    # Accent colors
    primary: str  # Main interaction color (buttons, focus)
    primary_hover: str
    primary_pressed: str
    primary_text: str  # Text on primary color

    # Border colors
    border: str
    border_focus: str

    # State colors
    hover: str  # General hover state
    selected: str  # Selected item background
    disabled_bg: str
    disabled_fg: str

    # Semantic colors
    error: str
    success: str
    warning: str


# Modern Dark Theme (VS Code / Professional Studio Style)
DARK_COLORS = ThemeColors(
    # Base
    background="#1e1e1e",  # VS Code Editor Bg
    background_alt="#252526",  # VS Code Sidebar Bg
    background_input="#3c3c3c",  # Inputs
    foreground="#d4d4d4",  # Main Text
    foreground_muted="#858585",  # Comments/Labels
    # Accent (Modern Blue/Teal mix)
    primary="#007acc",  # VS Code Blue
    primary_hover="#0098ff",
    primary_pressed="#005a9e",
    primary_text="#ffffff",
    # Borders
    border="#3e3e42",  # Subtle border
    border_focus="#007acc",  # Focus border
    # States
    hover="#2a2d2e",
    selected="#37373d",
    disabled_bg="#2d2d2d",
    disabled_fg="#5a5a5a",
    # Semantic
    error="#f48771",
    success="#89d185",
    warning="#cca700",
)


# Modern Light Theme (Clean, Flat)
LIGHT_COLORS = ThemeColors(
    # Base
    background="#ffffff",
    background_alt="#f3f3f3",
    background_input="#ffffff",
    foreground="#1e1e1e",
    foreground_muted="#666666",
    # Accent
    primary="#007acc",
    primary_hover="#0062a3",
    primary_pressed="#004c7a",
    primary_text="#ffffff",
    # Borders
    border="#e5e5e5",
    border_focus="#007acc",
    # States
    hover="#f0f0f0",
    selected="#e4e6f1",
    disabled_bg="#f5f5f5",
    disabled_fg="#a0a0a0",
    # Semantic
    error="#e51400",
    success="#008000",
    warning="#ddb700",
)


# Modern QSS Template
STYLESHEET_TEMPLATE = """
/* Global Reset & Fonts */
* {{
    outline: none; /* Remove focus rect */
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}}

/* Main Window */
QMainWindow, QDialog {{
    background-color: {background};
    color: {foreground};
}}

QWidget {{
    background-color: {background};
    color: {foreground};
}}

/* -------------------------------------------------------------------------
   Panels & Dock Widgets
   ------------------------------------------------------------------------- */
QDockWidget {{
    titlebar-close-icon: url(:/icons/close.png);
    titlebar-normal-icon: url(:/icons/float.png);
    border: 1px solid {border};
}}

QDockWidget::title {{
    background-color: {background_alt};
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid {border};
    font-weight: bold;
    color: {foreground};
}}

/* Splitter */
QSplitter::handle {{
    background-color: {background};
    border: 1px solid {border};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* -------------------------------------------------------------------------
   Menus & Toolbar
   ------------------------------------------------------------------------- */
QMenuBar {{
    background-color: {background_alt};
    color: {foreground};
    border-bottom: 1px solid {border};
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 10px;
}}

QMenuBar::item:selected {{
    background-color: {hover};
    border-radius: 4px;
}}

QMenu {{
    background-color: {background_alt};
    color: {foreground};
    border: 1px solid {border};
    padding: 4px;
    border-radius: 6px;
}}

QMenu::item {{
    padding: 6px 24px 6px 10px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {primary};
    color: {primary_text};
}}

QMenu::separator {{
    height: 1px;
    background-color: {border};
    margin: 4px 0;
}}

QToolBar {{
    background-color: {background_alt};
    border-bottom: 1px solid {border};
    padding: 4px;
    spacing: 6px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
}}

QToolButton:hover {{
    background-color: {hover};
    border: 1px solid {border};
}}

QToolButton:pressed {{
    background-color: {selected};
}}

/* -------------------------------------------------------------------------
   Inputs & Controls
   ------------------------------------------------------------------------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {background_input};
    color: {foreground};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px;
    selection-background-color: {primary};
    selection-color: {primary_text};
}}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{
    border: 1px solid {border_focus};
    background-color: {background_input};
}}

/* Buttons */
QPushButton {{
    background-color: {primary};
    color: {primary_text};
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {primary_hover};
}}

QPushButton:pressed {{
    background-color: {primary_pressed};
}}

QPushButton:disabled {{
    background-color: {disabled_bg};
    color: {disabled_fg};
    border: 1px solid {border};
}}

/* Tool/Flat Buttons (Secondary actions) */
QPushButton[flat="true"] {{
    background-color: transparent;
    color: {foreground};
    border: 1px solid {border};
}}

QPushButton[flat="true"]:hover {{
    background-color: {hover};
    border-color: {foreground_muted};
}}

/* ComboBox */
QComboBox {{
    background-color: {background_input};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 5px 10px;
    color: {foreground};
}}

QComboBox:hover {{
    border-color: {foreground_muted};
}}

QComboBox:on {{
    border-color: {border_focus};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: none;
}}

QComboBox QAbstractItemView {{
    background-color: {background_alt};
    border: 1px solid {border};
    selection-background-color: {primary};
    selection-color: {primary_text};
    outline: none;
}}

/* CheckBox & RadioButton */
QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {foreground};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {border};
    border-radius: 3px;
    background-color: {background_input};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {primary};
    border-color: {primary};
    image: url(:/icons/check.png); /* Fallback to color if no icon */
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

/* -------------------------------------------------------------------------
   Lists, Trees, Tables
   ------------------------------------------------------------------------- */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {background_input};
    alternate-background-color: {background};
    border: 1px solid {border};
    border-radius: 4px;
    gridline-color: {border};
}}

QListWidget::item, QTreeWidget::item, QTableWidget::item {{
    padding: 6px;
    border-bottom: 1px solid transparent;
}}

QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background-color: {selected};
    color: {primary_text};
    border-left: 2px solid {primary};
}}

QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
    background-color: {hover};
}}

QHeaderView::section {{
    background-color: {background_alt};
    color: {foreground};
    padding: 6px;
    border: none;
    border-bottom: 1px solid {border};
    border-right: 1px solid {border};
    font-weight: bold;
}}

/* -------------------------------------------------------------------------
   Tabs
   ------------------------------------------------------------------------- */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 4px;
    background-color: {background};
    top: -1px; /* Overlap with tab bar */
}}

QTabBar::tab {{
    background-color: {background_alt};
    color: {foreground_muted};
    border: 1px solid {border};
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {background};
    color: {primary};
    border-bottom: 1px solid {background}; /* Blend with pane */
    border-top: 2px solid {primary};
    font-weight: bold;
}}

QTabBar::tab:hover {{
    background-color: {hover};
    color: {foreground};
}}

/* -------------------------------------------------------------------------
   Scrollbars (Modern, Thin)
   ------------------------------------------------------------------------- */
QScrollBar:vertical {{
    border: none;
    background: {background};
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {border};
    min-height: 20px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: {foreground_muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    border: none;
    background: {background};
    height: 10px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background: {border};
    min-width: 20px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {foreground_muted};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* -------------------------------------------------------------------------
   Special Widgets
   ------------------------------------------------------------------------- */
/* Progress Bar */
QProgressBar {{
    border: 1px solid {border};
    border-radius: 4px;
    text-align: center;
    background-color: {background_input};
}}

QProgressBar::chunk {{
    background-color: {primary};
    border-radius: 3px;
}}

/* Group Box */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 1.2em;
    padding-top: 10px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
    color: {primary};
}}

/* Status Bar */
QStatusBar {{
    background-color: {primary};
    color: {primary_text};
    border-top: 1px solid {border};
}}

QStatusBar QLabel {{
    color: {primary_text};
}}

/* Tooltips */
QToolTip {{
    background-color: {background_alt};
    color: {foreground};
    border: 1px solid {border};
    padding: 4px;
    border-radius: 4px;
    opacity: 230;
}}
"""


def generate_stylesheet(colors: ThemeColors) -> str:
    """
    Generate a complete stylesheet from a color palette.

    Args:
        colors: ThemeColors instance with color definitions

    Returns:
        Complete Qt StyleSheet string
    """
    return STYLESHEET_TEMPLATE.format(
        background=colors.background,
        background_alt=colors.background_alt,
        background_input=colors.background_input,
        foreground=colors.foreground,
        foreground_muted=colors.foreground_muted,
        primary=colors.primary,
        primary_hover=colors.primary_hover,
        primary_pressed=colors.primary_pressed,
        primary_text=colors.primary_text,
        border=colors.border,
        border_focus=colors.border_focus,
        hover=colors.hover,
        selected=colors.selected,
        disabled_bg=colors.disabled_bg,
        disabled_fg=colors.disabled_fg,
        error=colors.error,
        success=colors.success,
        warning=colors.warning,
    )


# Pre-generated stylesheets for performance
DARK_STYLESHEET = generate_stylesheet(DARK_COLORS)
LIGHT_STYLESHEET = generate_stylesheet(LIGHT_COLORS)
