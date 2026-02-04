"""Console output utilities with color support."""

import os

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_RED = "\033[91m"
BRIGHT_CYAN = "\033[96m"

# Module-level color state
_force_color = None


def use_color() -> bool:
    """Check if colors should be used."""
    if _force_color is not None:
        return _force_color
    try:
        return os.isatty(1)
    except Exception:
        return False


def force_color(enabled: bool):
    """Force colors on or off."""
    global _force_color
    _force_color = enabled


def style(text: str, *codes: str) -> str:
    """Apply style codes to text."""
    if not use_color():
        return text
    return f"{''.join(codes)}{text}{RESET}"


# Semantic styling helpers
def success(text: str) -> str:
    return style(text, BOLD, BRIGHT_GREEN)


def error(text: str) -> str:
    return style(text, BOLD, BRIGHT_RED)


def info(text: str) -> str:
    return style(text, BRIGHT_CYAN)


def label(text: str) -> str:
    return style(text, BOLD, CYAN)


def dim(text: str) -> str:
    return style(text, DIM)


def warn(text: str) -> str:
    return style(text, YELLOW)


def bold(text: str) -> str:
    return style(text, BOLD)


# Output functions
def write(text: str):
    """Write text to stdout, bypassing any capture."""
    os.write(1, text.encode())


def writeln(text: str = ""):
    """Write line to stdout, bypassing any capture."""
    os.write(1, f"{text}\n".encode())


def log(text: str, flush: bool = True):
    """Print text to stdout."""
    print(text, flush=flush)
