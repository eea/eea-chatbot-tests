"""Output formatters for test results."""

import json
from pathlib import Path
from typing import TextIO, Optional
from datetime import datetime

from chatbot_tests.plugin import TestEvent


class JSONLWriter:
    """Writes test events to JSONL format in real-time."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._file: Optional[TextIO] = None

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    def write_event(self, event: TestEvent):
        """Write a single event as a JSON line."""
        if self._file:
            json_line = json.dumps(event.to_dict(), ensure_ascii=False)
            self._file.write(json_line + '\n')
            self._file.flush()  # Ensure real-time writing


def generate_output_filename(prefix: str = "test_run") -> str:
    """Generate timestamped output filename.

    Args:
        prefix: Filename prefix (default: 'test_run')

    Returns:
        Filename with Unix timestamp, e.g., 'test_run_1706367000.jsonl'
    """
    timestamp = int(datetime.now().timestamp())
    return f"{prefix}_{timestamp}.jsonl"
