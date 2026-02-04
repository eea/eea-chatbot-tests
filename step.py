from contextlib import contextmanager
from chatbot_tests.plugin import log_step
import time
import sys


@contextmanager
def step(description: str, continue_on_failure: bool = False, step_type: str = "action", start: float = None):
    """Context manager for test steps with logging.

    Args:
        description: Human-readable step description
        continue_on_failure: If True, don't re-raise exceptions
        start: Optional start timestamp (defaults to now)

    Output is handled by the test runner's on_event callback.
    """
    start_time = start or time.time()

    try:
        yield
        duration = time.time() - start_time
        duration_ms = int(duration * 1000)
        log_step(description, "passed", duration_ms=duration_ms, step_type=step_type)
    except Exception:
        duration = time.time() - start_time
        duration_ms = int(duration * 1000)

        # Get exception info
        exc_type, exc_value, exc_tb = sys.exc_info()

        # Get error message - may be empty due to pytest assertion rewriting issue
        error_msg = str(exc_value) if exc_value.args else ""

        # If no message (pytest assertion rewriting fails with -p no:terminal),
        # try to evaluate the assertion message from the source code
        if not error_msg and isinstance(exc_value, AssertionError):
            # Find the test file frame (skip step.py frame)
            tb = exc_tb.tb_next if exc_tb.tb_next else exc_tb
            if tb:
                frame = tb.tb_frame
                # Get the source line
                import linecache
                filename = frame.f_code.co_filename
                lineno = tb.tb_lineno
                source_line = linecache.getline(filename, lineno).strip()

                # Extract and evaluate the message expression
                if 'assert ' in source_line and ', ' in source_line:
                    # Split on first comma after assert
                    _, msg_part = source_line.split(', ', 1)

                    # Try to evaluate the message expression in the frame's context
                    try:
                        # Combine locals and globals from the frame
                        eval_context = {**frame.f_globals, **frame.f_locals}
                        # Evaluate the message expression (handles f-strings, variables, etc.)
                        error_msg = str(eval(msg_part, eval_context))
                    except Exception:
                        # If evaluation fails, just use the raw message part
                        error_msg = msg_part.strip('\'"')

        # Format the error
        error_msg = error_msg.replace("\n", f"\n{' ' * 6}")
        err = f"{exc_type.__name__}: {error_msg}" if error_msg else exc_type.__name__

        log_step(description, "failed", err, duration_ms=duration_ms, step_type=step_type)
        if not continue_on_failure:
            raise


def info(message: str, outcome: str = "passed"):
    """Log an informational step (no timing expected).

    Use this for status messages, results logging, or informational notes
    that don't represent timed actions.

    Args:
        message: Informational message to log
        outcome: "passed" or "failed" (default: "passed")
    """
    log_step(message, outcome, step_type="info")
