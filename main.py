"""Main entry point for chatbot tests - CLI."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from chatbot_tests.config import get_settings, load_settings_from_json
from chatbot_tests import console


# Timestamp validation constants
MIN_TIMESTAMP = 1767225600  # 2026-01-01
MAX_TIMESTAMP = 4102444800  # 2100-01-01


def extract_timestamp_from_filename(filename: str) -> Optional[int]:
    """Extract Unix timestamp from filename if present and valid.

    Args:
        filename: Filename to extract timestamp from (e.g., 'test_run_1738600000.jsonl')

    Returns:
        Extracted timestamp if valid, None otherwise
    """
    match = re.search(r'_(\d{10,})\.jsonl?$', filename)
    if match:
        extracted = int(match.group(1))
        if MIN_TIMESTAMP <= extracted <= MAX_TIMESTAMP:
            return extracted
    return None


def get_timestamp_for_output(input_path: Optional[Path] = None) -> int:
    """Get timestamp for output files.

    Extracts from input filename if available, otherwise uses current time.

    Args:
        input_path: Optional input file path to extract timestamp from

    Returns:
        Unix timestamp
    """
    if input_path:
        timestamp = extract_timestamp_from_filename(input_path.name)
        if timestamp:
            return timestamp
    return int(datetime.now().timestamp())


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


def write_json(data: dict, output_path: Path) -> bool:
    """Write data to JSON file.

    Args:
        data: Dictionary to write
        output_path: Path to write to

    Returns:
        True if successful
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def write_pdf(text_content: str, output_path: Path) -> bool:
    """Convert text content to PDF and write to file.

    Args:
        text_content: Text to convert (ANSI codes will be stripped)
        output_path: Path to write PDF file

    Returns:
        True if successful, False otherwise
    """
    from md2pdf.core import md2pdf

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_text = strip_ansi(text_content)

        md2pdf(output_path, raw=clean_text, css="styles.css")

        return True
    except Exception as e:
        console.log(f"{console.error('Error:')} Failed to generate PDF: {e}")
        return False


def run_llm_analysis(settings, data: dict, analyze_func_name: str) -> Optional[str]:
    """Run LLM analysis on data.

    Args:
        settings: Settings instance
        data: Data to analyze
        analyze_func_name: Name of analyzer method ('analyze_test_report' or 'analyze_test_comparison')

    Returns:
        LLM analysis result string, or None if failed
    """
    from chatbot_tests.llm_analysis import create_analyzer_from_settings

    try:
        analyzer = create_analyzer_from_settings(settings)
    except (ValueError, ImportError) as e:
        console.log(f"{console.error('Error:')} {e}")
        return None

    if analyzer:
        analyze_func = getattr(analyzer, analyze_func_name)
        return analyze_func(data)
    return None


def print_llm_result(result: str, title: str = "LLM Analysis"):
    """Print LLM analysis result to terminal."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)
    print(result)


def save_outputs(
    output_data: dict,
    reports_path: Path,
    prefix: str,
    timestamp: int,
    llm_result: Optional[str] = None,
):
    """Save JSON and optionally PDF outputs.

    Args:
        output_data: Data to save as JSON
        reports_path: Directory for output files
        prefix: Filename prefix (e.g., 'analysis', 'comparison')
        timestamp: Unix timestamp for filename
        llm_result: Optional LLM result to save as PDF
    """
    # Save JSON
    json_path = reports_path / f"{prefix}_{timestamp}.json"
    write_json(output_data, json_path)
    console.log(f"{prefix.capitalize()} JSON written to: {json_path}")

    # Save PDF if LLM result provided
    if llm_result:
        pdf_path = reports_path / f"{prefix}_{timestamp}.pdf"
        if write_pdf(llm_result, pdf_path):
            console.log(f"LLM {prefix} PDF written to: {pdf_path}")


def load_config(args) -> bool:
    """Load config from specified path or default config.json."""
    config_path = Path(args.config) if args.config else Path("config.json")

    if not config_path.exists():
        if args.config:
            console.log(f"{console.error('Error:')} Config file not found: {config_path}")
            return False
        return True

    try:
        load_settings_from_json(config_path)
        console.log(f"Loaded config from: {config_path}")
        return True
    except (json.JSONDecodeError, ValueError) as e:
        console.log(f"{console.error('Error:')} Invalid config file: {e}")
        return False


def _print_run_header(settings, args, markers, output_path):
    """Print test run configuration header."""
    headless = settings.headless and not args.headed
    console.log("Running chatbot tests...")
    console.log(f"  Headless: {headless}")
    console.log(f"  Base URL: {settings.chatbot_base_url}")
    console.log(f"  Markers: {markers or 'all'}")
    if args.limit:
        console.log(f"  Limit: {args.limit} questions")
    console.log(f"  Output: {output_path}")
    console.log("")


def _print_run_summary(result):
    """Print test run summary with colors."""
    line = console.dim("=" * 50)
    console.log(f"\n{line}")
    console.log(f"Test Run Complete: {console.info(result.test_id[:8])}")

    status = console.success("COMPLETED") if result.status.value == "completed" else console.error("FAILED")
    console.log(f"Status: {status}")
    console.log(f"Duration: {console.dim(f'{result.duration_seconds:.2f}s')}")

    passed = console.success(str(result.passed))
    failed = console.error(str(result.failed)) if result.failed else "0"
    skipped = str(result.skipped)
    console.log(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}, Total: {result.total}")

    console.log(f"Output: {console.dim(result.output_file)}")
    console.log(line)


def cli_run(args):
    """Run tests via CLI."""
    from chatbot_tests.runner import run_tests_sync
    from chatbot_tests.output import generate_output_filename

    if not load_config(args):
        return 1

    if args.color:
        console.force_color(True)

    markers = args.marker.split(",") if args.marker else None
    settings = get_settings()

    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / generate_output_filename()
    else:
        output_path = settings.reports_path / generate_output_filename()

    _print_run_header(settings, args, markers, output_path)

    headless = settings.headless and not args.headed

    result = run_tests_sync(
        markers=markers,
        headless=headless,
        output_path=output_path,
        limit=args.limit,
    )

    _print_run_summary(result)
    return 0 if result.status.value == "completed" else 1


def cli_analyze(args):
    """Analyze test results from JSONL file."""
    from chatbot_tests.analyze import (
        analyze_jsonl, print_summary, print_failures, print_steps,
        print_by_marker, print_performance, print_insights, print_llm_verdicts
    )

    if not load_config(args):
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        console.log(f"{console.error('Error:')} File not found: {input_path}")
        return 1

    settings = get_settings()
    timestamp = get_timestamp_for_output(input_path)
    analysis = analyze_jsonl(input_path)

    # Print to terminal
    print_summary(analysis)

    show_all = getattr(args, 'all', False)
    if args.by_marker or show_all:
        print_by_marker(analysis)
    if args.performance or show_all:
        print_performance(analysis)
    if args.steps or show_all:
        print_steps(analysis)
    if args.failures or show_all:
        print_failures(analysis)
    if args.insights or show_all:
        print_insights(analysis)

    # Always show LLM verdicts when present
    if analysis.llm_verdicts:
        print_llm_verdicts(analysis)

    # Prepare output data
    output_data = analysis.to_dict()
    llm_result = None

    # LLM analysis if requested
    if args.llm:
        llm_result = run_llm_analysis(settings, output_data, 'analyze_test_report')
        if llm_result:
            print_llm_result(llm_result, "LLM Analysis")
            output_data["llm_analysis"] = llm_result

    # Save outputs
    save_outputs(output_data, settings.reports_path, "analysis", timestamp, llm_result)

    return 0


def cli_compare(args):
    """Compare multiple test runs."""
    from chatbot_tests.analyze import compare_runs, print_comparison

    if not load_config(args):
        return 1

    file_paths = [Path(f) for f in args.files]

    for path in file_paths:
        if not path.exists():
            console.log(f"{console.error('Error:')} File not found: {path}")
            return 1

    if len(file_paths) < 2:
        console.log(f"{console.error('Error:')} Need at least 2 files to compare")
        return 1

    settings = get_settings()
    timestamp = get_timestamp_for_output()
    comparison = compare_runs(file_paths)

    # Print to terminal
    print_comparison(comparison)

    # Prepare output data
    output_data = comparison.to_dict()
    llm_result = None

    # LLM analysis if requested
    if args.llm:
        llm_result = run_llm_analysis(settings, output_data, 'analyze_test_comparison')
        if llm_result:
            print_llm_result(llm_result, "LLM Comparison Analysis")
            output_data["llm_analysis"] = llm_result

    # Save outputs
    save_outputs(output_data, settings.reports_path, "comparison", timestamp, llm_result)

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chatbot Playwright Tests",
        prog="chatbot_tests",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run tests")
    run_parser.add_argument("-c", "--config", help="Path to JSON config file")
    run_parser.add_argument("-m", "--marker", help="Comma-separated test markers")
    run_parser.add_argument("--headed", action="store_true", help="Show browser (overrides config)")
    run_parser.add_argument("-o", "--output", help="Output file/directory")
    run_parser.add_argument("--color", action="store_true", help="Force colors")
    run_parser.add_argument("--limit", type=int, help="Limit to first N questions from fixtures")
    run_parser.set_defaults(func=cli_run)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("-c", "--config", help="Path to JSON config file")
    analyze_parser.add_argument("input", help="JSONL file to analyze")
    analyze_parser.add_argument("--failures", action="store_true", help="Show failures")
    analyze_parser.add_argument("--steps", action="store_true", help="Show steps")
    analyze_parser.add_argument("--by-marker", action="store_true", help="Group by marker")
    analyze_parser.add_argument("--performance", action="store_true", help="Show performance metrics")
    analyze_parser.add_argument("--insights", action="store_true", help="Show auto-generated insights")
    analyze_parser.add_argument("--all", action="store_true", help="Show all analysis sections")
    analyze_parser.add_argument("--llm", action="store_true", help="Use LLM for analysis (also generates PDF)")
    analyze_parser.set_defaults(func=cli_analyze)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument("-c", "--config", help="Path to JSON config file")
    compare_parser.add_argument("files", nargs="+", help="JSONL files to compare")
    compare_parser.add_argument("--llm", action="store_true", help="Use LLM for comparison analysis (also generates PDF)")
    compare_parser.set_defaults(func=cli_compare)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
