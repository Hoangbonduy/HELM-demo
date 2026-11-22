"""Run one configured criterion through the shared benchmark pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import main


def run(criterion, *, default_run=None, full_datasets=False):
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(f"Usage: python3 {Path(sys.argv[0]).name} {{generate,status,evaluate}} [options]")
        print("Generate requests 512x512 tiles, assembles a PNG, and deletes the benchmark conversation.")
        print("Use status --run NAME to see remaining image requests, or --help after an action.")
        if default_run:
            print(f"Default run: {default_run}. Override it with --run NAME.")
        return 0 if args else 2
    action, *options = args
    if action not in {"generate", "status", "evaluate"}:
        print("Choose generate, status, or evaluate.", file=sys.stderr)
        return 2
    if any(option.startswith("--crit") or option in {"--c", "--cr"} for option in options):
        print("This script fixes the criterion.", file=sys.stderr)
        return 2
    if default_run and not any(option == "--run" or option.startswith("--run=") for option in options):
        options = ["--run", default_run, *options]
    if full_datasets and action in {"generate", "status"} and "--full-datasets" not in options:
        options = ["--full-datasets", *options]
    action = "progress" if action == "status" else action
    return main([action, *options, "--criteria", str(criterion)])
