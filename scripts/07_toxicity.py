"""Run toxicity, saving successful tiles by slot until all four can be combined."""
from _run_criterion import run


if __name__ == "__main__":
    raise SystemExit(run(7, default_run="toxicity", full_datasets=True))
