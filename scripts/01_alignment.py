"""Run alignment, generating two images per unfinished prompt."""
from _run_criterion import run


if __name__ == "__main__":
    raise SystemExit(run(1, default_run="vita-ask-uat-alignment-coco2000-drawbench-parti"))
