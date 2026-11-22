"""Prepare and run the Vietnamese translation of the 2,000-prompt COCO sample."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import (add_generation_arguments, add_progress_arguments, generate,
                       report_run_progress, validate_generation_args, validate_progress_args)
from datasets.prepare import DatasetError, prepared_path
from datasets.sample import COCO_SAMPLE_TOTAL, SAMPLE_PATH, write_atomic

DATASET = "coco2014-vietnamese"
VERSION = "val2014-vietnamese-2000prompts"
SPEC = {"version": VERSION}
CRITERIA = {"11": {"name": "vietnamese", "datasets": [{"id": DATASET}]}}
DEFAULT_RUN = "vietnamese-coco2014-2000prompts"
DEFAULT_MODEL = Path.home() / ".cache" / "helm-coco2014-vi" / "windy"


def source_records():
    if not SAMPLE_PATH.is_file():
        raise DatasetError("Missing COCO sample. Run python3 benchmark.py fetch coco2014.")
    rows = [json.loads(line) for line in SAMPLE_PATH.read_bytes().splitlines()]
    if len(rows) != COCO_SAMPLE_TOTAL or any(
        row["dataset"] != "coco2014" or row["version"] != "val2014" or row["id"] != index
        for index, row in enumerate(rows, 1)
    ):
        raise DatasetError("Expected the ordered sample of 2,000 COCO val2014 prompts.")
    return rows


def translated_record(row, prompt):
    return {**row, "dataset": DATASET, "version": VERSION, "criteria": ["vietnamese"],
            "source_prompt": row["prompt"], "prompt": prompt}


def check_translation(path, source):
    if not path.is_file():
        raise DatasetError("Missing Vietnamese manifest. Run python3 scripts/11_vietnamese.py prepare.")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != len(source):
        raise DatasetError(f"Expected {len(source):,} Vietnamese prompts, found {len(rows):,}.")
    for original, row in zip(source, rows, strict=True):
        prompt = row.get("prompt")
        if (not isinstance(prompt, str) or not prompt.strip()
                or prompt.casefold() == original["prompt"].strip().casefold()
                or row != translated_record(original, prompt)):
            raise DatasetError(f"Vietnamese translation mismatches COCO prompt {original['id']}.")


def prepare_translation(path, source, model_dir):
    if path.is_file():
        check_translation(path, source)
        print(f"Vietnamese manifest ready: {path.relative_to(ROOT)} ({len(source):,} prompts)")
        return
    if not (model_dir / "model.bin").is_file():
        raise DatasetError("Provide a local English-to-Vietnamese CTranslate2 model with --model-dir.")
    try:
        import ctranslate2
        import sentencepiece
    except ImportError as error:
        raise DatasetError("Install ctranslate2 and sentencepiece to prepare the Vietnamese manifest.") from error

    encoder = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
    decoder = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
    translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8", intra_threads=4)
    records = []
    for start in range(0, len(source), 128):
        batch = source[start:start + 128]
        prompts = [row["prompt"] for row in batch]
        tokens = [pieces + ["</s>"] for pieces in encoder.encode(prompts, out_type=str)]
        results = translator.translate_batch(tokens, beam_size=1, max_batch_size=64,
                                             max_decoding_length=128, replace_unknowns=True)
        for row, token_ids, result in zip(batch, tokens, results, strict=True):
            prompt = decoder.decode(result.hypotheses[0]).strip()
            if not prompt or prompt.casefold() == row["prompt"].strip().casefold():
                retry, = translator.translate_batch([token_ids], beam_size=4, min_decoding_length=2,
                                                     max_decoding_length=128, replace_unknowns=True)
                prompt = decoder.decode(retry.hypotheses[0]).strip()
            if not prompt or prompt.casefold() == row["prompt"].strip().casefold():
                raise DatasetError(f"Could not translate COCO prompt {row['id']}.")
            records.append(translated_record(row, prompt))

    manifest = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    write_atomic(path, manifest.encode("utf-8"))
    check_translation(path, source)
    print(f"Prepared {len(records):,} Vietnamese prompts: {path.relative_to(ROOT)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("prepare", allow_abbrev=False)
    command.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    command = commands.add_parser("generate", allow_abbrev=False)
    add_generation_arguments(command)
    command.set_defaults(run=DEFAULT_RUN, allow_missing=False)
    command = commands.add_parser("status", allow_abbrev=False)
    add_progress_arguments(command, default_run=DEFAULT_RUN)
    args = parser.parse_args(argv)
    try:
        path = prepared_path(DATASET, SPEC)
        source = source_records()
        if args.action == "prepare":
            prepare_translation(path, source, args.model_dir)
            return 0
        check_translation(path, source)
        specs = {DATASET: SPEC}
        if args.action == "status":
            validate_progress_args(parser, args)
            return report_run_progress(args, CRITERIA, specs)
        validate_generation_args(parser, args)
        return generate(args, CRITERIA, specs)
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
