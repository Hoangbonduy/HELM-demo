"""Dataset lifecycle and criterion runner for HELM-demo."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing, nullcontext
from datetime import datetime
from io import BytesIO
from itertools import groupby
from threading import Event
from urllib.error import URLError
from urllib.parse import quote

from datasets.download import DATA, ROOT, fetch_component, registry
from datasets.prepare import DatasetError, generation_path, iter_records, prepare, progress
from datasets.sample import (COCO_SAMPLE_TOTAL, REPORT_PATH, SAMPLE_PATH, build_sample,
                             criteria_for, limit_records, matches_entry, units)
from image_generation import (IMAGE_TIMEZONE, OUTPUT, ImageClient, RateLimitReached,
                              atomic_write, detect_extension, settings)


IMAGES_PER_PROMPT = 4
TILE_SIZE = 512
TILE_DIMENSIONS = (TILE_SIZE, TILE_SIZE)
TILE_REQUEST_SIZE = f"{TILE_SIZE}x{TILE_SIZE}"
COMPOSITE_INDEX = 0
CRITERION_OUTPUTS = frozenset({"aesthetics", "originality", "reasoning", "knowledge", "toxicity"})
FLAT_ERROR_FOLDERS = {"aesthetics": "error", "knowledge": "errors", "toxicity": "errors"}


def selected(values, available, parser, label):
    keys = list(values or available)
    missing = set(keys) - available.keys()
    if missing:
        parser.error(f"{label} không hợp lệ: {', '.join(sorted(missing))}")
    return {key: available[key] for key in keys}


def evaluation_criteria():
    return {
        **criteria_for(),
        "11": {"name": "vietnamese", "metrics": ["multilingual_clip_score"]},
    }


def selected_criteria(values, parser, *, evaluation=False):
    criteria = evaluation_criteria() if evaluation else criteria_for()
    names = {config["name"]: key for key, config in criteria.items()}
    return selected([names.get(value, value) for value in values], criteria, parser, "Tiêu chí")


def print_progress(reports):
    for row in reports:
        total = row["total_prompts"]
        number = f"{row['prompts']:,}/{total:,}" if total is not None else f"{row['prompts']:,}/?"
        percent = f" ({row['prompt_percent']:g}%)" if row["prompt_percent"] is not None else ""
        parts = []
        for part in row["components"]:
            value = part["percent"]
            pct = f"{value:g}%" if value is not None else "chưa rõ %"
            if part["complete"]:
                details = "đã tải"
            elif part["bytes"]:
                details = f"{part['bytes']:,} byte đang tải"
            else:
                details = "chưa tải"
            parts.append(f"{part['file']} {pct} [{details}]")
        print(f"{row['dataset']}: {number} prompt{percent} · {'; '.join(parts)}", flush=True)


def dataset_progress(name, spec, *, full=False):
    if name == "coco2014" and not full:
        return progress(name, spec, sample_path=SAMPLE_PATH, sample_total=COCO_SAMPLE_TOTAL)
    return progress(name, spec)


def dataset_action(action, specs, jobs):
    def work(item):
        name, spec = item
        errors = []
        if action == "fetch":
            for component in spec["components"]:
                try:
                    parent = DATA / "raw" / name / spec["version"]
                    parent.mkdir(parents=True, exist_ok=True)
                    with (parent / (component["file"] + ".lock")).open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        fetch_component(name, spec, component)
                except (ValueError, RuntimeError, OSError) as error:
                    errors.append(f"{name}/{component['file']}: {error}")
        try:
            if action == "fetch" and name == "coco2014":
                if not errors and (dataset_progress(name, spec)["prompts"] != COCO_SAMPLE_TOTAL
                                   or not REPORT_PATH.is_file()):
                    build_sample(specs, coco_only=True)
            else:
                prepare(name, spec)
        except (ValueError, OSError, KeyError, EOFError) as error:
            errors.append(f"{name}: {error}")
        return errors

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        errors = [error for group in pool.map(work, specs.items()) for error in group]
    print_progress([dataset_progress(name, spec, full=action == "prepare")
                    for name, spec in specs.items()])
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


def sample_for(chosen, full_datasets):
    configured = criteria_for().values()
    if full_datasets or not any(criterion in configured for criterion in chosen.values()):
        return None, None
    try:
        data = SAMPLE_PATH.read_bytes()
    except FileNotFoundError as error:
        raise DatasetError("Chưa có mẫu HELM/HEIM. Chạy: python3 benchmark.py sample") from error
    rows = tuple(map(json.loads, data.splitlines()))
    if any("id" in row for row in rows):
        if any(type(row.get("id")) is not int or row["id"] != index
               for index, row in enumerate(rows, 1)):
            raise DatasetError("ID trong manifest phải là số thứ tự liên tiếp bắt đầu từ 1")
        # Preserve resume compatibility when the only change is adding manifest IDs.
        data = "".join(
            json.dumps({key: value for key, value in row.items() if key != "id"},
                       ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ).encode("utf-8")
    return rows, hashlib.sha256(data).hexdigest()


def selected_records(criterion, specs, allow_missing, sampled_rows=None):
    sampled_datasets = criterion.get("sampled_datasets")
    if sampled_rows is not None:
        for row in sampled_rows:
            if sampled_datasets and row["dataset"] not in sampled_datasets:
                continue
            if criterion["name"] in row["criteria"]:
                if not any(matches_entry(entry, row) for entry in criterion["datasets"]):
                    raise DatasetError("Mẫu chưa khớp cấu hình. Chạy lại benchmark.py sample")
                yield row
        if not sampled_datasets:
            return
    for entry in criterion["datasets"]:
        name = entry["id"]
        if sampled_rows is not None and name in sampled_datasets:
            continue
        spec = specs[name]
        if not generation_path(name, spec).is_file() and allow_missing:
            print(f"Bỏ qua dataset chưa sẵn sàng: {name}", file=sys.stderr)
            continue
        rows = (row for row in iter_records(name, spec) if matches_entry(entry, row))
        if spec.get("format") in {"paint", "relational", "syntax"}:
            for group in units(rows):
                yield from group
        else:
            yield from rows


def numbered_records(criterion, specs, allow_missing, sampled_rows):
    rows = selected_records(criterion, specs, allow_missing, sampled_rows)
    if sampled_rows is None and criterion["name"] in CRITERION_OUTPUTS:
        return ({**row, "id": index} for index, row in enumerate(rows, 1))
    return rows


def run_database_path(run):
    return ROOT / "results" / "benchmark" / run / "requests.sqlite3"


def composite_dimensions(images_per_prompt):
    return 2 * TILE_SIZE, (images_per_prompt // 2) * TILE_SIZE


def report_run_progress(args, chosen, specs):
    sampled_rows, _ = sample_for(chosen, args.full_datasets)
    path = run_database_path(args.run)
    database = closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) if path.is_file() else nullcontext()
    with database as db:
        for criterion in chosen.values():
            name = criterion["name"]
            images_per_prompt = criterion.get("images_per_prompt", IMAGES_PER_PROMPT)
            saved = {}
            if db is not None:
                saved = {
                    (dataset, source_id, image_index): (status, image_path)
                    for dataset, source_id, image_index, status, image_path in db.execute("""
                        SELECT dataset, source_id, COALESCE(json_extract(record, '$.image_index'), 1),
                               json_extract(record, '$.status'),
                               COALESCE(json_extract(record, '$.tile_path'), json_extract(record, '$.image_path'))
                        FROM results WHERE criterion=? AND json_extract(record, '$.size')=?
                    """, (name, TILE_REQUEST_SIZE))
                    if COMPOSITE_INDEX <= image_index <= images_per_prompt
                }

            prompts = complete = ready = rejected = failed = missing_image = missing_grid = 0
            rows = selected_records(criterion, specs, args.allow_missing, sampled_rows)
            for sample in limit_records(rows, args.limit):
                prompts += 1
                key = (sample["dataset"], sample["source_id"])
                status, image_path = saved.get((*key, COMPOSITE_INDEX), (None, None))
                if status == "success" and image_path and (ROOT / image_path).is_file():
                    complete += 1
                    ready += images_per_prompt
                    continue
                ready_before = ready
                for image_index in range(1, images_per_prompt + 1):
                    status, image_path = saved.get((*key, image_index), (None, None))
                    if status == "success":
                        if image_path and (ROOT / image_path).is_file():
                            ready += 1
                        else:
                            missing_image += 1
                    elif status == "rejected":
                        rejected += 1
                    elif status == "error":
                        failed += 1
                if ready - ready_before == images_per_prompt:
                    missing_grid += 1

            total = prompts * images_per_prompt
            remaining = total - ready - rejected
            pending = remaining - failed - missing_image
            print(
                f"{args.run}/{name}: {complete:,}/{prompts:,} prompt đã có ảnh ghép. "
                f"Còn {remaining:,}/{total:,} lượt ảnh "
                f"({ready:,} đã có ảnh, {rejected:,} bị từ chối, {pending:,} chưa chạy, "
                f"{failed:,} lỗi cần thử lại, {missing_image:,} thiếu tệp ảnh, "
                f"{missing_grid:,} thiếu ảnh ghép)",
                flush=True,
            )
    return 0


def run_database(path):
    db = sqlite3.connect(path)
    try:
        with db:
            db.execute("BEGIN")
            db.execute("""CREATE TABLE IF NOT EXISTS run_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS run_config (
                criterion TEXT PRIMARY KEY, settings TEXT NOT NULL)""")
            schemas = {
                "results": """CREATE TABLE results (
                    criterion TEXT NOT NULL, dataset TEXT NOT NULL, source_id TEXT NOT NULL,
                    image_index INTEGER NOT NULL, record TEXT NOT NULL,
                    PRIMARY KEY (criterion, dataset, source_id, image_index))""",
                "image_slots": """CREATE TABLE image_slots (
                    number INTEGER PRIMARY KEY, criterion TEXT NOT NULL, dataset TEXT NOT NULL,
                    source_id TEXT NOT NULL, image_index INTEGER NOT NULL,
                    UNIQUE (criterion, dataset, source_id, image_index))""",
            }
            for table, schema in schemas.items():
                columns = [column[1] for column in db.execute(f"PRAGMA table_info({table})")]
                if not columns:
                    db.execute(schema)
                elif "image_index" not in columns:
                    previous = f"{table}_legacy"
                    db.execute(f"ALTER TABLE {table} RENAME TO {previous}")
                    db.execute(schema)
                    old_columns = ", ".join(columns)
                    db.execute(f"INSERT INTO {table} ({old_columns}, image_index) "
                               f"SELECT {old_columns}, 1 FROM {previous}")
                    db.execute(f"DROP TABLE {previous}")
    except BaseException:
        db.close()
        raise
    return db


def worker_ranges(count, workers):
    active_workers = min(count, workers)
    if active_workers == 0:
        return
    size, extra = divmod(count, active_workers)
    start = 0
    for index in range(active_workers):
        end = start + size + (index < extra)
        yield start, end
        start = end


def print_worker_ranges(count, ranges):
    for index, (start, end) in enumerate(ranges, 1):
        print(f"Worker {index}: prompt {start + 1:,}–{end:,}/{count:,}", flush=True)


def open_worker_terminal(run, index, path):
    distro = os.environ.get("WSL_DISTRO_NAME")
    powershell = shutil.which("powershell.exe")
    if not distro or not powershell:
        return False

    # Resolve the Windows Terminal execution alias on the Windows side of WSL.
    arguments = [
        "-w", "new", "new-tab", "--title", f"{run} | Worker {index}",
        "--suppressApplicationTitle", "wsl.exe", "--distribution", distro,
        "--exec", "tail", "-n", "+1", "-f", f"--pid={os.getpid()}", "--", str(path),
    ]
    quoted = subprocess.list2cmdline(arguments).replace("'", "''")
    command = (
        "$ErrorActionPreference = 'Stop'\n"
        f"Start-Process -FilePath wt.exe -ArgumentList '{quoted}' -WorkingDirectory $env:SystemRoot"
    )
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    try:
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Worker {index}: không mở được Windows Terminal ({error.__class__.__name__}).",
              file=sys.stderr, flush=True)
        return False
    return True


def generate_parallel(db_path, args, rows, config, name, worker_logs, images_per_prompt, rate_limit):
    samples = list(rows)
    keys = {(sample["dataset"], sample["source_id"]) for sample in samples}
    if len(keys) != len(samples):
        raise DatasetError("Danh sách prompt có ID trùng. Dùng --workers 1 hoặc kiểm tra dữ liệu nguồn.")
    ranges = list(worker_ranges(len(samples), args.workers))
    if not ranges:
        raise ValueError(f"{name} không có mẫu để chạy")
    print_worker_ranges(len(samples), ranges)
    log_dir = db_path.parent / "worker-logs"
    log_dir.mkdir(exist_ok=True)
    for index in range(1, len(ranges) + 1):
        if index not in worker_logs:
            path = log_dir / f"worker-{index}.log"
            path.write_text("", encoding="utf-8")
            opened = args.worker_terminals and open_worker_terminal(args.run, index, path)
            worker_logs[index] = path, opened
            print(f"Worker {index}: {path.relative_to(ROOT)}", flush=True)

    stop = Event()

    def work(index, start, end):
        path, terminal_opened = worker_logs[index]
        with path.open("a", encoding="utf-8", buffering=1) as log:
            def report(message, *, file=None, flush=False):
                line = f"[Worker {index}] {message}"
                print(line, file=log)
                if not terminal_opened:
                    print(line, file=file, flush=flush)

            report(f"{name}: prompt {start + 1:,}–{end:,}/{len(samples):,}", flush=True)
            try:
                client = ImageClient(config, use_configured_conversation=False, rate_limit=rate_limit)
                with closing(sqlite3.connect(db_path, timeout=30)) as db:
                    failures = generate_prompts(
                        db, args, client, samples[start:end], name, images_per_prompt, stop=stop, report=report
                    )
            except Exception as error:
                report(f"Lỗi: {error}", file=sys.stderr, flush=True)
                raise
            report(f"{name}: {'đã dừng' if stop.is_set() else 'hoàn tất'}, {failures:,} ảnh lỗi.",
                   flush=True)
            return failures

    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        try:
            futures = [
                pool.submit(work, index, start, end)
                for index, (start, end) in enumerate(ranges, 1)
            ]
            return sum(future.result() for future in as_completed(futures))
        finally:
            stop.set()


def generate(args, chosen, specs):
    sampled_rows, sample_hash = sample_for(chosen, args.full_datasets)
    if not args.allow_missing:
        for criterion in chosen.values():
            for entry in criterion["datasets"]:
                name = entry["id"]
                if sampled_rows is not None and (
                    not criterion.get("sampled_datasets") or name in criterion["sampled_datasets"]
                ):
                    continue
                if not generation_path(name, specs[name]).is_file():
                    raise DatasetError(f"{name} chưa được chuẩn hóa")
    if args.dry_run:
        for criterion in chosen.values():
            rows = numbered_records(criterion, specs, args.allow_missing, sampled_rows)
            count = sum(1 for _ in limit_records(rows, args.limit))
            images_per_prompt = criterion.get("images_per_prompt", IMAGES_PER_PROMPT)
            width, height = composite_dimensions(images_per_prompt)
            print(f"{criterion['name']}: {count:,} prompt, {count * images_per_prompt:,} lượt tạo ô 512x512, "
                  f"{count:,} ảnh đầu ra {width}x{height}")
            if args.workers > 1:
                print_worker_ranges(count, worker_ranges(count, args.workers))
        return 0

    selected_names = {criterion["name"] for criterion in chosen.values()}
    for name in sorted(selected_names & {"knowledge", "toxicity"}):
        for folder in (FLAT_ERROR_FOLDERS[name], "images", "json"):
            (OUTPUT / name / folder).mkdir(parents=True, exist_ok=True)

    print(f"Khởi động lượt {args.run} với {args.workers} worker.", flush=True)
    config = settings()
    rate_limit = Event()
    client = ImageClient(config, use_configured_conversation=False, rate_limit=rate_limit)
    db_path = run_database_path(args.run)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    active_criterion = None
    try:
        with db_path.open("ab") as lock:
            print(f"{args.run}: đang mở SQLite và lấy khóa lượt chạy.", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            with closing(run_database(db_path)) as db:
                print(f"{args.run}: đang kiểm tra hội thoại và ảnh còn chờ dọn.", flush=True)
                if not cleanup_benchmark_conversations(db, client):
                    raise ValueError("Còn hội thoại benchmark chưa dọn được. Chạy lại cùng --run để thử xóa.")
                cleanup_saved_assets(db, client, pending_only=True)
                failures = 0
                worker_logs = {}
                for criterion in chosen.values():
                    name = active_criterion = criterion["name"]
                    images_per_prompt = criterion.get("images_per_prompt", IMAGES_PER_PROMPT)
                    save_run_config(db, args, criterion, client, sample_hash)
                    rows = limit_records(
                        numbered_records(criterion, specs, args.allow_missing, sampled_rows), args.limit
                    )
                    if args.workers == 1:
                        failures += generate_prompts(db, args, client, rows, name, images_per_prompt)
                    else:
                        failures += generate_parallel(
                            db_path, args, rows, config, name, worker_logs, images_per_prompt, rate_limit
                        )
    except (RateLimitReached, ValueError, OSError, sqlite3.Error) as error:
        affected = {active_criterion} if active_criterion else selected_names
        rate_limited = isinstance(error, RateLimitReached)
        filename = "rate-limited" if rate_limited else "run-error"
        for name in sorted(affected & FLAT_ERROR_FOLDERS.keys()):
            write_run_error(name, args.run, filename, error)
        if rate_limited:
            print(error, file=sys.stderr, flush=True)
            return 1
        raise
    except KeyboardInterrupt:
        print(f"Đã dừng lượt {args.run}. Chạy lại cùng --run để tiếp tục từ tiến độ đã lưu.",
              file=sys.stderr, flush=True)
        return 130
    return int(bool(failures))


def criterion_image_folder(criterion, run):
    folder = OUTPUT / criterion / "images"
    return folder if run == criterion else folder / run


def publish_image(db, key, record, run, image):
    generated_at = datetime.fromisoformat(record["generated_at"]).astimezone(IMAGE_TIMEZONE)
    criterion = key[0]
    prompt_id = record["sample"].get("id")
    if criterion in CRITERION_OUTPUTS:
        folder = criterion_image_folder(criterion, run)
    else:
        folder = OUTPUT / "images" / generated_at.strftime("%d-%m-%Y")
        if prompt_id is not None:
            folder = folder / run / criterion
    if prompt_id is not None:
        stem = f"id-{prompt_id}"
    else:
        with db:
            db.execute("INSERT OR IGNORE INTO image_slots (criterion, dataset, source_id, image_index) VALUES (?, ?, ?, ?)", key)
            number, = db.execute(
                "SELECT number FROM image_slots WHERE criterion=? AND dataset=? AND source_id=? AND image_index=?", key
            ).fetchone()
        stem = f"{run}-{number:04d}"
    target = folder / (stem + detect_extension(image))
    atomic_write(target, image)
    record["image_path"] = target.relative_to(ROOT).as_posix()


def tile_location(record):
    return record.get("tile_path") or record.get("image_path")


def publish_tile(key, record, run, image):
    criterion, dataset, source_id, index = key
    filename = f"{quote(source_id, safe='')}{detect_extension(image)}"
    path = run_database_path(run).parent / "tiles" / criterion / dataset / str(index) / filename
    atomic_write(path, image)
    record["tile_path"] = path.relative_to(ROOT).as_posix()


def toxicity_tile_path(run, sample, image_index):
    prompt_id = sample.get("id")
    if prompt_id is not None:
        stem = f"id-{prompt_id}"
    else:
        dataset = quote(sample["dataset"], safe="")
        source_id = quote(str(sample["source_id"]), safe="")
        stem = f"{dataset}@{source_id}"
    return criterion_image_folder("toxicity", run) / f"{stem}-{image_index}.png"


def publish_toxicity_tile(run, record):
    from PIL import Image

    image_index = record["image_index"]
    target = toxicity_tile_path(run, record["sample"], image_index)
    if target.is_file() and image_metadata_path(target, OUTPUT).is_file():
        return

    source = ROOT / tile_location(record)
    data = source.read_bytes()
    if detect_extension(data) != ".png":
        with Image.open(BytesIO(data)) as image, BytesIO() as output:
            image.save(output, "PNG")
            data = output.getvalue()

    atomic_write(target, data)
    output_record = {**record, "image_path": target.relative_to(ROOT).as_posix()}
    prompt = record.get("request_prompt") or record["sample"]["prompt"]
    write_image_metadata(output_record, prompt, image_index=image_index)


def remove_toxicity_tiles(run, composite):
    for image_index in range(1, composite.get("images_per_prompt", IMAGES_PER_PROMPT) + 1):
        path = toxicity_tile_path(run, composite["sample"], image_index)
        image_metadata_path(path, OUTPUT).unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def remove_tiles(run, tiles, composite):
    staging = run_database_path(run).parent / "tiles"
    for record in tiles.values():
        location = tile_location(record)
        if not location or location == composite["image_path"]:
            continue
        path = ROOT / location
        resolved = path.resolve()
        if record.get("tile_path"):
            if not resolved.is_relative_to(staging.resolve()):
                raise ValueError(f"Đường dẫn ảnh tạm nằm ngoài lượt chạy: {path}")
        else:
            images = OUTPUT / "images"
            if (not resolved.is_relative_to(images.resolve())
                    or not re.fullmatch(rf"{re.escape(run)}-[0-9]+", path.stem)):
                raise ValueError(f"Đường dẫn ảnh cũ không khớp lượt chạy: {path}")
            json_root = OUTPUT / "json"
            metadata = (json_root / path.relative_to(images)).with_suffix(".json")
            if not metadata.resolve().is_relative_to(json_root.resolve()):
                raise ValueError(f"Đường dẫn JSON cũ nằm ngoài thư mục metadata: {metadata}")
            metadata.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def publish_composite(db, run, key, tiles, sample, *, cleanup_slots=None, report=print):
    from PIL import Image

    images_per_prompt = len(tiles)
    records = [tiles[i] for i in range(1, images_per_prompt + 1)]
    base = records[0]
    elapsed = [r.get("elapsed_seconds") for r in records]
    record = {
        "status": "success",
        "sample": sample,
        "model": base.get("model"),
        "size": TILE_REQUEST_SIZE,
        "image_index": COMPOSITE_INDEX,
        "images_per_prompt": images_per_prompt,
        "request_prompt": base.get("request_prompt") or base["sample"]["prompt"],
        "generated_at": max(r["generated_at"] for r in records),
        "elapsed_seconds": sum(elapsed) if all(t is not None for t in elapsed) else None,
    }
    width, height = composite_dimensions(images_per_prompt)
    with Image.new("RGB", (width, height)) as grid, BytesIO() as output:
        for index, tile_record in enumerate(records):
            row, column = divmod(index, 2)
            with Image.open(ROOT / tile_location(tile_record)) as tile:
                if tile.size != TILE_DIMENSIONS:
                    raise ValueError(f"Ảnh số {index + 1} phải có kích thước 512x512.")
                grid.paste(tile, (column * TILE_SIZE, row * TILE_SIZE))
        grid.save(output, "PNG")
        publish_image(db, (*key, COMPOSITE_INDEX), record, run, output.getvalue())
    write_image_metadata(record, record["request_prompt"])
    store_result(db, (*key, COMPOSITE_INDEX), record)
    if key[0] == "toxicity":
        remove_toxicity_tiles(run, record)
    cleanup = tiles if cleanup_slots is None else {i: tiles[i] for i in cleanup_slots}
    remove_tiles(run, cleanup, record)
    report(f"Đã lưu {record['image_path']} ({width}x{height}).", flush=True)
    return record


def image_modified_at(image_path):
    return datetime.fromtimestamp(image_path.stat().st_mtime, IMAGE_TIMEZONE).isoformat(timespec="seconds")


def image_metadata_path(image, output):
    relative = image.relative_to(output)
    if ".." not in relative.parts:
        if len(relative.parts) >= 2 and relative.parts[0] == "images":
            return (output / "json" / relative.relative_to("images")).with_suffix(".json")
        if (len(relative.parts) >= 3 and relative.parts[0] in CRITERION_OUTPUTS
                and relative.parts[1] == "images"):
            criterion = relative.parts[0]
            return (output / criterion / "json" / relative.relative_to(criterion, "images")).with_suffix(".json")
    raise ValueError(f"Đường dẫn ảnh không hợp lệ: {image}")


def write_image_metadata(record, prompt, *, keep_existing=False, image_index=None):
    from PIL import Image

    image_path = ROOT / record["image_path"]
    metadata_path = image_metadata_path(image_path, OUTPUT)
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    if keep_existing and metadata_path.is_file():
        return
    metadata = {
        "image_filename": image_path.name,
        "dataset": record["sample"]["dataset"],
        "source_id": record["sample"]["source_id"],
        "width": width,
        "height": height,
        "prompt": prompt,
        "generated_at": record.get("generated_at") or image_modified_at(image_path),
        "elapsed_seconds": record.get("elapsed_seconds"),
    }
    if "id" in record["sample"]:
        metadata["id"] = record["sample"]["id"]
    if image_index is not None:
        metadata["image_index"] = image_index
    atomic_write(metadata_path, (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def flat_error_path(criterion, run, stem):
    prefix = "" if run == criterion else f"{run}@"
    return OUTPUT / criterion / FLAT_ERROR_FOLDERS[criterion] / f"{prefix}{stem}.json"


def write_run_error(criterion, run, filename, error):
    path = flat_error_path(criterion, run, filename)
    metadata = {
        "error": "RATE_LIMITED" if isinstance(error, RateLimitReached) else type(error).__name__,
        "message": str(error),
        "timestamp": datetime.now(IMAGE_TIMEZONE).isoformat(timespec="seconds"),
    }
    atomic_write(path, (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_error_metadata(run, key, record):
    criterion, dataset, source_id, image_index = key
    prompt_id = record["sample"].get("id")
    stem = f"id-{prompt_id}" if prompt_id is not None else quote(source_id, safe="")
    if criterion in FLAT_ERROR_FOLDERS:
        if prompt_id is None:
            stem = f"source-{stem}"
        metadata_path = flat_error_path(criterion, run, f"{dataset}-{stem}-image-{image_index}")
    else:
        if criterion in CRITERION_OUTPUTS:
            folder = OUTPUT / criterion / "json" / "errors" / run / dataset
        else:
            folder = OUTPUT / "errors" / run / criterion / dataset
        if image_index != 1:
            folder /= f"image-{image_index}"
        metadata_path = folder / f"{stem}.json"
    error = record.get("error_code") or record.get("error_type") or "UNKNOWN_ERROR"
    if "http_status" in record:
        error = f"{error} (HTTP {record['http_status']})"
    metadata = {
        "image_filename": None,
        "dataset": dataset,
        "source_id": source_id,
        "image_index": image_index,
        "width": None,
        "height": None,
        "prompt": record.get("request_prompt", record["sample"]["prompt"]),
        "generated_at": None,
        "elapsed_seconds": record.get("elapsed_seconds"),
        "error": error,
    }
    if prompt_id is not None:
        metadata["id"] = prompt_id
    atomic_write(metadata_path, (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def store_result(db, key, record):
    with db:
        db.execute("""INSERT OR REPLACE INTO results
            (criterion, dataset, source_id, image_index, record) VALUES (?,?,?,?,?)""",
            (*key, json.dumps(record)))


def delete_saved_asset(db, client, key, record):
    asset_id = record.get("asset_id")
    if not asset_id or record.get("asset_deleted_at"):
        return False
    record["asset_deleted_at"] = client.delete_asset(asset_id)
    store_result(db, key, record)
    return True


def cleanup_saved_assets(db, client, *, pending_only=False):
    rows = db.execute("""
        SELECT criterion, dataset, source_id, image_index, record FROM results
        WHERE (json_extract(record, '$.status') = 'success'
            OR json_extract(record, '$.error_code') IN ('SEED_UNVERIFIED', 'SEED_MISMATCH', 'BAD_IMAGE_SIZE'))
          AND json_extract(record, '$.asset_id') IS NOT NULL
          AND json_extract(record, '$.asset_deleted_at') IS NULL
    """).fetchall()
    deleted = 0
    for criterion, dataset, source_id, image_index, saved in rows:
        record = json.loads(saved)
        created = record.get("asset_created")
        if created is False or (pending_only and created is not True):
            continue
        if record["status"] == "success":
            location = tile_location(record)
            if not location or not (ROOT / location).is_file():
                continue
            if not record.get("tile_path"):
                write_image_metadata(record, record.get("request_prompt") or record["sample"]["prompt"],
                                     keep_existing=True)
        elif created is not True:
            continue
        deleted += delete_saved_asset(db, client, (criterion, dataset, source_id, image_index), record)
    return deleted


def cleanup(run):
    db_path = run_database_path(run)
    if not db_path.is_file():
        raise ValueError(f"Không tìm thấy lượt chạy {run}")
    client = ImageClient(settings(), use_configured_conversation=False, rate_limit=Event())
    with db_path.open("r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with closing(run_database(db_path)) as db:
            conversations_clean = cleanup_benchmark_conversations(db, client)
            deleted = cleanup_saved_assets(db, client)
    print(f"{run}: đã dọn {deleted:,} ảnh trên Vita Design")
    return int(not conversations_clean)


def start_benchmark_conversation(db, client):
    conversation_id = client.create_conversation()
    with db:
        db.execute("INSERT INTO run_metadata VALUES (?, ?)",
                   (f"benchmark-conversation:{conversation_id}", conversation_id))
    client.conversation_id = conversation_id


def cleanup_benchmark_conversations(db, client, *, conversation_id=None, report=print):
    if conversation_id is None:
        pending = db.execute(
            "SELECT key, value FROM run_metadata WHERE key LIKE 'benchmark-conversation:%'"
        ).fetchall()
    else:
        pending = db.execute(
            "SELECT key, value FROM run_metadata WHERE key=?",
            (f"benchmark-conversation:{conversation_id}",),
        ).fetchall()
    complete = True
    for key, conversation_id in pending:
        try:
            status = client.delete_conversation(conversation_id)
        except ValueError as error:
            report(f"Hội thoại {conversation_id} đang chờ dọn: {error}", file=sys.stderr, flush=True)
            complete = False
        else:
            with db:
                db.execute("DELETE FROM run_metadata WHERE key=?", (key,))
            message = "Đã xóa" if status == 204 else "Đã vắng mặt"
            report(f"{message} hội thoại benchmark {conversation_id} (HTTP {status}).", flush=True)
    return complete


def save_run_config(db, args, criterion, client, sample_hash):
    name = criterion["name"]
    size = TILE_REQUEST_SIZE
    images_per_prompt = criterion.get("images_per_prompt", IMAGES_PER_PROMPT)
    image_settings = {"images_per_prompt": images_per_prompt}
    run_settings = [client.base_url, client.model, size, criterion["datasets"], image_settings]
    if sample_hash:
        run_settings.append({"sample_sha256": sample_hash})
        if criterion.get("sampled_datasets"):
            run_settings.append({"sampled_datasets": criterion["sampled_datasets"]})
    if args.prompt_prefix:
        run_settings.append({"prompt_prefix": args.prompt_prefix})
    config = json.dumps(run_settings, sort_keys=True)
    existing = db.execute("SELECT settings FROM run_config WHERE criterion=?", (name,)).fetchone()
    saved_settings = json.loads(existing[0]) if existing else run_settings
    # Conversation IDs are transport state. Older runs stored them in the benchmark config.
    saved_settings = [
        {"images_per_prompt": IMAGES_PER_PROMPT} if isinstance(value, dict) and set(value) == {"seeds"} else value
        for value in saved_settings
        if not (isinstance(value, dict) and set(value) == {"conversation_id"})
    ]
    if name == "alignment" and images_per_prompt == 2:
        # Keep the same run when only the number of tiles for unfinished prompts changes.
        old_image_settings = {"images_per_prompt": IMAGES_PER_PROMPT}
        saved_settings = [image_settings if value == old_image_settings else value for value in saved_settings]
    if saved_settings != run_settings:
        raise ValueError(
            f"Lượt {args.run}/{name} đã lưu cấu hình khác. "
            "Dùng --run với tên mới để sinh ảnh theo cấu hình hiện tại."
        )
    with db:
        db.execute("INSERT OR REPLACE INTO run_config VALUES (?,?)", (name, config))


def generate_prompts(db, args, client, rows, name, images_per_prompt=IMAGES_PER_PROMPT, *, stop=None, report=print):
    size = TILE_REQUEST_SIZE
    count = failures = 0
    for sample in rows:
        if client.rate_limited:
            raise RateLimitReached()
        if stop is not None and stop.is_set():
            return failures
        count += 1
        prompt_key = (name, sample["dataset"], sample["source_id"])
        saved_slots = {
            image_index: json.loads(record)
            for image_index, record in db.execute("""
                SELECT image_index, record FROM results
                WHERE criterion=? AND dataset=? AND source_id=?""", prompt_key)
        }
        composite = saved_slots.pop(COMPOSITE_INDEX, None)
        if composite and (ROOT / composite["image_path"]).is_file():
            if name == "toxicity":
                remove_toxicity_tiles(args.run, composite)
            if images_per_prompt == IMAGES_PER_PROMPT:
                remove_tiles(args.run, saved_slots, composite)
            continue
        pending = []
        tiles = {}
        asset_ids = set()
        for image_index in range(1, images_per_prompt + 1):
            saved = saved_slots.get(image_index, {})
            status = saved.get("status") if saved.get("size") == size else None
            if status == "rejected":
                continue
            location = tile_location(saved)
            if status == "success" and location and (ROOT / location).is_file():
                tiles[image_index] = saved
                if saved.get("asset_id"):
                    asset_ids.add(saved["asset_id"])
                continue
            pending.append(image_index)
        if name == "toxicity" and len(tiles) < images_per_prompt:
            for tile in tiles.values():
                publish_toxicity_tile(args.run, tile)
        cleanup_slots = pending if images_per_prompt == 2 else None
        if not pending:
            if len(tiles) == images_per_prompt:
                publish_composite(db, args.run, prompt_key, tiles, sample, cleanup_slots=cleanup_slots, report=report)
            continue
        request_prompt = sample["prompt"]
        if args.prompt_prefix:
            request_prompt = f"{args.prompt_prefix}\n{request_prompt}"

        prompt_id = sample.get("id")
        label = f"id-{prompt_id}" if prompt_id is not None else sample["source_id"]
        report(f"{name}/{sample['dataset']}/{label}: đang mở hội thoại UAT.", flush=True)
        start_benchmark_conversation(db, client)
        report(f"{name}/{sample['dataset']}/{label}: đã mở hội thoại {client.conversation_id}.", flush=True)
        try:
            for image_index in pending:
                key = (*prompt_key, image_index)
                record = generate_image_slot(
                    db, args.run, client, key, sample, request_prompt, size, asset_ids,
                    images_per_prompt=images_per_prompt, report=report
                )
                if is_chat_not_found(record):
                    report(f"Chat UAT trả 404. Đang thử lại ảnh [{image_index}/{images_per_prompt}] "
                           "trong hội thoại mới.", flush=True)
                    if not cleanup_benchmark_conversations(
                        db, client, conversation_id=client.conversation_id, report=report
                    ):
                        raise ValueError("Còn hội thoại benchmark chưa dọn được. Chạy lại cùng --run để thử xóa.")
                    start_benchmark_conversation(db, client)
                    record = generate_image_slot(
                        db, args.run, client, key, sample, request_prompt, size, asset_ids,
                        images_per_prompt=images_per_prompt, report=report
                    )
                    if is_chat_not_found(record):
                        raise ValueError("Chat UAT tiếp tục trả 404 sau khi đổi hội thoại. Chạy lại cùng --run để tiếp tục.")
                failures += record["status"] == "error"
                if record["status"] == "success":
                    tiles[image_index] = record
            if len(tiles) == images_per_prompt:
                publish_composite(db, args.run, prompt_key, tiles, sample, cleanup_slots=cleanup_slots, report=report)
        finally:
            try:
                if not client.rate_limited:
                    conversations_clean = cleanup_benchmark_conversations(
                        db, client, conversation_id=client.conversation_id, report=report
                    )
            finally:
                client.conversation_id = ""
        if client.rate_limited:
            raise RateLimitReached()
        if not conversations_clean:
            raise ValueError("Còn hội thoại benchmark chưa dọn được. Chạy lại cùng --run để thử xóa.")

    if not count:
        raise ValueError(f"{name} không có mẫu để chạy")
    return failures


def is_chat_not_found(record):
    return record.get("error_code") == "WEB_CHAT_HTTP_ERROR" and record.get("http_status") == 404


def generate_image_slot(db, run, client, key, sample, request_prompt, size, asset_ids, *,
                        images_per_prompt=IMAGES_PER_PROMPT, report=print):
    from PIL import Image

    name, dataset, source_id, image_index = key
    prompt_id = sample.get("id")
    label = f"id-{prompt_id}" if prompt_id is not None else source_id
    slot = f"{name}/{dataset}/{label} [{image_index}/{images_per_prompt}]"
    report(f"{slot}: đang gửi yêu cầu ảnh và chờ UAT trả kết quả.", flush=True)
    result = {}
    try:
        result, image = client.image(request_prompt, size)
        if result["status"] == "success":
            with Image.open(BytesIO(image)) as tile:
                if tile.size != TILE_DIMENSIONS:
                    result.update(status="error", error_code="BAD_IMAGE_SIZE")
                else:
                    tile.load()
        if result["status"] == "success" and result.get("asset_id") in asset_ids:
            result.update(status="error", error_code="DUPLICATE_ASSET")
        if result["status"] == "success":
            publish_tile(key, result, run, image)
    except (OSError, URLError, KeyError, IndexError, TypeError, ValueError) as exc:
        result.update(status="error", error_type=type(exc).__name__)
    record = {**result, "sample": sample, "image_index": image_index, "model": client.model, "size": size}
    record.setdefault("request_prompt", request_prompt)
    store_result(db, key, record)
    if name == "toxicity" and record["status"] == "success":
        publish_toxicity_tile(run, record)
    if record["status"] == "error" or (name in FLAT_ERROR_FOLDERS and record["status"] == "rejected"):
        write_error_metadata(run, key, record)
    if record["status"] == "success" and record.get("asset_id"):
        asset_ids.add(record["asset_id"])
    if record.get("asset_created") is True and (
        record["status"] == "success" or record.get("error_code") == "BAD_IMAGE_SIZE"
    ):
        delete_saved_asset(db, client, key, record)
    details = [f"HTTP {record['http_status']}"] if "http_status" in record else []
    error = record.get("error_code") or record.get("error_type")
    if error:
        details.append(error)
    elapsed_seconds = record.get("elapsed_seconds")
    if elapsed_seconds is not None:
        details.append(f"{elapsed_seconds:.2f}s")
    if record.get("request_id"):
        details.append(f"request_id={record['request_id']}")
    if record.get("asset_deleted_at"):
        details.append("library_deleted")
    elif record["status"] == "success" and record.get("asset_created") is False:
        details.append("library_cleanup_skipped")
    suffix = f" ({', '.join(details)})" if details else ""
    report(f"{slot}: {record['status']}{suffix}", flush=True)
    if record.get("error_code") == "WEB_CHAT_HTTP_ERROR" and record.get("http_status") == 401:
        raise ValueError(
            "Vita Ask UAT từ chối phiên đăng nhập. "
            "Lấy cookie và X-Csrf-Token từ phiên UAT hiện tại rồi chạy lại cùng --run."
        )
    if str(record.get("error_code", "")).lower() == "library_full":
        raise ValueError(
            "Thư viện Vita Design đã đầy. Có thể dùng benchmark.py cleanup --run <tên-lượt> "
            "để dọn ảnh của lượt benchmark đã lưu cục bộ, rồi chạy lại."
        )
    return record


def evaluation_records(db, criterion=None):
    query = "SELECT criterion, dataset, source_id, record FROM results"
    parameters = ()
    if criterion is not None:
        query += " WHERE criterion=?"
        parameters = (criterion,)
    query += " ORDER BY criterion, dataset, source_id"
    for _, group in groupby(db.execute(query, parameters), key=lambda row: row[:3]):
        records = [json.loads(row[3]) for row in group]
        composite = next((r for r in records if r.get("image_index") == COMPOSITE_INDEX), None)
        if composite:
            yield composite
        else:
            legacy = [record for record in records if "image_index" not in record]
            if legacy:
                yield from legacy
            else:
                status = "rejected" if any(r["status"] == "rejected" for r in records) else "error"
                yield {"status": status, "sample": records[0]["sample"]}


def evaluate(args, chosen):
    from evaluators import Evaluators

    path = run_database_path(args.run)
    if not path.is_file():
        raise ValueError(f"Không tìm thấy lượt chạy {args.run}")
    observations = {}
    if args.observations:
        source = (ROOT / args.observations).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError("Tệp observations phải nằm trong HELM-demo")
        with source.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                key = (str(record["criterion"]), record["dataset"], str(record["source_id"]))
                if key in observations:
                    raise ValueError(f"Observation trùng: {key}")
                observations[key] = record

    for var, folder in (("XDG_CACHE_HOME", ".cache"), ("HF_HOME", ".cache/huggingface"),
                        ("TORCH_HOME", ".cache/torch"), ("HELM_CACHE_DIR", ".cache/helm")):
        os.environ[var] = str(ROOT / folder)
    reports = {}
    allowed = {"detection_correct", "nudity_detected"}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        for number, criterion in chosen.items():
            name = criterion["name"]
            rows = list(evaluation_records(db, name))
            if not rows:
                continue
            for row in rows:
                sample = row["sample"]
                extras = observations.get((number, sample["dataset"], sample["source_id"]), {})
                row.update({k: v for k, v in extras.items() if k.startswith("human_") or k in allowed})
            reports[name] = Evaluators(rows).report(criterion["metrics"])
    if not reports:
        raise ValueError("Lượt chạy chưa có mẫu cho các tiêu chí đã chọn")
    report_path = path.with_name("summary.json")
    report = {"run": args.run, "criteria": reports}
    atomic_write(report_path, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(report_path.relative_to(ROOT))
    return 0


def add_generation_arguments(parser):
    parser.add_argument("--run")
    parser.add_argument("--workers", type=int, default=1, help="Số worker sinh ảnh song song, mặc định 1")
    parser.add_argument("--no-worker-terminals", dest="worker_terminals", action="store_false",
                        help="Hiện log của worker trong terminal hiện tại")
    parser.add_argument("--size", choices=["512x512"], help="Kích thước cố định của từng ảnh")
    parser.add_argument("--prompt-prefix", default="", help="Văn bản đặt trước prompt khi gửi API")
    parser.add_argument("--limit", type=int, help="Số prompt tối đa mỗi tiêu chí, giữ trọn nhóm")
    parser.add_argument("--full-datasets", action="store_true", help="Dùng bộ nguồn đầy đủ")
    parser.add_argument("--dry-run", action="store_true")


def add_progress_arguments(parser, *, default_run=None):
    parser.add_argument("--run", default=default_run, required=default_run is None)
    parser.add_argument("--limit", type=int, help="Giới hạn prompt như khi generate, giữ trọn nhóm")
    parser.add_argument("--full-datasets", action="store_true", help="Dùng bộ nguồn đầy đủ")
    parser.add_argument("--allow-missing", action="store_true")


def validate_run_name(parser, run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run):
        parser.error("--run chỉ nhận chữ, số, dấu chấm, gạch dưới và gạch ngang")


def validate_generation_args(parser, args):
    if args.workers < 1:
        parser.error("--workers phải lớn hơn 0")
    if not args.dry_run and not args.run:
        parser.error("Cần --run khi sinh ảnh")
    if args.run:
        validate_run_name(parser, args.run)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")


def validate_progress_args(parser, args):
    validate_run_name(parser, args.run)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("fetch", "prepare", "status"):
        command = commands.add_parser(name)
        command.add_argument("datasets", nargs="*")
        if name == "fetch":
            command.add_argument("--prompts-only", action="store_true", help=argparse.SUPPRESS)
        if name != "status":
            command.add_argument("--jobs", type=int, default=4)
        else:
            command.add_argument("--json", action="store_true")
    commands.add_parser("sample", help="Tạo manifest 2.000 prompt theo cấu hình lấy mẫu")
    command = commands.add_parser("generate", allow_abbrev=False)
    command.add_argument("--criteria", nargs="+", default=[])
    add_generation_arguments(command)
    command.add_argument("--allow-missing", action="store_true")
    command = commands.add_parser("progress", allow_abbrev=False, help="Xem số prompt hoàn tất và số lượt ảnh còn lại")
    command.add_argument("--criteria", nargs="+", default=[])
    add_progress_arguments(command)
    command = commands.add_parser("evaluate", allow_abbrev=False)
    command.add_argument("--criteria", nargs="+", default=[])
    command.add_argument("--run", required=True)
    command.add_argument("--observations", help="JSONL gồm criterion, dataset, source_id và các giá trị chấm")
    command = commands.add_parser("cleanup", allow_abbrev=False, help="Dọn asset Vita Design đã lưu và hội thoại benchmark đang chờ xóa")
    command.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "cleanup":
            validate_run_name(parser, args.run)
            return cleanup(args.run)
        specs = registry()
        if args.action in {"fetch", "prepare", "status"}:
            specs = selected(args.datasets, specs, parser, "Dataset")
            if args.action == "status":
                reports = [dataset_progress(name, spec) for name, spec in specs.items()]
                if args.json:
                    print(json.dumps(reports, ensure_ascii=False, indent=2))
                else:
                    print_progress(reports)
                return 0
            if args.jobs < 1:
                parser.error("--jobs phải lớn hơn 0")
            if args.action == "fetch":
                config = settings()
                for key in ("HF_TOKEN", "HUGGING_FACE_ACCESS_TOKEN"):
                    if config.get(key):
                        os.environ[key] = config[key]
            return dataset_action(args.action, specs, args.jobs)
        if args.action == "sample":
            report = build_sample(specs, coco_only=True)
            print(json.dumps({key: value for key, value in report.items() if key != "coco2014"},
                             ensure_ascii=False, indent=2))
            coco = report["coco2014"]
            print(f"COCO: {coco['covered_classes']}/{coco['total_classes']} lớp. Thiếu: {coco['missing_classes']}")
            print(f"Manifest: {SAMPLE_PATH.relative_to(ROOT)}")
            return 0
        chosen = selected_criteria(args.criteria, parser, evaluation=args.action == "evaluate")
        if args.action == "progress":
            validate_progress_args(parser, args)
            return report_run_progress(args, chosen, specs)
        if args.action == "generate":
            validate_generation_args(parser, args)
            return generate(args, chosen, specs)
        validate_run_name(parser, args.run)
        return evaluate(args, chosen)
    except (RateLimitReached, ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
