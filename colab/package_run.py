"""Create the input ZIP for evaluating a HELM-demo run on Colab."""
import argparse
import json
import re
import sqlite3
import sys
from contextlib import closing
from heapq import nsmallest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import (COMPOSITE_INDEX, IMAGES_PER_PROMPT, composite_dimensions,
                       evaluation_criteria, evaluation_records, image_metadata_path, run_database, store_result)
from evaluators.image_metrics import COCO_VAL2014_ARCHIVE


SOURCES = (
    "benchmark.py", "image_generation.py",
    "configs/criteria.json",
    "datasets/__init__.py", "datasets/coco.py", "datasets/download.py", "datasets/prepare.py",
    "datasets/registry.json", "datasets/sample.py",
    "evaluators/__init__.py", "evaluators/image_metrics.py",
)


def repo_file(location):
    path = (ROOT / location).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Không tìm thấy tệp trong dự án: {location}")
    return path


def shared_alignment_records(catalog):
    for record in evaluation_records(catalog, "alignment"):
        sample = record["sample"]
        if (record["status"] != "success" or sample["dataset"] != "coco2014"
                or sample.get("version") != "val2014"):
            continue
        image = Path(record["image_path"])
        if not (ROOT / image).is_file():
            continue
        metadata = image_metadata_path(image, Path("outputs/benchmark"))
        if record.get("image_index") == COMPOSITE_INDEX and not (ROOT / metadata).is_file():
            continue
        image_id = sample.get("image_id")
        if not sample.get("reference_image") and type(image_id) is int and image_id >= 0:
            reference = {
                "archive": COCO_VAL2014_ARCHIVE,
                "member": f"val2014/COCO_val2014_{image_id:012d}.jpg",
            }
            record = {**record, "sample": {**sample, "reference_image": reference}}
        yield record


def package(run, output, observations, limit=None, *, reuse_alignment=False):
    db_name = f"results/benchmark/{run}/requests.sqlite3"
    db_path = repo_file(db_name)
    criteria = evaluation_criteria()
    with TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        snapshot = temporary / "requests.sqlite3"
        archive_path = temporary / "evaluation.zip"
        with (
            closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as source,
            closing(run_database(snapshot)) as destination,
        ):
            if reuse_alignment:
                source.execute("BEGIN")
                selected = {number: criteria[number]["name"] for number in ("1", "2", "8")}
                records = shared_alignment_records(source)
                images = (list(records) if limit is None
                          else nsmallest(limit, records, key=lambda record: record["image_path"]))
                if not images or (limit is not None and len(images) < limit):
                    raise ValueError(f"Lượt {run} chỉ có {len(images)} ảnh alignment COCO val2014 hoàn tất.")
                successful = [(name, record) for name in selected.values() for record in images]
            else:
                if limit is None:
                    source.backup(destination)
                    catalog = destination
                else:
                    source.execute("BEGIN")
                    catalog = source
                names = {name for name, in catalog.execute("SELECT DISTINCT criterion FROM results")}
                selected = {number: config["name"] for number, config in criteria.items() if config["name"] in names}
                if not selected:
                    raise ValueError(f"Lượt {run} chưa có kết quả cho tiêu chí Colab hỗ trợ")
                successful = (
                    (name, record)
                    for name in selected.values()
                    for record in evaluation_records(catalog, name)
                    if record["status"] == "success"
                )
                if limit is not None:
                    successful = nsmallest(limit, successful, key=lambda pair: pair[1]["image_path"])
                    if len(successful) < limit:
                        raise ValueError(f"Lượt {run} chỉ có {len(successful)} ảnh để đóng gói, cần {limit}.")
                    included = {name for name, _ in successful}
                    selected = {number: name for number, name in selected.items() if name in included}

            if reuse_alignment or limit is not None:
                for name, record in successful:
                    sample = record["sample"]
                    key = (name, sample["dataset"], sample["source_id"], record.get("image_index", 1))
                    store_result(destination, key, record)

            image_counts = {name: 0 for name in selected.values()}
            with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
                archive.write(snapshot, db_name)
                for location in SOURCES:
                    archive.write(repo_file(location), location)
                if observations:
                    observations_path = repo_file(observations)
                    with observations_path.open(encoding="utf-8") as lines:
                        for number, line in enumerate(lines, 1):
                            row = json.loads(line)
                            if not isinstance(row, dict) or not {"criterion", "dataset", "source_id"} <= row.keys():
                                raise ValueError(f"Observations dòng {number} thiếu criterion, dataset hoặc source_id")
                    archive.write(observations_path, "observations.jsonl")

                archived = set()
                for name, record in successful:
                    for location in output_locations(record):
                        if location not in archived:
                            archive.write(repo_file(location), location)
                            archived.add(location)
                    image_counts[name] += 1
                archive.writestr("evaluation.json", json.dumps({
                    "run": run,
                    "criteria": list(selected),
                    "database": db_name,
                    "image_counts": image_counts,
                    "observations": bool(observations),
                }))
        archive_path.replace(output)


def output_locations(record):
    image = Path(record["image_path"])
    if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"Đường dẫn ảnh không hợp lệ: {image}")
    locations = [image.as_posix()]

    metadata = image_metadata_path(image, Path("outputs/benchmark"))
    if record.get("image_index") == COMPOSITE_INDEX:
        if image.suffix.lower() != ".png":
            raise ValueError(f"Ảnh ghép phải là PNG: {image}")
        details = json.loads(repo_file(metadata).read_text(encoding="utf-8"))
        sample = record["sample"]
        width, height = composite_dimensions(record.get("images_per_prompt", IMAGES_PER_PROMPT))
        expected = {
            "image_filename": image.name,
            "dataset": sample["dataset"],
            "source_id": sample["source_id"],
            "width": width,
            "height": height,
        }
        if not isinstance(details, dict) or any(details.get(k) != v for k, v in expected.items()):
            raise ValueError(f"JSON không khớp với ảnh ghép: {metadata}")
        locations.append(metadata.as_posix())
    elif (ROOT / metadata).is_file():
        locations.append(metadata.as_posix())

    reference = record["sample"].get("reference_image")
    if reference and reference["archive"].endswith(".zip") and (ROOT / reference["archive"]).is_file():
        locations.append(reference["archive"])
    return locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int, help="Chỉ đóng gói đúng N ảnh đã hoàn tất để thử trên Colab")
    parser.add_argument("--reuse-alignment", action="store_true",
                        help="Chấm tiêu chí 1, 2 và 8 trên cùng ảnh COCO val2014 đã sinh cho alignment")
    parser.add_argument("--observations", help="Đường dẫn JSONL chấm tay hoặc kết quả detector")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run):
        parser.error("--run chỉ nhận chữ, số, dấu chấm, gạch dưới và gạch ngang")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    suffix = "-criteria-1-2-8" if args.reuse_alignment else ""
    if args.limit is not None:
        suffix += f"-sample-{args.limit}"
    output = ROOT / "outputs" / "benchmark" / f"{args.run}{suffix}-colab-input.zip"
    try:
        package(args.run, output, args.observations, args.limit, reuse_alignment=args.reuse_alignment)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
