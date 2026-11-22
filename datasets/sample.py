"""Build the fixed-size input manifest for the configured benchmark."""
import hashlib
import json
from collections import Counter, defaultdict, deque
from itertools import groupby
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

from .coco import caption_classes, class_matchers, load_coco, preferred_captions
from .download import DATA, ROOT, raw_path
from .prepare import DatasetError, iter_records, make_record

COCO_SAMPLE_TOTAL = 2_000
SAMPLE_PATH = DATA / "prepared" / "coco2014-2000prompts" / "prompts.jsonl"
REPORT_PATH = SAMPLE_PATH.with_name("report.json")


def criteria_for():
    return json.loads((ROOT / "configs/criteria.json").read_text(encoding="utf-8"))


def matches_entry(entry, row):
    return (
        row["dataset"] == entry["id"]
        and (not entry.get("categories") or row.get("category") in entry["categories"])
        and (entry.get("filter") != "heim_i2p_hard"
             or (row["hard"] == 1 and row["prompt_toxicity"] < 0.5))
    )


def prompt_key(row):
    return " ".join(row["prompt"].casefold().split())


def group_key(row):
    if "group_id" in row:
        return row["dataset"], "group", row["group_id"]
    return row["dataset"], "prompt", row["source_id"]


def units(rows):
    groups = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return list(groups.values())


def limit_records(rows, limit):
    if limit is None:
        yield from rows
        return
    for _, group in groupby(rows, group_key):
        members = list(group)
        if len(members) > limit:
            break
        yield from members
        limit -= len(members)
        if not limit:
            break


def stratified(groups, quota, stratum, used, rng, *, image_unique=False):
    """Round-robin across populated strata, retaining complete groups."""
    buckets = defaultdict(deque)
    rng.shuffle(groups)
    for group in groups:
        strata = {stratum(row) for row in group}
        if len(strata) != 1:
            raise DatasetError(f"Nhóm thuộc nhiều phân tầng: {group_key(group[0])}")
        buckets[strata.pop()].append(group)
    active = deque(buckets.values())
    result = []
    texts = set(used)
    images = set()
    while active and len(result) < quota:
        bucket = active.popleft()
        while bucket:
            members = bucket.popleft()
            keys = {prompt_key(row) for row in members}
            if keys & texts or len(keys) != len(members):
                continue
            image_ids = {row["image_id"] for row in members} if image_unique else set()
            if image_ids & images or len(result) + len(members) > quota:
                continue
            texts.update(keys)
            images.update(image_ids)
            result.extend(members)
            break
        if bucket:
            active.append(bucket)
    if len(result) != quota:
        raise DatasetError(f"Chỉ chọn được {len(result)}/{quota} prompt nguyên nhóm")
    return result


def select_coco(rows, quota, used, rng, counts):
    rng.shuffle(rows)
    # Prefer images without people when selecting other classes and favor
    # rare image labels when otherwise equally suitable.
    rows.sort(key=lambda row: ("person" in row["image_classes"],
                              max((counts[c] for c in row["image_classes"]), default=0)))
    buckets = {name: deque() for name in counts}
    for row in rows:
        for name in row["described_classes"]:
            buckets[name].append(row)

    chosen = []
    coverage = Counter()
    used = set(used)

    def add(name):
        bucket = buckets[name]
        while bucket:
            row = bucket.popleft()
            key = prompt_key(row)
            if key not in used:
                used.add(key)
                chosen.append(row)
                coverage.update(row["described_classes"])
                return True
        return False

    for name in sorted(counts, key=lambda c: (len(buckets[c]), counts[c], c)):
        if not coverage[name] and not add(name):
            raise DatasetError(f"COCO không có caption mô tả lớp {name}")
    if len(chosen) > quota:
        raise DatasetError(f"Quota COCO {quota} không đủ bao phủ {len(counts)} lớp")
    available = set(counts)
    while len(chosen) < quota and available:
        name = min(available, key=lambda c: (coverage[c], counts[c], c))
        if not add(name):
            available.remove(name)
    if len(chosen) != quota:
        raise DatasetError(f"COCO chỉ chọn được {len(chosen)}/{quota} prompt")
    return chosen


def select_dataset(name, rows, quota, used, rng, config, coco_counts):
    if name in {"relational_understanding", "common_syntactic_processes"} and any(
        "group_id" not in row for row in rows
    ):
        raise DatasetError(f"Chạy prepare {name} để giữ nguyên nhóm đối chứng")
    grouped = units(rows)
    if name == "coco2014":
        if any("described_classes" not in row for row in rows):
            raise DatasetError("Chạy prepare coco2014 để ghép annotation với caption")
        return select_coco(rows, quota, used, rng, coco_counts)
    if name == "winoground":
        if any({row.get("variant") for row in group} != {0, 1} for group in grouped):
            raise DatasetError("Chạy prepare winoground để giữ đủ hai caption")
        return stratified(grouped, quota, lambda row: row["metadata"]["collapsed_tag"], used, rng)
    if name == "paintskills":
        controls = [group for group in grouped if group[0]["category"] != "spatial"]
        spatial = [group for group in grouped if group[0]["category"] == "spatial"]
        if (any(len(group) != 5 for group in controls)
                or any(len(group) != len(set(group[0]["expected"]["objects"])) * 4 for group in spatial)):
            raise DatasetError("Chạy prepare paintskills để giữ nguyên nhóm đối chứng")
        limits = config["paintskills_spatial_groups"]
        result = [row for group in controls for row in group]
        for size, key in ((4, "same_object"), (8, "different_objects")):
            candidates = [group for group in spatial if len(group) == size]
            result.extend(stratified(candidates, limits[key] * size, lambda row: None, used, rng))
        return result
    if len(rows) == quota:
        return [row for group in grouped for row in group]
    if name == "cub200":
        return stratified(grouped, quota, lambda row: row["image_id"].split("/", 1)[0],
                          used, rng, image_unique=True)
    if name == "i2p":
        return stratified(grouped, quota, lambda row: row["categories"].split(",", 1)[0], used, rng)
    return stratified(grouped, quota, lambda row: row.get("category"), used, rng)


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        target = Path(output.name)
        try:
            output.write(data)
            output.close()
            target.replace(path)
        finally:
            target.unlink(missing_ok=True)


def build_sample(specs, *, coco_only=False):
    criteria = criteria_for()
    config = json.loads((ROOT / "configs/sampling.json").read_text(encoding="utf-8"))
    budgets = {"coco2014": COCO_SAMPLE_TOTAL} if coco_only else config["datasets"]
    total = COCO_SAMPLE_TOTAL if coco_only else config["total_prompts"]
    expected = {entry["id"] for criterion in criteria.values() for entry in criterion["datasets"]}
    if not coco_only and (budgets.keys() != expected or sum(budgets.values()) != total):
        raise DatasetError("Ngân sách lấy mẫu không khớp cấu hình tiêu chí hoặc tổng prompt")

    coco = specs["coco2014"]
    component = coco["components"][0]
    captions, image_classes, coco_counts = load_coco(raw_path("coco2014", coco, component))
    rng = Random(config["seed"])
    used = set()
    selected = []
    by_criterion = dict.fromkeys((criterion["name"] for criterion in criteria.values()), 0)
    for name, quota in budgets.items():
        entries = {criterion["name"]: [entry for entry in criterion["datasets"] if entry["id"] == name]
                   for criterion in criteria.values()}
        eligible = []
        records = (
            (make_record(name, coco, row["id"], row["caption"], "val",
                         image_id=row["image_id"], image_classes=labels, described_classes=described)
             for row, labels, described in preferred_captions(captions, image_classes, coco_counts))
            if coco_only else iter_records(name, specs[name])
        )
        for row in records:
            memberships = [key for key, filters in entries.items()
                           if any(matches_entry(entry, row) for entry in filters)]
            if memberships:
                eligible.append({**row, "criteria": memberships})
        rows = select_dataset(name, eligible, quota, used, rng, config, coco_counts)
        if len(rows) != quota:
            raise DatasetError(f"{name}: chọn {len(rows)}/{quota} prompt")
        for row in rows:
            key = prompt_key(row)
            if key in used:
                raise DatasetError(f"Prompt trùng trong tập mẫu: {name}/{row['source_id']}")
            used.add(key)
            for criterion in row["criteria"]:
                by_criterion[criterion] += 1
        selected.extend(rows)

    matchers = class_matchers(coco_counts)
    coco_rows = [row for row in selected if row["dataset"] == "coco2014"]
    proven = {name: [] for name in coco_counts}
    annotated = Counter()
    for row in coco_rows:
        labels = sorted(image_classes[row["image_id"]])
        described = caption_classes(row["prompt"], labels, matchers)
        if labels != row["image_classes"] or described != row["described_classes"]:
            raise DatasetError(f"Metadata COCO không khớp: {row['source_id']}")
        annotated.update(labels)
        for name in described:
            proven[name].append(row["source_id"])

    missing = [name for name, ids in proven.items() if not ids]
    if missing or len(coco_counts) != 80 or len(selected) != total:
        raise DatasetError(f"Độ bao phủ COCO hoặc tổng prompt không đạt. Lớp thiếu: {missing}")
    manifest = "".join(
        json.dumps({"id": index, **row}, ensure_ascii=False, allow_nan=False) + "\n"
        for index, row in enumerate(selected, 1)
    ).encode("utf-8")
    report = {
        "seed": config["seed"],
        "total_prompts": len(selected),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "by_criterion": {key: count for key, count in by_criterion.items() if count} if coco_only else by_criterion,
        "by_dataset": budgets,
        "coco2014": {
            "covered_classes": len(coco_counts) - len(missing),
            "total_classes": len(coco_counts),
            "missing_classes": missing,
            "classes": {name: {"val2014_images": coco_counts[name],
                               "selected_annotated_images": annotated[name],
                               "selected_describing_prompts": len(proven[name]),
                               "example_caption_id": proven[name][0]}
                        for name in coco_counts},
        },
    }
    if coco_only:
        report["source"] = {
            "split": coco["version"],
            "url": component["urls"][0],
            "sha256": component["sha256"],
            "validation_images": coco["total"],
            "validation_captions": len(captions),
            "sampling": "One preferred caption per image using HELM-demo preferred_captions; "
                        "select_coco balances described categories and deduplicates normalized prompt text.",
        }
    write_atomic(SAMPLE_PATH, manifest)
    write_atomic(REPORT_PATH, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    return report
