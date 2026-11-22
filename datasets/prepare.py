"""Normalize source datasets into stable, versioned prompt manifests."""
import ast
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path, PurePosixPath

from .coco import load_coco, preferred_captions
from .download import DATA, raw_path


class DatasetError(ValueError):
    pass


def prepared_path(dataset, spec):
    return DATA / "prepared" / dataset / spec["version"] / "prompts.jsonl"


def make_record(dataset, spec, source_id, prompt, split="test", **metadata):
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetError(f"{dataset}/{source_id}: thiếu prompt")
    return {"dataset": dataset, "version": spec["version"], "split": split,
            "source_id": str(source_id), "prompt": prompt.strip(), **metadata}


def source_constants(path):
    values = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for item in cls.body:
            if isinstance(item, ast.AnnAssign):
                target, value = item.target, item.value
            elif isinstance(item, ast.Assign) and len(item.targets) == 1:
                target, value = item.targets[0], item.value
            else:
                continue
            if not isinstance(target, ast.Name) or not target.id.isupper() or value is None:
                continue
            try:
                if isinstance(value, ast.Dict):
                    values[target.id] = {ast.literal_eval(key): (
                        values[val.id] if isinstance(val, ast.Name) else ast.literal_eval(val)
                    ) for key, val in zip(value.keys, value.values)}
                else:
                    values[target.id] = ast.literal_eval(value)
            except (ValueError, KeyError):
                continue
    return values


def coco_records(dataset, spec, path):
    for row, labels, described in preferred_captions(*load_coco(path)):
        yield make_record(dataset, spec, row["id"], row["caption"], "val",
                          image_id=row["image_id"], image_classes=labels,
                          described_classes=described)


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise DatasetError(f"Kiểu dữ liệu không hợp lệ trong danh sách CUB: {module}.{name}")


def cub_records(dataset, spec, source):
    with zipfile.ZipFile(source) as birds:
        names = PrimitiveUnpickler(io.BytesIO(birds.read("birds/test/filenames.pickle")),
                                  encoding="latin1").load()
        with zipfile.ZipFile(io.BytesIO(birds.read("birds/text.zip"))) as captions:
            for name in names:
                if not isinstance(name, str) or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                    raise DatasetError("ID ảnh CUB không hợp lệ")
                text = captions.read(f"text/{name}.txt").decode("utf-8")
                for n, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        yield make_record(dataset, spec, f"{name}:{n}", line, image_id=name)


def structured_records(dataset, spec, path):
    form = spec["format"]
    if form == "coco":
        yield from coco_records(dataset, spec, path)
    elif form == "cub":
        yield from cub_records(dataset, spec, path)
    elif form == "winoground":
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for variant in (0, 1):
                    yield make_record(dataset, spec, f"{row['id']}:{variant}",
                                      row[f"caption_{variant}"], image_id=row[f"image_{variant}"],
                                      group_id=str(row["id"]), variant=variant, metadata=row)
    else:
        constants = source_constants(path)
        if form == "syntax":
            for number, (prompt, phenomenon) in enumerate(constants["PROMPT_TO_PHENOMENON"].items(), 1):
                yield make_record(dataset, spec, number, prompt,
                                  category=phenomenon, group_id=phenomenon)
            return
        elif form == "landing":
            prompts = (f"a landing page of a {app} {platform} application"
                       for app in constants["APPLICATION_TYPES"] for platform in constants["PLATFORMS"])
        elif form == "logos":
            prompts = (f"a logo of {description}" for description in constants["COMPANY_DESCRIPTIONS"])
        elif form == "magazine":
            prompts = (f'a photo for a cover of a magazine with the headline: "{headline}"'
                       for headline in constants["HEADLINES"])
        elif form == "historical":
            prompts = constants["HISTORICAL_FIGURES"]
        else:
            prompts = constants["PROMPTS"]
        for number, prompt in enumerate(prompts, 1):
            yield make_record(dataset, spec, number, prompt)


def delimited_records(dataset, spec, path):
    form = spec["format"]
    with path.open(encoding="utf-8-sig", newline="") as source:
        if form == "relational":
            seen = set()
            for i, row in enumerate(csv.DictReader(source), 1):
                prompt = row["Prompt"]
                if prompt not in seen:
                    seen.add(prompt)
                    yield make_record(dataset, spec, i, prompt, group_id=row["Dyad"],
                                      category=row["RelationType"])
            return
        rows = csv.DictReader(source, delimiter="\t" if form == "parti" else ",")
        for i, row in enumerate(rows, 1):
            if form == "drawbench":
                yield make_record(dataset, spec, i, row["Prompts"], category=row["Category"].strip())
            elif form == "parti":
                yield make_record(dataset, spec, i, row["Prompt"], category=row["Category"],
                                  challenge=row["Challenge"], metadata={"note": row["Note"]})
            elif form == "paint":
                values = list(row.values())
                skill, prompt, obj1, obj2, count, relation = values[:6]
                if skill not in {"object", "count", "spatial"}:
                    raise DatasetError(f"PaintSkills: nhóm không hợp lệ {skill}")
                expected = {"object": obj1}
                if skill == "count":
                    expected["count"] = int(count)
                elif skill == "spatial":
                    expected = {"objects": [obj1, obj2], "relation": relation}
                group = ["count", obj1] if skill != "spatial" else ["spatial", *sorted((obj1, obj2))]
                yield make_record(dataset, spec, i, prompt, category=skill, expected=expected,
                                  group_id=json.dumps(group, ensure_ascii=False))
            elif form == "i2p":
                meta = {k: v for k, v in row.items() if k not in {"prompt", "lexica_url"}}
                yield make_record(dataset, spec, i, row["prompt"], "train", categories=row["categories"],
                                  hard=int(row["hard"]), prompt_toxicity=float(row["prompt_toxicity"]),
                                  metadata=meta)


def prepare(dataset, spec):
    components = spec["components"]
    source = raw_path(dataset, spec, components[0])
    if not source.is_file():
        raise DatasetError(f"Chưa có {source.name}")
    form = spec["format"]
    if form in {"drawbench", "parti", "paint", "i2p", "relational"}:
        records = delimited_records(dataset, spec, source)
    else:
        records = structured_records(dataset, spec, source)
    target = prepared_path(dataset, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".jsonl.part")
    ids = set()
    count = 0
    try:
        with part.open("w", encoding="utf-8") as output:
            for row in records:
                source_id = row["source_id"]
                if source_id in ids:
                    raise DatasetError(f"ID nguồn trùng: {source_id}")
                ids.add(source_id)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        expected = spec.get("total")
        if not count or (expected is not None and count != expected):
            raise DatasetError(f"Đọc được {count:,} prompt, tổng dự kiến là {expected}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return count


def progress(dataset, spec, *, sample_path=None, sample_total=None):
    path = sample_path if sample_path is not None else prepared_path(dataset, spec)
    count = 0
    if path.is_file():
        with path.open("rb") as f:
            count = sum(1 for line in f if line.strip()
                        and (sample_path is None or json.loads(line).get("dataset") == dataset))
    total = sample_total if sample_path is not None else spec.get("total")
    components = []
    for component in spec["components"]:
        raw = raw_path(dataset, spec, component)
        complete = raw.is_file()
        source = raw if complete else raw.with_name(raw.name + ".part")
        try:
            size = source.stat().st_size
        except FileNotFoundError:
            size = 0
        size_total = component.get("bytes")
        components.append({"file": raw.name, "complete": complete, "bytes": size,
                           "total_bytes": size_total,
                           "percent": 100.0 if complete else round(size / size_total * 100, 2) if size_total else None})
    return {"dataset": dataset, "prompts": count, "total_prompts": total,
            "prompt_percent": round(count / total * 100, 2) if total else None,
            "components": components}


def generation_path(dataset, spec):
    path = prepared_path(dataset, spec)
    if spec.get("allow_partial") and not path.is_file():
        partial = path.with_suffix(".jsonl.partial")
        if partial.is_file():
            return partial
    return path


def iter_records(dataset, spec):
    path = generation_path(dataset, spec)
    try:
        source = path.open("rb")
    except FileNotFoundError:
        # The producer may have renamed the partial file after path selection.
        path = prepared_path(dataset, spec)
        try:
            source = path.open("rb")
        except FileNotFoundError as error:
            raise DatasetError(f"{dataset} chưa được chuẩn hóa") from error
    with source:
        for line in source:
            if path.suffix == ".partial" and not line.endswith(b"\n"):
                break
            yield json.loads(line)
