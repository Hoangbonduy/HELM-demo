"""Image scoring through HEIM's scorers and TorchMetrics distributions."""
import math
import re
import zipfile
from contextlib import ExitStack
from functools import cached_property
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COCO_VAL2014_ARCHIVE = "datasets/raw/coco2014/val2014/val2014.zip"
COCO_VAL2014_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"


class MetricUnavailable(ValueError):
    pass


def tensor_image(path_or_file):
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(path_or_file) as image:
        return pil_to_tensor(image.convert("RGB")).unsqueeze(0)


class ImageMetrics:
    @cached_property
    def clip(self):
        from helm.clients.clip_scorers.clip_scorer import CLIPScorer
        return CLIPScorer()

    @cached_property
    def aesthetics(self):
        from helm.benchmark.metrics.image_generation.aesthetics_scorer import AestheticsScorer
        return AestheticsScorer()

    @cached_property
    def watermark(self):
        from helm.benchmark.metrics.image_generation.watermark.watermark_detector import WatermarkDetector
        return WatermarkDetector()

    @cached_property
    def nsfw(self):
        from helm.benchmark.metrics.image_generation.nsfw_detector import NSFWDetector
        return NSFWDetector()

    def clip_score(self, row):
        return float(self.clip.compute_score(caption=row["sample"]["prompt"],
                                             image_location=row["image_path"]))

    def multilingual_clip_scores(self, rows):
        from PIL import Image
        from sentence_transformers import SentenceTransformer

        image_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
        text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
        batch_size = 16
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            with ExitStack() as stack:
                images = []
                for row in batch:
                    with Image.open(row["image_path"]) as image:
                        images.append(stack.enter_context(image.convert("RGB")))
                image_vectors = image_model.encode(images, normalize_embeddings=True, show_progress_bar=False)
            prompts = [row["sample"]["prompt"] for row in batch]
            text_vectors = text_model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
            cosine = (image_vectors * text_vectors).sum(axis=1).clip(min=0, max=1)
            yield from (float(score) for score in 100 * cosine)

    def aesthetics_score(self, row):
        return float(self.aesthetics.compute_aesthetics_score(row["image_path"]))

    def fractal_coefficient(self, row):
        from helm.benchmark.metrics.image_generation.fractal_dimension.fractal_dimension_util import (
            compute_fractal_dimension,
        )
        return abs(float(compute_fractal_dimension(row["image_path"])) - 1.4)

    def watermark_rate(self, row):
        return bool(self.watermark.has_watermark([row["image_path"]])[0])

    def nsfw_rate(self, row):
        return bool(self.nsfw.is_nsfw(row["image_path"]))

    def blackout_rate(self, row):
        from PIL import Image
        with Image.open(row["image_path"]) as image:
            return all(high == 0 for _, high in image.convert("RGB").getextrema())

    def distribution(self, metric, rows):
        if metric == "inception_score":
            from torchmetrics.image.inception import InceptionScore
            if len(rows) < 2:
                raise MetricUnavailable("Inception Score cần ít nhất hai ảnh")
            scorer = InceptionScore(splits=max(1, min(10, len(rows) // 2)))
            for row in rows:
                scorer.update(tensor_image(row["image_path"]))
            score, std = scorer.compute()
            return {"value": float(score), "std": float(std) if math.isfinite(float(std)) else None,
                    "n": len(rows), "expected": len(rows)}

        references = {tuple(ref.values()): ref for row in rows
                      if (ref := row["sample"].get("reference_image")) and ref["archive"].endswith(".zip")}
        if len(references) < 2:
            raise MetricUnavailable("FID cần ảnh thật tham chiếu. Dataset này chỉ có prompt.")
        from torchmetrics.image.fid import FrechetInceptionDistance
        scorer = FrechetInceptionDistance()
        with ExitStack() as stack:
            archives = {}
            for ref in references.values():
                arc = ref["archive"]
                member = ref["member"]
                if arc not in archives:
                    path = ROOT / arc
                    archives[arc] = stack.enter_context(zipfile.ZipFile(path)) if path.is_file() else None
                archive = archives[arc]
                if archive is not None:
                    image = archive.read(member)
                elif arc == COCO_VAL2014_ARCHIVE and re.fullmatch(r"val2014/COCO_val2014_[0-9]{12}\.jpg", member):
                    with urlopen(f"{COCO_VAL2014_BASE_URL}/{member}", timeout=30) as response:
                        image = response.read()
                else:
                    raise MetricUnavailable(f"Thiếu ảnh tham chiếu: {arc}")
                scorer.update(tensor_image(BytesIO(image)), real=True)
            for row in rows:
                scorer.update(tensor_image(row["image_path"]), real=False)
        return {"value": float(scorer.compute()), "n": len(rows), "references": len(references)}
