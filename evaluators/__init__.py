"""Evaluation dispatch by criterion, with aggregate metrics and evidence counts."""
from collections import Counter
from functools import cached_property
from statistics import mean

from .image_metrics import ImageMetrics, MetricUnavailable


def numeric(rows, field):
    return [row[field] for row in rows if type(row.get(field)) in (int, float)]


def result(values, expected):
    return {"value": mean(values) if values else None, "n": len(values), "expected": expected}


def observed(rows, field, kind):
    return [row[field] for row in rows if type(row.get(field)) is kind]


class Evaluators:
    def __init__(self, rows):
        self.rows = rows
        self.images = [row for row in rows if row["status"] == "success"]
        self.detector_cases = [row for row in self.images if row["sample"].get("expected")]

    @cached_property
    def image_metrics(self):
        return ImageMetrics()

    def compute(self, metric):
        if metric.startswith("human_"):
            scores = numeric(self.images, metric)
            upper = 3 if metric == "human_subject_clarity" else 5
            return result([score for score in scores if score in range(1, upper + 1)], len(self.images))
        if metric == "raw_inference_time":
            return result(numeric(self.images, "elapsed_seconds"), len(self.images))
        if metric == "denoised_inference_time":
            return result(numeric(self.images, "provider_inference_seconds"), len(self.images))
        if metric == "api_rejection_rate":
            requests = [x for x in self.rows if x["status"] in ("success", "rejected")]
            return result([x["status"] == "rejected" for x in requests], len(self.rows))
        if metric == "detection_accuracy":
            return result(observed(self.detector_cases, "detection_correct", bool), len(self.detector_cases))
        if metric == "nudity_rate":
            return result(observed(self.images, "nudity_detected", bool), len(self.images))
        if not self.images:
            return result([], 0)
        if metric == "multilingual_clip_score":
            return result(list(self.image_metrics.multilingual_clip_scores(self.images)), len(self.images))
        if metric in ("fid", "inception_score"):
            return self.image_metrics.distribution(metric, self.images)
        score_image = getattr(self.image_metrics, metric)
        return result([score_image(row) for row in self.images], len(self.images))

    def report(self, metrics):
        scores = {}
        for metric in metrics:
            try:
                scores[metric] = self.compute(metric)
            except (ImportError, OSError, MetricUnavailable) as exc:
                scores[metric] = {"value": None, "error": str(exc)}
        return {"samples": len(self.rows),
                "by_status": dict(Counter(row["status"] for row in self.rows)),
                "metrics": scores}
