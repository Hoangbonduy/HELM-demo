"""Batch-1 benchmark matching HEIM pilot CLIP + aesthetic scoring."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from PIL import Image


class Scorer:
    def __init__(self):
        import clip
        from urllib.request import urlretrieve
        self.clip = clip
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('ViT-L/14', device=self.device)
        self.model.eval()
        self.head = torch.nn.Linear(768, 1).to(self.device)
        weights = Path.home() / '.cache/heim_pilot/sa_0_4_vit_l_14_linear.pth'
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            urlretrieve('https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth', weights)
        self.head.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.head.eval()
        self.cache = {}

    def sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    @torch.no_grad()
    def score(self, path, caption):
        # Decode, preprocess, tokenize and host-to-device copies are outside the
        # inference interval but inside the enclosing pipeline timer.
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        tokens = None
        if caption not in self.cache:
            tokens = self.clip.tokenize([caption], truncate=True).to(self.device)
        if self.device == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            begin = time.perf_counter()
        features = self.model.encode_image(tensor).float()
        features = features / features.norm(dim=-1, keepdim=True)
        if tokens is not None:
            text = self.model.encode_text(tokens).float()
            self.cache[caption] = text / text.norm(dim=-1, keepdim=True)
        similarity = (features @ self.cache[caption].T).clamp(min=0) * 100
        aesthetic = self.head(features)
        if self.device == 'cuda':
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000
        else:
            seconds = time.perf_counter() - begin
        return seconds, similarity.item(), aesthetic.item()


def run_pass(scorer, records, root, repeat):
    rows = []
    scorer.sync()
    start = time.perf_counter()
    for record in records:
        item_start = time.perf_counter()
        # Fail explicitly on any scoring error; never silently score < N images.
        seconds, clip_score, aesthetic_score = scorer.score(root / record['image'], record['caption'])
        rows.append({'repeat': repeat, 'id': record['id'], 'inference_seconds': seconds,
                     'pipeline_seconds': time.perf_counter() - item_start,
                     'clip_score': clip_score, 'aesthetics_score': aesthetic_score})
    scorer.sync()
    elapsed = time.perf_counter() - start
    model_seconds = sum(r['inference_seconds'] for r in rows)
    summary = {'repeat': repeat, 'images': len(rows), 'pipeline_seconds': elapsed,
               'pipeline_images_per_second': len(rows) / elapsed,
               'pipeline_ms_per_image': 1000 * elapsed / len(rows),
               'inference_seconds': model_seconds,
               'inference_images_per_second': len(rows) / model_seconds,
               'inference_ms_per_image': 1000 * model_seconds / len(rows)}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--expected-count', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.expected_count < 1:
        parser.error('Invalid repeats, warmup or expected-count')
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    records = manifest['records']
    if len(records) != args.expected_count or manifest['count'] != len(records):
        raise ValueError('Manifest count differs from expected-count')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs')
    root = args.manifest.parent.resolve()
    print('Validating all files before timing...', flush=True)
    for record in records:
        path = (root / record['image']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path outside dataset')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Image checksum mismatch: {path}')
        with Image.open(path) as image:
            image.load()
    scorer = Scorer()
    print(f'Device: {scorer.device}; warming up {args.warmup} images', flush=True)
    for index in range(args.warmup):
        record = records[index % len(records)]
        scorer.score(root / record['image'], record['caption'])
    scorer.sync()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(1, args.repeats + 1):
        scorer.cache.clear()  # Empty text cache at the start of every measured pass.
        if scorer.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rows, summary = run_pass(scorer, records, root, repeat)
        summary['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated() if scorer.device == 'cuda' else None
        with (args.output / f'per_image_run_{repeat}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        runs.append(summary)
        print(json.dumps(summary), flush=True)
    report = {'configuration': {'batch_size': 1, 'model': 'CLIP ViT-L/14',
              'aesthetic_head': 'LAION sa_0_4_vit_l_14_linear',
              'text_cache': 'empty each pass; reused for identical captions within pass',
              'warmup_images': args.warmup, 'repeats': args.repeats,
              'manifest_sha256': hashlib.sha256(raw).hexdigest()},
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'device': scorer.device,
              'gpu': torch.cuda.get_device_name() if scorer.device == 'cuda' else None,
              'clip_dtype': str(next(scorer.model.parameters()).dtype)},
              'timing': {'inference': 'image encoding, uncached text encoding, normalization, similarity and aesthetic head; CUDA events on GPU, perf_counter on CPU',
              'pipeline': 'local file read/decode, preprocessing, tokenization, transfers, inference, scalar results, timing instrumentation and loop overhead',
              'excluded': 'network downloads, setup, model loading, validation, warmup and output file writes',
              'filesystem_cache': 'warm OS cache: validation reads all images before timing; sequential batch 1 with GPU sync per image'},
              'runs': runs,
              'median': {key: statistics.median(run[key] for run in runs)
                         for key in runs[0] if key not in ('repeat', 'peak_gpu_allocated_bytes')},
              'pipeline_seconds_min': min(r['pipeline_seconds'] for r in runs),
              'pipeline_seconds_max': max(r['pipeline_seconds'] for r in runs)}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    print('Median:', json.dumps(report['median'], indent=2))


if __name__ == '__main__':
    main()
