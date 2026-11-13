"""Generate HEIM pilot images and score the added aspect-ratio criterion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--criterion', default='01_alignment')
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--limit', type=int, help='First N prompt records in each selected criterion')
    parser.add_argument('--images-per-prompt', type=int, help='Override samples per prompt')
    parser.add_argument('--model', help='Defaults to MUSE2API_MODEL in HELM/.env or muse-image')
    parser.add_argument('--size')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'heim_pilot')
    parser.add_argument('--dry-run', action='store_true', help='Inspect jobs without importing API dependencies or making requests')
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.images_per_prompt is not None and args.images_per_prompt < 1:
        parser.error('limit and images-per-prompt must be positive')
    manifest = json.loads((ROOT / 'prompts/heim_pilot/manifest.json').read_text())
    groups = manifest['criteria']
    if not args.all:
        groups = [group for group in groups if group['id'] == args.criterion]
        if not groups:
            parser.error('Unknown criterion; see prompts/heim_pilot/manifest.json')
    records = []
    for group in groups:
        rows = [json.loads(line) for line in (ROOT / group['file']).read_text().splitlines() if line.strip()]
        records.extend(rows[:args.limit] if args.limit else rows)
    count = sum(args.images_per_prompt or row['samples'] for row in records)
    if args.size and any('requested_ratio' in row for row in records):
        parser.error('The aspect-ratio group uses its recorded sizes; omit --size')
    print(f'{len(records)} prompt records; {count} image requests', flush=True)
    if args.dry_run:
        for row in records:
            print(f"{row['criterion']} | {row['id']} | {args.images_per_prompt or row['samples']} image(s)")
        return
    # Load HELM's configuration before importing the shared image client.
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    sys.path.insert(0, str(ROOT.parent / 'OneIG-Benchmark'))
    from inference import inference
    from aspect_ratio_metrics import measure_ratio, summarize_aspect_ratios
    model = args.model or os.getenv('MUSE2API_MODEL') or 'muse-image'
    size = args.size or os.getenv('MUSE2API_SIZE', '1024x1024')
    model_folder = model.replace('/', '_').replace('\\', '_')
    if model_folder in ('.', '..'):
        parser.error('Invalid model ID')
    output = args.output_dir / model_folder
    output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for row in records:
        row_size = row.get('requested_size') or size
        for sample in range(args.images_per_prompt or row['samples']):
            folder = output / row['criterion']
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{row['id']}__{sample:02d}.png"
            info_path = image_path.with_suffix('.json')
            identity = {'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                        'model': model, 'size': row_size}
            if image_path.exists():
                old = json.loads(info_path.read_text()) if info_path.exists() else {}
                if old.get('status') == 'success' and all(old.get(k) == v for k, v in identity.items()):
                    print(f'Skip existing {image_path}', flush=True)
                    continue
                raise RuntimeError(f'Existing image has different or missing metadata: {image_path}')
            started = time.perf_counter()
            info = {**row, **identity, 'sample': sample, 'output': str(image_path),
                    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                image = inference(row['prompt'], model=model, size=row_size)
                # Measure before saving, without resizing or cropping the image.
                info.update(width=image.width, height=image.height)
                if 'requested_ratio' in row:
                    info['aspect_ratio_result'] = measure_ratio(image.width, image.height, row['requested_ratio'])
                image.save(image_path, format='PNG')
                info['status'] = 'success'
            except Exception as error:
                failures += 1
                info.update(status='error', error=str(error))
            info['elapsed_seconds'] = time.perf_counter() - started
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            with (output / 'results.jsonl').open('a') as log:
                log.write(json.dumps(info, ensure_ascii=False) + '\n')
            print(f"{info['status']}: {image_path} ({info['elapsed_seconds']:.2f}s)", flush=True)
            if info['status'] == 'error' and not args.continue_on_error:
                summarize_aspect_ratios(records, output, args.images_per_prompt)
                raise RuntimeError(info['error'])
    summarize_aspect_ratios(records, output, args.images_per_prompt)
    if failures:
        raise SystemExit(f'{failures} image requests failed; see results.jsonl')


if __name__ == '__main__':
    main()
