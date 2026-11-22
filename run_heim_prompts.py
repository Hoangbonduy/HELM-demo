"""Generate HEIM pilot images and score the added aspect-ratio criterion."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import time

from image_generation import DEFAULT_SIZE, ROOT, ImageClient, atomic_write, settings


def save_info(path, info):
    atomic_write(path, (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def finish_library_cleanup(client, image_path, info_path, info):
    if info.get("asset_created") is not True or info.get("asset_deleted_at"):
        return
    from PIL import Image

    with Image.open(image_path) as image:
        image.load()
    info["asset_deleted_at"] = client.delete_asset(info["asset_id"])
    save_info(info_path, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--criterion', default='01_alignment')
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--limit', type=int, help='First N prompt records in each selected criterion')
    parser.add_argument('--images-per-prompt', type=int, help='Override samples per prompt')
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
    from PIL import Image
    from aspect_ratio_metrics import measure_ratio, summarize_aspect_ratios

    client = ImageClient(settings())
    model = client.model
    size = args.size or DEFAULT_SIZE
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
                    finish_library_cleanup(client, image_path, info_path, old)
                    print(f'Skip existing {image_path}', flush=True)
                    continue
                raise RuntimeError(f'Existing image has different or missing metadata: {image_path}')
            started = time.perf_counter()
            info = {**row, **identity, 'sample': sample, 'output': str(image_path),
                    'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                result, image_bytes = client.image(row['prompt'], row_size)
                info['request_prompt'] = result['request_prompt']
                if result['status'] != 'success':
                    details = [result.get('http_status'), result.get('error_code')]
                    raise RuntimeError(' '.join(str(detail) for detail in details if detail) or result['status'])
                with Image.open(BytesIO(image_bytes)) as image:
                    # Measure before saving, without resizing or cropping the image.
                    info.update(width=image.width, height=image.height)
                    if 'requested_ratio' in row:
                        info['aspect_ratio_result'] = measure_ratio(image.width, image.height, row['requested_ratio'])
                    saved_image = BytesIO()
                    image.save(saved_image, format='PNG')
                    atomic_write(image_path, saved_image.getvalue())
                info.update(status='success', asset_id=result.get('asset_id'),
                            asset_created=result.get('asset_created', False))
            except Exception as error:
                failures += 1
                info.update(status='error', error=str(error))
            info['elapsed_seconds'] = time.perf_counter() - started
            save_info(info_path, info)
            if info['status'] == 'success':
                finish_library_cleanup(client, image_path, info_path, info)
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
