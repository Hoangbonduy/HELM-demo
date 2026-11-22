"""Download a deterministic COCO validation subset; never generate images."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ANNOTATIONS = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'


def fetch(url, target):
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'HEIM-runtime-test/1.0'})
            with urllib.request.urlopen(request, timeout=90) as src, part.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            part.replace(target)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def inspect(path):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError('Empty image')
    return width, height, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.count <= 5000 or args.workers < 1:
        parser.error('count must be 1..5000 and workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / 'images'
    images_dir.mkdir(exist_ok=True)
    annotation = args.output / 'captions_val2017.json'
    if not annotation.exists():
        archive = args.output / 'annotations_trainval2017.zip'
        if not archive.exists():
            print('Downloading official annotations (~253 MB)', flush=True)
            fetch(ANNOTATIONS, archive)
        with zipfile.ZipFile(archive) as source:
            annotation.write_bytes(source.read('annotations/captions_val2017.json'))
    source = json.loads(annotation.read_text())
    captions = {}
    for row in sorted(source['annotations'], key=lambda r: r['id']):
        captions.setdefault(row['image_id'], row['caption'])
    candidates = sorted(source['images'], key=lambda r: r['id'])
    random.Random(args.seed).shuffle(candidates)

    def download(row):
        path = images_dir / row['file_name']
        url = 'https://s3.amazonaws.com/images.cocodataset.org/val2017/' + row['file_name']
        try:
            if path.exists():
                try:
                    inspect(path)
                except Exception:
                    path.unlink()
            if not path.exists():
                fetch(url, path)
            width, height, digest = inspect(path)
            return {'id': row['id'], 'image': 'images/' + row['file_name'],
                    'caption': captions[row['id']], 'url': url,
                    'width': width, 'height': height, 'sha256': digest}
        except Exception as error:
            return {'id': row['id'], 'error': str(error), 'url': url}

    records, errors, hashes = [], [], set()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        # Map preserves the seeded order, independent of download completion order.
        # Download only the number still needed; replace failed or duplicate files.
        position = 0
        while len(records) < args.count and position < len(candidates):
            needed = args.count - len(records)
            batch = candidates[position:position + needed]
            position += len(batch)
            for result in pool.map(download, batch):
                if 'error' in result:
                    errors.append(result)
                elif result['sha256'] not in hashes:
                    records.append(result)
                    hashes.add(result['sha256'])
            print(f'Valid unique images: {len(records)}/{args.count}', flush=True)
    (args.output / 'download_errors.json').write_text(json.dumps(errors, indent=2))
    if len(records) != args.count:
        raise RuntimeError(f'Only {len(records)} valid images; retry to resume downloads')
    manifest = {'dataset': 'COCO val2017', 'seed': args.seed, 'count': len(records),
                'caption_policy': 'lowest annotation ID for each image',
                'records': records}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Ready:', args.output / 'manifest.json')


if __name__ == '__main__':
    main()
