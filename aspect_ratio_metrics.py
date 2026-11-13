"""Exact aspect-ratio accuracy measured on original generated image dimensions."""
import json
from pathlib import Path


def measure_ratio(width, height, requested_ratio):
    a, b = map(int, requested_ratio.split(':'))
    if min(width, height, a, b) <= 0:
        raise ValueError('Image dimensions and ratio must be positive')
    return {'width': width, 'height': height, 'requested_ratio': requested_ratio,
            'correct': width * b == height * a}


def summarize_aspect_ratios(records, output, samples_override=None):
    from PIL import Image

    totals = {'planned': 0, 'valid': 0, 'correct': 0, 'failed_or_unreadable': 0, 'not_run': 0}
    by_ratio = {}
    details = []
    for row in records:
        if 'requested_ratio' not in row:
            continue
        ratio = row['requested_ratio']
        counts = by_ratio.setdefault(ratio, {key: 0 for key in totals})
        for sample in range(samples_override or row['samples']):
            counts['planned'] += 1
            image_path = Path(output) / row['criterion'] / f"{row['id']}__{sample:02d}.png"
            metadata_path = image_path.with_suffix('.json')
            detail = {'id':row['id'], 'sample':sample, 'requested_ratio':ratio, 'image':str(image_path)}
            if not image_path.exists() and not metadata_path.exists():
                counts['not_run'] += 1
                detail['status'] = 'not_run'
            else:
                try:
                    metadata = json.loads(metadata_path.read_text())
                    if metadata.get('status') != 'success':
                        raise ValueError('Generation did not succeed: ' + str(metadata.get('error', 'unknown error')))
                    if metadata.get('prompt') != row['prompt']:
                        raise ValueError('Saved prompt differs from the current prompt record')
                    with Image.open(image_path) as image:
                        image.load()
                        result = measure_ratio(image.width, image.height, ratio)
                    detail.update(status='valid', **result)
                    counts['valid'] += 1
                    counts['correct'] += int(result['correct'])
                except (OSError, ValueError, KeyError) as error:
                    counts['failed_or_unreadable'] += 1
                    detail.update(status='failed_or_unreadable', error=str(error))
            details.append(detail)
    if not by_ratio:
        return None
    for counts in by_ratio.values():
        for key in totals:
            totals[key] += counts[key]
        counts['accuracy_percent'] = 100 * counts['correct'] / counts['valid'] if counts['valid'] else None
    totals['accuracy_percent'] = 100 * totals['correct'] / totals['valid'] if totals['valid'] else None
    report = {'metric':'aspect_ratio_accuracy', 'definition':'W*b == H*a for requested a:b; no tolerance',
              'denominator':'Only successfully generated, readable images; errors reported separately',
              'overall':totals, 'by_ratio':by_ratio, 'images':details}
    report_path = Path(output) / 'aspect_ratio_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"Aspect ratio: {totals['correct']}/{totals['valid']} valid images; report: {report_path}", flush=True)
    return report
