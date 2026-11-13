"""Evaluate a reduced HEIM pilot from local images; no image generation or API keys."""
import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image

CRITERIA = {
    '01_alignment': '1. Bám prompt',
    '02_photorealism': '2. Độ chân thực',
    '03_aesthetics': '3. Tính thẩm mỹ',
    '04_originality': '4. Tính nguyên bản',
    '05_reasoning': '5. Khả năng suy luận',
    '06_knowledge': '6. Tri thức',
    '08_toxicity': '7. Nội dung độc hại',
    '12_efficiency': '8. Độ hiệu quả',
    '14_style': '9. Phong cách',
    '13_aspect_ratio': '10. Tỷ lệ ảnh',
}
RATING_FIELDS = ('alignment_1_5','photorealism_1_5','aesthetics_1_5',
                 'originality_1_5','reasoning_0_1','knowledge_0_1',
                 'violence_0_1','sexual_0_1','style_1_5','copyright_urls','note')
RATING_BY_GROUP = {
    '01_alignment':['alignment_1_5'], '02_photorealism':['photorealism_1_5'],
    '03_aesthetics':['aesthetics_1_5'], '04_originality':['originality_1_5','copyright_urls'],
    '05_reasoning':['reasoning_0_1'], '06_knowledge':['knowledge_0_1'],
    '08_toxicity':['violence_0_1','sexual_0_1'],
    '14_style':['style_1_5'],
}


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values)/len(values) if values else None


def load_records(root, model='muse-image'):
    root = Path(root)
    manifest = json.loads((root/'prompts/heim_pilot/manifest.json').read_text(encoding='utf-8'))
    target = root/'outputs/heim_pilot'/model.replace('/','_').replace('\\','_')
    if not target.is_dir():
        raise FileNotFoundError(f'Model output directory not found: {target}')
    records = []
    for group in manifest['criteria']:
        if group['id'] not in CRITERIA:
            continue
        for line in (root/group['file']).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for sample in range(row['samples']):
                image = target/row['criterion']/f"{row['id']}__{sample:02d}.png"
                sidecar = image.with_suffix('.json')
                record = dict(row, sample=sample, image_path=str(image), status='not_run')
                if sidecar.exists():
                    try:
                        info = json.loads(sidecar.read_text(encoding='utf-8'))
                        record['elapsed_seconds'] = info.get('elapsed_seconds')
                        record['error'] = info.get('error')
                        if info.get('status') != 'success':
                            record['status'] = 'generation_failed'
                        elif info.get('prompt_sha256') != hashlib.sha256(row['prompt'].encode()).hexdigest():
                            record['status'] = 'metadata_mismatch'
                        elif info.get('model') != model:
                            record['status'] = 'metadata_mismatch'
                        else:
                            with Image.open(image) as im:
                                im.load()
                                record['width'],record['height'] = im.size
                            record['status'] = 'success'
                    except (OSError,ValueError,KeyError) as error:
                        record['status'],record['error'] = 'unreadable',str(error)
                elif image.exists():
                    record['status'] = 'missing_metadata'
                records.append(record)
    return records, target


class HeimClip:
    """CLIP ViT-L/14 + LAION linear aesthetic head used in HEIM (pilot implementation)."""
    def __init__(self):
        import clip
        import torch
        from urllib.request import urlretrieve
        self.clip,self.torch = clip,torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model,self.preprocess = clip.load('ViT-L/14',device=self.device)
        self.model.eval()
        self.linear = torch.nn.Linear(768,1).to(self.device)
        cache_dir = Path(os.environ.get('HEIM_PILOT_CACHE',str(Path.home()/'.cache/heim_pilot')))
        cache_dir.mkdir(parents=True,exist_ok=True)
        weights = cache_dir/'sa_0_4_vit_l_14_linear.pth'
        if not weights.exists():
            urlretrieve('https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth',weights)
        self.linear.load_state_dict(torch.load(weights,map_location=self.device,weights_only=True))
        self.linear.eval()
        self.cache = {}

    def _text(self, prompt):
        if prompt not in self.cache:
            tokens = self.clip.tokenize([prompt],truncate=True).to(self.device)
            with self.torch.no_grad():
                t = self.model.encode_text(tokens).float()
                self.cache[prompt] = t/t.norm(dim=-1,keepdim=True)
        return self.cache[prompt]

    def score(self, image_path, caption, bias=False):
        with Image.open(image_path) as im:
            image = self.preprocess(im.convert('RGB')).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model.encode_image(image).float()
            feat = feat / feat.norm(dim=-1,keepdim=True)
            similarity = lambda caption: float((feat @ self._text(caption).T).item())
            result = {'clip_score':max(100 * similarity(caption),0.0),
                      'aesthetics_score':float(self.linear(feat).item())}
            if bias:
                result['perceived_female'] = similarity('a photo of a female') > similarity('a photo of a male')
            return result


def score_records(records, scorer):
    for r in records:
        if r['criterion'] not in CRITERIA or r['status'] != 'success':
            continue
        # Style is scored by humans against the requested style, not by CLIP.
        if r['criterion'] == '14_style':
            continue
        try:
            r.update(scorer.score(r['image_path'], r['prompt']))
            r['clip_scoring_prompt'] = r['prompt']
        except Exception as error:
            r['metric_error'] = str(error)
    return records


def estimate_monk_tone(image_path):
    """HEIM's skin pixel mask and nearest Monk RGB tone, vectorized."""
    import cv2
    import numpy as np
    bgr = cv2.imread(str(image_path),3)
    if bgr is None:
        raise ValueError('Cannot decode image for skin-tone analysis')
    rgba = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGBA).astype('float32')
    ycc = cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb).astype('float32')
    r,g,b,a = [rgba[:,:,i] for i in range(4)]
    y,cr,cb = [ycc[:,:,i] for i in range(3)]
    mask=(r>95)&(g>40)&(b>20)&(r>g)&(r>b)&(np.abs(r-g)>15)&(a>15)&(cr>135)&(cb>85)&(y>80)
    mask &= (cr<=1.5862*cb+20)&(cr>=0.3448*cb+76.2069)&(cr>=-4.5652*cb+234.5652)
    mask &= (cr<=-1.15*cb+301.75)&(cr<=-2.2857*cb+432.85)
    if not mask.any():
        return None
    tones=np.array([(246,237,228),(243,231,219),(247,234,208),(234,218,186),(215,189,150),
                    (160,126,86),(130,92,67),(96,65,52),(58,49,42),(41,36,32)])
    rgb=rgba[:,:,:3][mask].mean(axis=0)
    return int(np.abs(tones-rgb).sum(axis=1).argmin()+1)


def write_ratings_template(records, path):
    path = Path(path)
    if path.exists():
        print(f'Giữ nguyên phiếu chấm hiện có: {path}')
        return
    fields = ['criterion','id','sample','image_path','prompt','annotator']+list(RATING_FIELDS)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in records:
            if row['status']=='success' and row['criterion'] in RATING_BY_GROUP:
                writer.writerow({**row,'image_path':str(Path(row['image_path']))})


def read_ratings(path):
    result=[]
    if not Path(path).exists():
        return result
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('id'):
                continue
            for field in RATING_FIELDS:
                if field.endswith(('_1_5','_0_1')) and row.get(field,'').strip():
                    try: value=float(row[field])
                    except ValueError: raise ValueError(f"Điểm không hợp lệ cho {row['id']}: {field}")
                    allowed = (1,2,3,4,5) if field.endswith('_1_5') else (0,1)
                    if value not in allowed:
                        raise ValueError(f"Điểm không hợp lệ cho {row['id']}: {field}")
            result.append(row)
    return result


def build_report(records, ratings):
    records = [r for r in records if r['criterion'] in CRITERIA]
    groups = defaultdict(list)
    for row in records:
        groups[row['criterion']].append(row)
    by_human=defaultdict(list)
    valid_keys={(r['criterion'],r['id'],str(r['sample'])) for r in records if r['status']=='success'}
    for row in ratings:
        key=(row.get('criterion'),row.get('id'),str(row.get('sample')))
        if key not in valid_keys:
            continue
        for field in RATING_BY_GROUP.get(row['criterion'], []):
            if field.endswith(('_1_5','_0_1')) and row.get(field,'').strip():
                by_human[(row['criterion'],field)].append(float(row[field]))
    report={'scope':'Custom pilot: nine selected criteria plus aspect ratio; not the full HEIM leaderboard.',
      'limitations':['No real image references were packaged; FID/LPIPS/PSNR and full HEIM photorealism comparison are unavailable.',
        'Manual ratings and small pilot samples are not HEIM leaderboard results. CLIP similarity is not accuracy; this implementation does not reproduce HEIM black-image filtering.',
        'Toxicity review requires humans; nudity/violence are separate and API refusals are not images.',
        'Style is a custom human-rated criterion (1-5) for adherence to the requested style; it is not an official HEIM metric. Missing style data or ratings are not replaced with automatic scores.',
        'API response time includes network, queue, retries and upstream time; not HEIM denoised runtime.'],
      'criteria':{}}
    for group,label in CRITERIA.items():
        rows=groups[group]
        valid=[r for r in rows if r['status']=='success']
        metrics={'images_planned':len(rows),'images_readable':len(valid),
                 'failed_missing_or_mismatch':len(rows)-len(valid),
                 'mean_clip_score':mean([r.get('clip_score') for r in valid]),
                 'mean_aesthetics_predictor':mean([r.get('aesthetics_score') for r in valid]),
                 'metric_errors':sum('metric_error' in r for r in valid)}
        metrics['human_ratings']={field:{'mean':mean(by_human[(group,field)]),
                                  'responses':len(by_human[(group,field)])}
             for field in RATING_BY_GROUP.get(group,[]) if field.endswith(('_1_5','_0_1'))}
        status = 'partial' if valid else 'pending_data'
        if group == '13_aspect_ratio' and valid:
            status = 'ok'
        report['criteria'][group]={'name':label,'status':status,**metrics}
    style = report['criteria']['14_style']
    style_rating = style['human_ratings']['style_1_5']
    style['method'] = 'Human rating 1-5: adherence to the explicitly requested style; custom pilot criterion.'
    if style['images_readable'] and style_rating['responses']:
        style['status'] = ('ok' if style_rating['responses'] >= style['images_readable'] else 'partial')
    elif style['images_readable']:
        style['status'] = 'pending_human_ratings'
    else:
        style['status'] = 'pending_data'
    counts=defaultdict(lambda:{'planned':0,'valid':0,'correct':0,'failed_or_missing':0})
    for r in groups['13_aspect_ratio']:
        c=counts[r['requested_ratio']]; c['planned']+=1
        if r['status']=='success':
            a,b=map(int,r['requested_ratio'].split(':'))
            c['valid']+=1; c['correct']+=int(r['width']*b==r['height']*a)
        else: c['failed_or_missing']+=1
    overall={'planned':0,'valid':0,'correct':0,'failed_or_missing':0}
    for c in counts.values():
        for key in overall: overall[key]+=c[key]
        c['accuracy_percent']=100*c['correct']/c['valid'] if c['valid'] else None
    overall['accuracy_percent']=100*overall['correct']/overall['valid'] if overall['valid'] else None
    report['criteria']['13_aspect_ratio'].update(overall=overall,by_ratio=dict(counts),method='Exact W*b==H*a; original output pixels')
    times=[r['elapsed_seconds'] for r in groups['01_alignment'] if r['status']=='success' and isinstance(r.get('elapsed_seconds'),(int,float))]
    report['criteria']['12_efficiency'].update(status='ok' if times else 'pending_data',n=len(times),
        mean_api_seconds=mean(times),median_api_seconds=statistics.median(times) if times else None)
    return report


def save_report(records, ratings_path, result_dir):
    result_dir=Path(result_dir); result_dir.mkdir(parents=True,exist_ok=True)
    records = [r for r in records if r['criterion'] in CRITERIA]
    report=build_report(records,read_ratings(ratings_path))
    (result_dir/'heim_pilot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (result_dir/'per_image.jsonl').open('w',encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with (result_dir/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['criterion','name','status','images_planned','images_readable',
            'failed_missing_or_mismatch','mean_clip_score','mean_aesthetics_predictor','human_ratings','additional'])
        writer.writeheader()
        for criterion,details in report['criteria'].items():
            additional={key:value for key,value in details.items() if key not in writer.fieldnames}
            writer.writerow({'criterion':criterion, **{key:details.get(key) for key in writer.fieldnames if key in details},
                'human_ratings':json.dumps(details['human_ratings'],ensure_ascii=False),
                'additional':json.dumps(additional,ensure_ascii=False)})
    return report
