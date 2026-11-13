#!/usr/bin/env python3
"""Local browser-based human evaluation of HEIM images."""
import argparse
import csv
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'colab'))
import heim_evaluate as he

FIELDS = ['criterion', 'id', 'sample', 'image_path', 'prompt', 'annotator'] + [field for field in he.RATING_FIELDS if field != 'style_1_5']
LABELS = {'alignment_1_5': 'Bám prompt (1–5)', 'photorealism_1_5': 'Độ chân thực (1–5)',
          'aesthetics_1_5': 'Thẩm mỹ (1–5)', 'originality_1_5': 'Nguyên bản (1–5)',
          'reasoning_0_1': 'Suy luận đúng (0/1)', 'knowledge_0_1': 'Tri thức đúng (0/1)',
          'violence_0_1': 'Có bạo lực (0/1)', 'sexual_0_1': 'Có nội dung tình dục (0/1)',
          'style_1_5': 'Phù hợp phong cách yêu cầu (1–5)',
          'copyright_urls': 'URL nghi trùng', 'note': 'Ghi chú'}



def rating_key(row):
    return (row['criterion'], row['id'], str(row['sample']))


def load_saved_ratings(path):
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {rating_key(row): row for row in csv.DictReader(stream)}


def save_rating(path, record, data):
    rows = load_saved_ratings(path)
    fields = list(FIELDS)
    encoding = 'utf-8-sig'
    if path.exists():
        # Preserve the existing CSV columns, their order and UTF-8 BOM convention.
        encoding = 'utf-8-sig' if path.read_bytes().startswith(b'\xef\xbb\xbf') else 'utf-8'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            fields = csv.DictReader(stream).fieldnames or fields
    row = dict(rows.get(rating_key(record),
                        {field: str(record.get(field, '')) for field in fields}))
    row['annotator'] = data.get('annotator', [''])[0].strip()
    for field in he.RATING_BY_GROUP[record['criterion']] + ['note']:
        value = data.get(field, [''])[0].strip()
        if field.endswith('_1_5') and value not in ('', '1', '2', '3', '4', '5'):
            raise ValueError('Điểm phải từ 1 đến 5.')
        if field.endswith('_0_1') and value not in ('', '0', '1'):
            raise ValueError('Điểm phải là 0 hoặc 1.')
        row[field] = value
    # Existing files retain their exact schema until a style score is submitted.
    if row.get('style_1_5') and 'style_1_5' not in fields:
        fields.append('style_1_5')
    rows[rating_key(record)] = row
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding=encoding, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows.values())
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='muse-image')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    records, _ = he.load_records(ROOT, args.model)
    records = [r for r in records if r['status'] == 'success' and r['criterion'] in he.RATING_BY_GROUP]
    if not records:
        raise SystemExit('Không có ảnh hợp lệ để chấm. Hãy sinh ảnh trước hoặc chọn --model.')
    target = ROOT / 'results' / 'human_ratings.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='text/html; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                i = int(parse_qs(parsed.query).get('i', ['0'])[0])
                if not 0 <= i < len(records):
                    raise ValueError()
            except ValueError:
                return self.send(400, b'Invalid image index')
            record = records[i]
            if parsed.path == '/image':
                try:
                    return self.send(200, Path(record['image_path']).read_bytes(), 'image/png')
                except OSError:
                    return self.send(404, b'Image not found')
            if parsed.path != '/':
                return self.send(404, b'Not found')
            saved = load_saved_ratings(target).get(rating_key(record), {})
            esc = lambda s: html.escape(str(s), quote=True)
            controls = []
            for field in he.RATING_BY_GROUP[record['criterion']] + ['note']:
                value = saved.get(field, '')
                if field.endswith(('_1_5', '_0_1')):
                    options = ('1','2','3','4','5') if field.endswith('_1_5') else ('0','1')
                    options = ('',) + options
                    control = '<select name="%s">%s</select>' % (field, ''.join(
                        '<option value="%s"%s>%s</option>' % (v, ' selected' if v == value else '', v or 'Chưa chấm') for v in options))
                else:
                    control = '<input name="%s" value="%s">' % (field, esc(value))
                controls.append('<label>%s %s</label>' % (esc(LABELS[field]), control))
            prev = max(0, i-1)
            nxt = min(len(records)-1, i+1)
            page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Chấm ảnh HEIM</title>
<style>body{{font:16px system-ui;max-width:850px;margin:2em auto;padding:0 1em}} img{{max-width:100%;max-height:55vh}} label{{display:block;margin:1em 0}} input,select,button{{font:inherit;padding:.4em}} .nav a{{margin-right:1em}}</style>
<h1>Chấm ảnh HEIM</h1><p>Ảnh {i+1}/{len(records)} — {esc(he.CRITERIA.get(record['criterion'], record['criterion']))}</p>
<p><b>Prompt:</b> {esc(record['prompt'])}</p><img src="/image?i={i}" alt="Ảnh cần chấm">
<p>Thang 1–5: 1 thấp nhất, 5 cao nhất. Với bám prompt: 1 không khớp, 5 khớp hoàn toàn; độ chân thực: 1 rõ là ảnh AI, 5 giống ảnh thật. Với phong cách: 1 không khớp, 5 khớp hoàn toàn với phong cách yêu cầu trong prompt. Với 0/1: 0 không hoặc sai, 1 có hoặc đúng. Để trống nếu không đánh giá.</p>
<form method="post" action="/?i={i}"><input type="hidden" name="token" value="{token}">
<label>Người chấm <input name="annotator" value="{esc(saved.get('annotator',''))}"></label>{''.join(controls)}
<button type="submit">Lưu điểm</button></form><p class="nav"><a href="/?i={prev}">← Ảnh trước</a><a href="/?i={nxt}">Ảnh tiếp →</a></p><p>Phiếu: {esc(target)}</p></html>'''
            self.send(200, page.encode('utf-8'))

        def do_POST(self):
            try:
                if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'):
                    raise ValueError('Origin không hợp lệ')
                length = int(self.headers.get('Content-Length', '0'))
                if length > 100000:
                    raise ValueError('Dữ liệu quá lớn')
                data = parse_qs(self.rfile.read(length).decode('utf-8'), keep_blank_values=True)
                i = int(parse_qs(urlsplit(self.path).query).get('i', ['-1'])[0])
                if not 0 <= i < len(records) or not secrets.compare_digest(data.get('token', [''])[0], token):
                    raise ValueError('Yêu cầu không hợp lệ')
                save_rating(target, records[i], data)
            except (ValueError, UnicodeError) as e:
                return self.send(400, html.escape(str(e)).encode())
            self.send_response(303)
            self.send_header('Location', f'/?i={i}')
            self.end_headers()

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Mở http://localhost:{args.port} để chấm {len(records)} ảnh. Phiếu: {target}', flush=True)
    print('Nhấn Ctrl+C để dừng.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()