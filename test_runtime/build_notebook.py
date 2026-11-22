"""Rebuild the standalone Colab notebook from adjacent scripts; no downloads."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell('markdown', '''# HEIM — benchmark inference trên 2.000 ảnh

Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU khác), rồi chạy từng ô.
Notebook tự tải ảnh có caption từ COCO val2017 và chấm bằng CLIP ViT-L/14 + LAION aesthetic head.
Chỉ cần upload notebook này; các script đã được nhúng bên trong. Không cần API key.

**Kết quả chính:** thời gian toàn pipeline và ảnh/giây. **Kết quả bổ sung:** thời gian inference.
Mặc định batch size 1, warm-up 20 ảnh, 3 lượt đo × 2.000 ảnh.
Trọng số, dữ liệu, cài đặt và warm-up được tải/chạy trước khoảng đo.
'''),
    cell('code', '''from pathlib import Path
import datetime, json, os, subprocess, sys

WORK = Path('/content/heim_runtime')
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
COUNT = 2000
SEED = 2021
REPEATS = 3
WARMUP = 20
RUN_NAME = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
RESULTS = WORK / 'results' / RUN_NAME
print('Working directory:', WORK)
print('Result directory:', RESULTS)
'''),
    cell('markdown', '''## 1. Cài phụ thuộc

Dùng PyTorch/torchvision của Colab. OpenAI CLIP được cố định theo commit của dự án.
Nếu Colab yêu cầu khởi động lại runtime sau cài đặt, thực hiện rồi chạy lại từ ô cấu hình.
'''),
    cell('code', '''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'Pillow', 'ftfy', 'regex', 'tqdm', 'setuptools<81', 'wheel'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation',
                'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'], check=True)
import torch
assert torch.cuda.is_available(), 'Hãy chọn GPU runtime trước khi benchmark.'
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name())
'''),
    cell('markdown', '## 2. Ghi script từ nội dung đã nhúng\nÔ này chỉ tạo file, chưa tải ảnh hoặc chạy mô hình.\n'),
    cell('code', '\n'.join(
        f"(WORK / {name!r}).write_text({(ROOT / name).read_text(encoding='utf-8')!r}, encoding='utf-8')"
        for name in ('download_data.py', 'benchmark.py')) + "\nprint('Scripts ready')\n"),
    cell('markdown', '''## 3. Tải 2.000 ảnh có caption

Tải annotation chính thức (~253 MB) và từng ảnh được chọn. Cần Internet và dung lượng trống.
Thứ tự chọn cố định theo seed; mỗi ảnh dùng caption có annotation ID nhỏ nhất.
Ảnh lỗi hoặc trùng nội dung được thay bằng ứng viên tiếp theo. Script có thể chạy lại để dùng ảnh đã tải.
Nguồn: https://cocodataset.org/#download
'''),
    cell('code', '''script = WORK / 'download_data.py'
assert script.is_file(), 'Hãy chạy ô 2 để tạo script trước.'
subprocess.run([sys.executable, '-u', str(script), '--output', str(WORK / 'data'),
                '--count', str(COUNT), '--seed', str(SEED), '--workers', '8'],
               stderr=subprocess.STDOUT, check=True)
manifest = json.loads((WORK / 'data/manifest.json').read_text())
print('Images:', manifest['count'])
print('First caption:', manifest['records'][0]['caption'])
'''),
    cell('markdown', '''## 4. Đo tốc độ

Mỗi lượt chạy đúng COUNT ảnh. Thời gian **inference** gồm image encoder, text encoder khi caption chưa có trong cache, chuẩn hóa đặc trưng, CLIP similarity và aesthetic head. GPU dùng CUDA events; cache caption được xóa đầu mỗi lượt và tái sử dụng trong lượt, giống cách scorer hiện tại hoạt động.

**Toàn pipeline** gồm đọc/giải mã ảnh, tiền xử lý, token hóa, truyền dữ liệu CPU↔GPU, inference, lấy điểm số, vòng lặp và phần đo thời gian. Đo tuần tự, đồng bộ GPU từng ảnh. CUDA event đo khoảng thời gian trên luồng GPU, có thể bao gồm khoảng chờ giữa các kernel; không phải tổng thời gian kernel thuần túy.

File được kiểm tra trước khi đo, nên đây là phép đo với cache hệ điều hành đã được làm nóng. Không đo tốc độ đọc ổ đĩa lạnh. Kết quả phụ thuộc GPU, CPU, kích thước ảnh gốc và số caption khác nhau.

Không tính tải dữ liệu/trọng số, khởi tạo mô hình, kiểm tra dữ liệu, warm-up hoặc ghi file báo cáo. Kết quả là tốc độ chấm CLIP + aesthetic của pilot; các tiêu chí cần người chấm không nằm trong phép đo này.
'''),
    cell('code', '''subprocess.run([sys.executable, '-u', str(WORK / 'benchmark.py'),
                '--manifest', str(WORK / 'data/manifest.json'), '--output', str(RESULTS),
                '--expected-count', str(COUNT), '--repeats', str(REPEATS),
                '--warmup', str(WARMUP)], stderr=subprocess.STDOUT, check=True)
'''),
    cell('markdown', '## 5. Xem và tải kết quả\nZIP chứa report.json, runs.csv, CSV điểm/timing từng ảnh và thông tin môi trường.\n'),
    cell('code', '''import shutil
from google.colab import files
from IPython.display import display
import pandas as pd

report = json.loads((RESULTS / 'report.json').read_text())
display(pd.read_csv(RESULTS / 'runs.csv'))
print('Trung vị qua các lượt:')
print(json.dumps(report['median'], indent=2))
print('Môi trường:')
print(json.dumps(report['environment'], indent=2))
shutil.copy2(WORK / 'data/manifest.json', RESULTS / 'manifest.json')
with (RESULTS / 'pip_freeze.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
archive = shutil.make_archive(str(WORK / ('heim_runtime_' + RUN_NAME)), 'zip', RESULTS)
files.download(archive)
'''),
]
notebook = {'cells': cells, 'metadata': {'colab': {'name': 'HEIM_runtime_2000.ipynb', 'provenance': []},
            'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for index, item in enumerate(cells):
    item['id'] = f'runtime-{index:02d}'
(ROOT / 'HEIM_runtime_2000.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Created HEIM_runtime_2000.ipynb')
