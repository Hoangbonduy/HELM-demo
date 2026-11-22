# HELM-demo

- Benchmark sinh ảnh.
- Chạy các lệnh dưới đây từ thư mục gốc của dự án, nơi có `README.md`, `benchmark.py` và `scripts/`.
- Dùng Python 3.10 trở lên trên Linux hoặc WSL.

## Chuẩn bị API

- Kích hoạt môi trường Conda `crfm-helm`, cài thư viện và tạo `.env` từ tệp mẫu.

```bash
conda activate crfm-helm
python3 -m pip install httpx Pillow
test -f .env || cp .env.example .env
```

- Benchmark gửi prompt qua Vita Ask UAT và lưu ảnh Vita Design trả về.
- Trong `.env`, điền các giá trị từ cùng một phiên web UAT.

| Biến | Giá trị |
| --- | --- |
| `VITA_DESIGN_TOKEN` | Giá trị cookie `vita-suite-at` |
| `VITA_ASK_CSRF_COOKIE` | Giá trị cookie `vita-suite-csrf` |
| `VITA_ASK_CSRF_TOKEN` | Header request `X-Csrf-Token` của lần gửi `mastra` thành công |

## Tải dataset

Tải và chuẩn hóa dataset trong `HELM-demo/datasets/`, sau đó kiểm tra trạng thái dữ liệu.

```bash
python3 benchmark.py fetch
python3 benchmark.py status
```

## Các tiêu chí chấm

| Script | Tiêu chí | Chấm gì | Dataset | Số prompt |
| --- | --- | --- | --- | ---: |
| [`scripts/01_alignment.py`](scripts/01_alignment.py) | Bám prompt | Ảnh khớp với nội dung và ý nghĩa của prompt đến đâu. | MS-COCO 2014 (bản rút gọn 2.000 prompt), DrawBench (200), PartiPrompts (1.632) | 3.832 |
| [`scripts/02_photorealism.py`](scripts/02_photorealism.py) | Độ chân thực | Ảnh trông giống ảnh chụp ngoài đời đến đâu. | MS-COCO 2014 (`val2014`, bản rút gọn 2.000 prompt) | 2.000 |
| [`scripts/03_aesthetics.py`](scripts/03_aesthetics.py) | Tính thẩm mỹ | Mức độ đẹp mắt, độ phức tạp thị giác và độ rõ của chủ thể. | Landing Pages (36), Logos (100), Magazine Covers (50), Daily DALL-E (93) | 279 |
| [`scripts/04_originality.py`](scripts/04_originality.py) | Tính nguyên bản | Mức độ mới mẻ, sáng tạo và dấu hiệu watermark trong ảnh. | Landing Pages (36), Logos (100), Magazine Covers (50), Daily DALL-E (93) | 279 |
| [`scripts/05_reasoning.py`](scripts/05_reasoning.py) | Khả năng suy luận | Ảnh có đúng vật thể, số lượng và quan hệ không gian mà prompt yêu cầu hay không. | PaintSkills (1.869), Relational Understanding (75), Winoground (800), Common Syntactic Processes (52), DrawBench (46), PartiPrompts (124) | 2.966 |
| [`scripts/06_knowledge.py`](scripts/06_knowledge.py) | Tri thức | Ảnh thể hiện đúng nhân vật, địa danh hoặc thực thể cụ thể đến đâu. | Historical Figures (99), PartiPrompts (214), DrawBench (38) | 351 |
| [`scripts/07_toxicity.py`](scripts/07_toxicity.py) | Nội dung độc hại | Tỷ lệ ảnh nhạy cảm, khỏa thân, bị bôi đen và tỷ lệ API từ chối yêu cầu. | I2P (nhóm HEIM Hard) | 1.784 |
| [`scripts/08_efficiency.py`](scripts/08_efficiency.py) | Độ hiệu quả | Thời gian từ lúc gửi yêu cầu đến khi nhận ảnh, cùng thời gian suy luận nếu API cung cấp. | MS-COCO 2014 (`val2014`, bản rút gọn 2.000 prompt) | 2.000 |
| [`scripts/11_vietnamese.py`](scripts/11_vietnamese.py) | Hiểu prompt tiếng Việt | Mức độ ảnh khớp với caption tiếng Việt. | MS-COCO 2014 (bản tiếng Việt của bộ rút gọn) | 2.000 |

## Tạo ảnh

Chạy dòng tương ứng với tiêu chí cần tạo ảnh.

```bash
python3 scripts/01_alignment.py generate --workers 1
python3 scripts/02_photorealism.py generate --workers 1 --run photorealism
python3 scripts/03_aesthetics.py generate --workers 1
python3 scripts/04_originality.py generate --workers 1
python3 scripts/05_reasoning.py generate --workers 1
python3 scripts/06_knowledge.py generate --workers 1
python3 scripts/07_toxicity.py generate --workers 1
python3 scripts/08_efficiency.py generate --workers 1 --run efficiency
python3 scripts/11_vietnamese.py generate --workers 1
```

- `--workers 1` xử lý các prompt lần lượt.
- `--run` đặt tên lượt chạy và bắt buộc với script 02 và 08.
- Nếu chưa có manifest tiếng Việt, chạy `python3 scripts/11_vietnamese.py prepare` trước.

## Đánh giá

Sau khi tạo ảnh, chạy notebook của tiêu chí tương ứng trong [`colab/`](colab/) trên Google Colab theo [hướng dẫn](colab/README.md).

## Tiến độ chạy

1. `scripts/01_alignment.py` đã chạy xong MS-COCO 2014 (bản rút gọn 2.000 prompt). Còn DrawBench (200 prompt) và PartiPrompts (1.632 prompt) chưa chạy.
2. `scripts/02_photorealism.py` đã chạy xong.
3. `scripts/07_toxicity.py` đã chạy được một nửa.

Các tiêu chí còn lại chưa chạy.
