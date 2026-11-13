# HELM / HEIM pilot

Sinh ảnh qua 9Router (API tương thích OpenAI), đánh giá ảnh bằng HEIM pilot (9 tiêu chí + tỷ lệ ảnh). Chạy các lệnh dưới trong Ubuntu/WSL tại `/home/hoang/experiments/HELM`.

## 1. Cài thư viện HELM / HEIM bằng Anaconda

Cài theo tài liệu chính thức: [Installation](https://github.com/stanford-crfm/helm/blob/main/docs/installation.md) và [HEIM](https://github.com/stanford-crfm/helm/blob/main/docs/heim.md). HELM yêu cầu Python >= 3.10. Nếu chưa có Conda, tải [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

```bash
cd /home/hoang/experiments/HELM
source ~/miniconda3/etc/profile.d/conda.sh  # Anaconda: ~/anaconda3/...
conda create -n crfm-helm python=3.10 pip
conda activate crfm-helm
pip install crfm-helm
pip install "crfm-helm[heim]"
```

Chỉ khi cần model/metric có dependency ngoài PyPI (DALLE-mini/mega, CogView2, `DetectionMetric`), tải và chạy script extras chính thức trong môi trường phù hợp GPU/CUDA:

```bash
curl -fL https://raw.githubusercontent.com/stanford-crfm/helm/main/install-heim-extras.sh -o /tmp/install-heim-extras.sh
bash /tmp/install-heim-extras.sh
```

Các runner API/pilot riêng của thư mục này còn dùng `python-dotenv`, Pillow, OpenAI CLIP và OpenCV. Generation cần `../OneIG-Benchmark/inference.py`. Nếu chạy các script pilot ở mục 3–4, cài thêm trong cùng môi trường:

```bash
python -m pip install python-dotenv Pillow opencv-python-headless 'setuptools<81' wheel
python -m pip install --no-build-isolation git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
```

Cài package HELM/HEIM không tự biến các runner pilot thành bộ chạy leaderboard chính thức. Lần đầu chấm pilot sẽ tải CLIP ViT-L/14 và trọng số thẩm mỹ LAION; CPU chạy được, Colab GPU nhanh hơn.

## 2. Setup API: 9Router hoặc OpenAI key

```bash
cp -n .env.example .env
chmod 600 .env
nano .env
```

**A. Qua 9Router** — cấu hình `.env`:

```dotenv
MUSE2API_URL=http://localhost:20128/v1
MUSE2API_API_KEY=<KEY_9ROUTER>
MUSE2API_MODEL=<MODEL_ANH_DA_CAU_HINH_TRONG_9ROUTER>
MUSE2API_SIZE=1024x1024
MUSE2API_TIMEOUT=600
```

Nếu dùng OpenAI làm upstream: tạo key ở [OpenAI API Keys](https://platform.openai.com/api-keys), bảo đảm project có billing/quyền dùng model ảnh, rồi thêm key đó vào provider OpenAI trong 9Router và bật connection. Chọn đúng ID model/route ảnh do 9Router cung cấp. Key OpenAI đặt ở provider upstream; `MUSE2API_API_KEY` vẫn là key **9Router**. `muse-image` là tên model cũ của dự án, chỉ dùng nếu đã có route tương ứng trong gateway.

**B. OpenAI trực tiếp** — bỏ qua 9Router, cấu hình `.env`:

```dotenv
MUSE2API_URL=https://api.openai.com/v1
MUSE2API_API_KEY=<OPENAI_API_KEY>
MUSE2API_MODEL=gpt-image-1.5
MUSE2API_SIZE=1024x1024
MUSE2API_TIMEOUT=600
```

Tên biến vẫn là `MUSE2API_*` vì client hiện tại đọc các tên này; chỉ đặt `OPENAI_API_KEY` sẽ chưa được runner đọc. Đây là cấu hình endpoint/key, **chưa bảo đảm runner gốc chạy trực tiếp được**: client `../OneIG-Benchmark/inference.py` luôn gửi `response_format=b64_json`, trong khi GPT Image không hỗ trợ tham số đó và tự trả base64. Muốn chạy trực tiếp cần client/adapter tương thích; code gốc được giữ nguyên theo yêu cầu. Phương án A cần gateway hỗ trợ payload mà runner gửi. Tham khảo [OpenAI Images API](https://developers.openai.com/api/reference/resources/images/methods/generate).

Chọn một cấu hình mỗi lần. `.env` được Git bỏ qua; không đưa key vào README/notebook. API local phải truy cập được từ WSL. Biến shell đã export có ưu tiên trước `.env`; đổi cấu hình thì kiểm tra các biến đó. Không cần gọi API khi setup key.

## Dataset và dữ liệu ảnh

- Prompt, manifest và nguồn dataset pilot: [`prompts/heim_pilot/`](prompts/heim_pilot/); chi tiết nguồn và số mẫu nằm trong `prompts/heim_pilot/README.md` và `manifest.json`.
- Ảnh đã sinh và metadata: [`outputs/heim_pilot/muse-image/`](outputs/heim_pilot/muse-image/).
- Các file ZIP dữ liệu không được đưa lên GitHub. Để tạo `colab/HEIM_pilot_images.zip` tại máy từ ảnh hiện có, chạy `python colab/build_colab_artifacts.py`; lưu ý script cũng ghi lại notebook trong `colab/`, cần sao lưu bản final trước khi chạy.

## 3. Generation

Các lệnh này dùng runner gốc với gateway/model tương thích đã cấu hình ở mục 2A.

```bash
# Xem kế hoạch, không gọi API
python run_heim_prompts.py --all --dry-run
# Thử đúng 1 ảnh
python run_heim_prompts.py --criterion 01_alignment --limit 1 --images-per-prompt 1
# Khi cần chạy toàn bộ: tối đa 95 ảnh/model, có thể phát sinh phí
python run_heim_prompts.py --all --continue-on-error
```

Ảnh + metadata của `muse-image` nằm trong `HELM/outputs/heim_pilot/muse-image/`, chia theo từng tiêu chí. Với model khác: `HELM/outputs/heim_pilot/<model>/<criterion>/`. Chạy lại bỏ qua ảnh thành công khớp prompt/model/size. Chi tiết nhóm prompt: [prompts/heim_pilot/README.md](prompts/heim_pilot/README.md).

## 4. Evaluation

Phạm vi gồm **9 tiêu chí + tỷ lệ ảnh**: bám prompt, độ chân thực, tính thẩm mỹ, tính nguyên bản, khả năng suy luận, tri thức, nội dung độc hại, độ hiệu quả và phong cách. Phong cách là mục chấm người bổ sung theo thang 1–5 về mức phù hợp phong cách được yêu cầu; cần nhóm ảnh/prompt `14_style` để có điểm, hiện báo chờ dữ liệu. Các mục còn thiếu điểm người được giữ trạng thái thiếu, không điền 0.

Đánh giá gồm hai phần, dùng ảnh và metadata đã sinh; không gọi API sinh ảnh thêm. Đổi `muse-image` thành đúng model cần chấm.

### 4.1. Chấm người bằng tool

```bash
python manual_review.py --model muse-image --port 8765
```

Mở `http://localhost:8765`, nhập tên người chấm, xem ảnh/prompt, điền các tiêu chí hiển thị và bấm **Lưu điểm** trước khi chuyển ảnh. Thang 1–5: 1 thấp nhất, 5 cao nhất; thang 0/1: không/sai hoặc có/đúng. Để trống nếu chưa đánh giá.

Điểm lưu tại `HELM/results/human_ratings.csv`; mở lại tool để tiếp tục chấm. Nhấn `Ctrl+C` trong terminal để dừng.

### 4.2. Chạy notebook

Dùng bản final: [colab/HEIM_12_tieu_chi.ipynb](colab/HEIM_12_tieu_chi.ipynb).

1. Mở notebook trên [Google Colab](https://colab.research.google.com/), chọn **Runtime → Change runtime type → T4 GPU** nếu có.
2. Chạy các ô cài thư viện, tải `colab/HEIM_pilot_images.zip` lên (hoặc đặt `ZIP_PATH` từ Google Drive), rồi chọn `MODEL_ID` trùng model đã sinh ảnh.
3. Chạy các ô kiểm tra dữ liệu, tính metric tự động và tạo báo cáo. Điểm chấm người được tải lên ở bước tiếp theo.
4. Để gộp điểm người, chạy ô **Tải lại phiếu đã chấm**, upload `HELM/results/human_ratings.csv`, rồi chạy lại ô **Cập nhật và tải kết quả**. Không cần chạy lại CLIP.
5. Tải `heim_evaluation_results.zip`, gồm báo cáo JSON/CSV, kết quả từng ảnh và phiếu chấm người.

Nếu có ảnh mới, tạo lại ZIP dữ liệu bằng `python colab/build_colab_artifacts.py`. Lưu ý: script này cũng ghi lại `colab/HEIM_12_tieu_chi.ipynb`, nên cần sao lưu bản final trước khi chạy và khôi phục lại sau khi tạo ZIP. Colab chấm ảnh trong ZIP, không cần API key.

## 5. Kết quả

- Kết quả thô: [heim_pilot_report.json](results/heim_pilot_report.json).
- Bảng tổng hợp: [HEIM_bang_tong_hop_ket_qua.md](results/HEIM_bang_tong_hop_ket_qua.md).
