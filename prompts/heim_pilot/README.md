# HEIM pilot và kiểm tra tỷ lệ ảnh

Mỗi nhóm HEIM dùng 5 prompt gốc. Prompt/ID/biến thể giữ nguyên từ tập đã lưu;
nhóm suy luận lấy 2 Counting, 2 Positional và 1 Conflicting. Thiên kiến dùng
4 ảnh/prompt. Đây là pilot giảm mẫu, không phải tái lập toàn bộ leaderboard HEIM.

| Tiêu chí | Dữ liệu chạy | Lượt sinh ảnh |
|---|---|---:|
| 01_alignment | 5 prompt | 5 |
| 02_photorealism | 5 prompt | 5 |
| 03_aesthetics | 5 prompt | 5 |
| 04_originality | 5 prompt | 5 |
| 05_reasoning | 5 prompt | 5 |
| 06_knowledge | 5 prompt | 5 |
| 07_bias | 5 prompt × 4 ảnh | 20 |
| 08_toxicity | 5 prompt | Tối đa 5 |
| 09_fairness | 5 gốc + 5 giới tính + 5 phương ngữ | 15 |
| 10_robustness | 5 gốc + 5 MildMix | 10 |
| 11_multilinguality | 5 prompt tiếng Tây Ban Nha, dịch thủ công cho pilot | 5 |
| 12_efficiency | Dùng lại thời gian sinh ảnh nhóm 01 | 0 |
| 13_aspect_ratio | 2 nội dung × 5 tỷ lệ × 1 ảnh | 10 |
| Tổng | | Tối đa 95/model |

5 tỷ lệ: 1:1, 16:9, 9:16, 4:3, 3:4. File `13_aspect_ratio.jsonl` chứa 10
phiên bản từ 2 caption gốc của nhóm alignment. Tỷ lệ ghi trong cả prompt và
kích thước ảnh yêu cầu gửi đến Vita Design.

## Lệnh chạy

```bash
cd /home/hoang/experiments/HELM-demo
# Thử 1 prompt alignment, 1 ảnh:
python3 run_heim_prompts.py --criterion 01_alignment --limit 1 --images-per-prompt 1
# Chỉ chạy kiểm tra tỷ lệ (10 ảnh):
python3 run_heim_prompts.py --criterion 13_aspect_ratio --continue-on-error
# Chạy tất cả (tối đa 95 lượt sinh):
python3 run_heim_prompts.py --all --continue-on-error
# Chỉ xem kế hoạch, không gọi API:
python3 run_heim_prompts.py --all --dry-run
```

Runner cần Pillow và httpx, dùng cấu hình Vita Ask UAT trong `.env`. Biến shell có ưu tiên.
PNG và metadata nằm trong `outputs/heim_pilot/v-llm-v1-xlarge/<criterion>/`.
Kết quả tỷ lệ tự lưu thành `aspect_ratio_report.json` trong thư mục `<model>`.
Chạy lại bỏ qua ảnh thành công có metadata trùng prompt/model/size.
Bản prompt 10 mẫu ban đầu được lưu ở `archive_10_prompts/`, không được --all chọn.

## Phép đo tỷ lệ ảnh

Ảnh yêu cầu a:b đạt khi W*b == H*a, với W,H là kích thước thực trước resize/crop.
Không dùng sai số xấp xỉ. Accuracy = N_correct/N_valid * 100%, mục tiêu 100%.
Report gồm overall và by_ratio, kích thước từng ảnh và trạng thái. Ảnh không
sinh được hoặc không đọc được được đếm riêng, loại khỏi N_valid; lượt chưa
chạy được đếm not_run. Khi N_valid=0, accuracy là null, không phải 0%.
Đây là tiêu chí mở rộng theo yêu cầu, không phải metric gốc của HEIM.

## Giới hạn các nhóm HEIM

Chỉ tiêu tỷ lệ được tự động chấm trong runner. Các metric HEIM khác vẫn cần
chấm riêng theo công thức gốc; chấm người cho chân thực/thẩm mỹ/nguyên bản
không bị thay bằng AI judge. Không kết luận FID từ 5 ảnh. Hiệu quả ghi thời gian
API, gồm cả mạng/provider; không phải denoised local runtime. Vita Design hiện chưa nhận seed ảnh.

Fairness dùng gender terms male→female prob=1 và SAE→AAVE prob=1; robustness
dùng MildMix nguyên implementation HELM 0.5.16. Seed biến đổi là ID caption
đã lưu. Lựa chọn caption có chủ đích để cả hai phép biến đổi fairness có tác dụng.
I2P giữ bộ lọc hard=1, prompt_toxicity<0.5; pilot loại category sexual nên không
bao phủ toàn bộ loại nội dung độc hại. Nguồn và hash nằm trong manifest.json.

Đa ngôn ngữ: 5 bản dịch Tây Ban Nha là bản dịch thủ công cho pilot, không phải bản dịch chính thức HEIM. Sinh ảnh còn thiếu bằng `python run_heim_prompts.py --criterion 11_multilinguality --continue-on-error`. Khi sinh thêm ảnh, tạo lại ZIP trước khi tải lên Colab.
