# Đánh giá HELM-demo trên Google Colab

Mỗi tiêu chí có notebook Colab riêng. Tiêu chí 1, 2 và 8 dùng chung bộ ảnh COCO val2014 của Bám prompt.

| Tiêu chí | Notebook |
| --- | --- |
| 1 · Bám prompt | [alignment](alignment/HELM_demo_evaluate.ipynb) |
| 2 · Độ chân thực | [photorealism](photorealism/HELM_demo_evaluate.ipynb) |
| 8 · Hiệu suất | [efficiency](efficiency/HELM_demo_evaluate.ipynb) |

Với các tiêu chí dưới đây, dùng lượt sinh ảnh tương ứng.

| Tiêu chí | Notebook | Phần chấm tự động | Dữ liệu bổ sung |
| --- | --- | --- | --- |
| 3 · Tính thẩm mỹ | [aesthetics](aesthetics/HELM_demo_evaluate.ipynb) | Điểm thẩm mỹ, hệ số fractal | Điểm thẩm mỹ và độ rõ chủ thể do người chấm |
| 4 · Tính nguyên bản | [originality](originality/HELM_demo_evaluate.ipynb) | Tỷ lệ watermark | Điểm nguyên bản do người chấm |
| 5 · Khả năng suy luận | [reasoning](reasoning/HELM_demo_evaluate.ipynb) | CLIP Score | Kết quả detector, điểm suy luận do người chấm |
| 6 · Tri thức | [knowledge](knowledge/HELM_demo_evaluate.ipynb) | CLIP Score | Điểm tri thức do người chấm |
| 7 · Nội dung độc hại | [toxicity](toxicity/HELM_demo_evaluate.ipynb) | Tỷ lệ ảnh nhạy cảm, ảnh đen và API từ chối | Nhãn khỏa thân |
| 11 · Hiểu prompt tiếng Việt | [vietnamese](vietnamese/HELM_demo_evaluate.ipynb) | CLIP Score đa ngôn ngữ giữa ảnh và caption tiếng Việt | |

1. Đóng gói ảnh từ thư mục gốc dự án sau khi sinh xong.

   ~~~bash
   python3 colab/package_run.py --run vita-ask-uat-alignment-coco2000-drawbench-parti
   ~~~

   Thêm `--reuse-alignment` để lấy riêng ảnh COCO cho các tiêu chí 1, 2 và 8.
   Với tiêu chí 3–7 hoặc 11, dùng tên lượt thực tế.

   Thêm `--limit 2` để thử với hai ảnh ghép. Với tiêu chí 7, dùng ZIP toàn lượt để tính tỷ lệ API từ chối vì gói thử chỉ lấy mẫu có ảnh. ZIP lưu một bản cho mỗi PNG ghép và JSON đi kèm. Với alignment, ZIP nhận cả ảnh ghép mới 1024×512 và ảnh cũ 1024×1024, được lưu tại `outputs/benchmark/`. Nếu có điểm chấm tay hoặc kết quả bộ phát hiện, thêm `--observations results/observations.jsonl`.

2. Tải ZIP từ `outputs/benchmark/` lên Google Drive. Trong Google Colab, chọn File → Upload notebook và mở notebook của tiêu chí cần chấm. Sửa `INPUT_ZIP` trong ô đầu theo vị trí thật của ZIP trên Drive, rồi chạy các ô theo thứ tự. Chọn runtime Python 3.12 cho tiêu chí 3 và 7.

3. Ô cuối hiện bảng điểm và lưu ZIP kết quả cạnh ZIP đầu vào trên Drive. Tên ZIP kết quả kết thúc bằng `-<tên-tiêu-chí>-colab-results.zip`. Báo cáo trong ZIP nằm tại `results/benchmark/<tên-lượt>/summary.json`.

- **Bám prompt** dùng CLIP Score. Điểm `human_alignment` lấy từ tệp observations nếu có.
- **Độ chân thực** dùng Inception Score và FID. FID dùng ảnh thật COCO val2014 theo ID trong bản ghi. Nếu ZIP đầu vào thiếu kho ảnh thật, bộ chấm tải từng ảnh tham chiếu cần dùng từ COCO. Cần ít nhất hai ảnh sinh và hai ảnh thật khác nhau để tính FID. Điểm `human_photorealism` lấy từ tệp observations nếu có.
- **Hiệu suất** đọc `elapsed_seconds` đã ghi khi sinh ảnh. Với một ảnh ghép, đây là tổng thời gian tạo và tải các ảnh thành phần. Cùng một lượt alignment có thể gồm cả ảnh ghép 2 ô và 4 ô, nên `raw_inference_time` là trung bình thời gian theo ảnh ghép. `denoised_inference_time` chỉ có điểm khi bản ghi có `provider_inference_seconds`.
