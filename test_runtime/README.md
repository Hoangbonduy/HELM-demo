# Đo runtime HEIM trên 2.000 ảnh

Mở `HEIM_runtime_2000.ipynb` bằng Google Colab, chọn GPU, chạy từng ô.
Notebook độc lập: đã nhúng script tải dữ liệu và benchmark, không cần upload thêm script hoặc repo.
Các file hiện chỉ được chuẩn bị; chưa tải bộ ảnh, chưa chạy inference.

## Các file

- `HEIM_runtime_2000.ipynb`: notebook để tự chạy trên Colab.
- `download_data.py`: chọn cố định 2.000 ảnh COCO val2017 có caption (seed 2021), tải song song, retry và dùng lại ảnh hợp lệ đã có; kiểm tra checksum và ảnh trùng nội dung.
- `benchmark.py`: tái hiện scorer CLIP ViT-L/14 + LAION linear aesthetic head của `colab/heim_evaluate.py`, batch size 1; đo 3 lượt sau warm-up 20 ảnh.
- `build_notebook.py`: tạo lại notebook sau khi sửa script (`python build_notebook.py`). Không tải dữ liệu hoặc chạy mô hình.

## Phạm vi phép đo

| Thời gian | Bao gồm |
|---|---|
| Inference | CLIP image encoder, text encoder cho caption chưa cache, chuẩn hóa, CLIP similarity và aesthetic head |
| Toàn pipeline | Đọc/giải mã ảnh, tiền xử lý, token hóa, truyền dữ liệu, inference, lấy scalar điểm số, vòng lặp và phần đo thời gian |

Inference dùng CUDA events trên GPU hoặc perf_counter trên CPU. CUDA event là thời gian giữa hai event trên luồng GPU, có thể gồm khoảng chờ giữa kernel. Toàn pipeline dùng đồng hồ thực và đồng bộ GPU từng ảnh. Đây là chế độ tuần tự batch 1 như scorer hiện tại, không phải phép đo throughput batching tối đa.

Cache caption bắt đầu rỗng mỗi lượt; các caption giống nhau trong cùng lượt dùng lại text embedding. Trọng số và cách tính theo pilot: OpenAI CLIP commit `a1d071733d7111c9c014f024669f959182114e33`, `ViT-L/14`, LAION `sa_0_4_vit_l_14_linear.pth`. Dùng no_grad, CLIP dtype theo device mặc định; dtype thực tế được ghi trong báo cáo.

Đọc kiểm tra toàn bộ ảnh và warm-up xảy ra trước đo: cache file của hệ điều hành đã được làm nóng. Không tính cài thư viện, tải dữ liệu/trọng số, nạp model, kiểm tra dữ liệu, warm-up hoặc ghi báo cáo. Caption là mô tả ảnh thật từ COCO; kết quả phục vụ đo tốc độ chấm, không đánh giá chất lượng mô hình sinh ảnh.

## Kết quả

`report.json` ghi cấu hình, GPU, PyTorch/CUDA, dtype, hash manifest, từng lượt và trung vị/min/max thời gian pipeline.
`runs.csv` ghi tổng giây, ms/ảnh và ảnh/giây cho cả hai phép đo.
`per_image_run_*.csv` ghi điểm và thời gian từng ảnh; gặp ảnh lỗi thì dừng, không âm thầm báo kết quả trên ít hơn 2.000 ảnh.
Ô cuối đóng gói kết quả, manifest và pip freeze thành ZIP để tải về.

Thời gian inference là tổng các khoảng inference từng ảnh; thời gian pipeline là đồng hồ toàn lượt. Ảnh/giây được tính riêng từ mỗi khoảng tương ứng. Pipeline là số chính để ước lượng thời gian thực tế chấm ảnh đã lưu trên Colab.

## Nếu chạy bằng dòng lệnh

Trong môi trường có torch/torchvision, Pillow, OpenAI CLIP và các phụ thuộc tương ứng:

```bash
python download_data.py --output data --count 2000 --seed 2021
python benchmark.py --manifest data/manifest.json --expected-count 2000 --repeats 3 --warmup 20 --output results
```

Nguồn dữ liệu: https://cocodataset.org/#download
Trọng số aesthetic: https://github.com/LAION-AI/aesthetic-predictor

Downloader đã được sửa dùng HTTPS S3 `https://s3.amazonaws.com/images.cocodataset.org/...` để tránh lỗi chứng chỉ hostname ở host cũ. Kiểm tra TLS vẫn bật. Notebook cũng hiển thị stderr chung với stdout khi chạy tải ảnh và benchmark.


## Bản notebook lưu Google Drive
Notebook hiện tại mount Drive và lưu ảnh gốc trong `MyDrive/test_runtime/data`, ảnh 1024 × 1024 trong `MyDrive/test_runtime/data_1024`. Trước khi đo, notebook chép ảnh đã resize sang `/content/heim_runtime/data_1024`. Mở notebook hiện tại trực tiếp; `build_notebook.py` là mẫu cũ và sẽ ghi đè notebook nếu chạy.
