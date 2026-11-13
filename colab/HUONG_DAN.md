# HEIM: Colab và dữ liệu ảnh

Các file trong thư mục này:
- HEIM_12_tieu_chi.ipynb: mở bằng https://colab.research.google.com/ → File → Upload notebook.
- HEIM_pilot_images.zip: chạy ô 2 trong notebook để tải ZIP này lên Colab.
- build_colab_artifacts.py + heim_evaluate.py: tạo lại ZIP nếu có ảnh mới.

Mở notebook, chọn Runtime → Change runtime type → T4 GPU nếu có, sau đó chạy từng ô.
Báo cáo và phiếu chấm người được tải về dưới dạng heim_evaluation_results.zip.
Notebook không chứa .env hay API key; ảnh, prompt và metadata được tải lên Google Colab.

Mục 11 bổ sung 5 bản dịch tiếng Tây Ban Nha, dịch thủ công cho pilot, không phải
bản dịch HEIM chính thức. Khi chưa có ảnh, mục đó ghi pending_data. Để sinh ảnh
cho mục 11 trên WSL và tạo ZIP mới:

```bash
cd /home/hoang/experiments/HELM
/home/hoang/.conda/envs/oneig/bin/python run_heim_prompts.py --criterion 11_multilinguality --continue-on-error
/home/hoang/.conda/envs/oneig/bin/python colab/build_colab_artifacts.py
```

Bộ chạy toàn bộ hiện tối đa 95 lượt: 85 nhóm HEIM (bao gồm 5 ảnh đa ngôn ngữ)
và 10 kiểm tra tỷ lệ. Lệnh `--all` sẽ bao gồm 5 ảnh mới này và bỏ qua ảnh thành công.
Notebook đánh giá CLIP, thẩm mỹ, bias (proxy giới và màu da), fairness/robustness
theo cặp, tỷ lệ và thời gian API; hỗ trợ chấm người cho những mục còn lại.
Không phải tái lập toàn bộ metric/leaderboard HEIM; thiếu nhãn, ảnh thật đối chứng
và đánh giá người sẽ được đánh dấu chờ. Không suy luận FID từ 5 ảnh.
