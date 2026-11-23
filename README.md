# Crop / Weed Segmentation trên PhenoBench

Phân vùng Background, Crop và Weed bằng DeepLabV3+ ResNet50 và SegFormer-B0.
Prediction được dùng để tạo spray mask và đánh giá mô phỏng phun thuốc theo pixel.

## Cài đặt

Môi trường local dùng Python 3.11. Từ thư mục gốc của repo, tạo và kích hoạt `.venv`,
sau đó cài thư viện:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch và torchvision cần bản CUDA phù hợp với máy. Trên Windows,
`scripts/setup_local.ps1` tạo môi trường Python 3.11 và cài PyTorch CUDA 12.6.
Kaggle/Colab dùng `requirements-cloud.txt` để giữ bộ PyTorch CUDA có sẵn;
mỗi lần train lưu lại phiên bản thư viện thực tế.

## Dữ liệu

```text
PhenoBench/
  train/images/       # 1.407 ảnh
  train/semantics/
  val/images/         # 772 ảnh
  val/semantics/
  test/images/        # 693 ảnh, không có semantic GT công khai
```

Mapping mặc định `merge` chuyển raw IDs `0,1,2,3,4` thành `0,1,2,1,2`.
Ba lớp sau mapping là Background (0), Crop (1), Weed (2).
Train dùng random crop 512, flip, xoay 90 độ và augmentation màu.
Ảnh RGB dùng ImageNet normalization; semantic mask giữ ID lớp nguyên.
Validation dùng ảnh nguyên kích thước 1024 × 1024.

Đặt dataset theo `data.root` trong YAML hoặc truyền `--data-root` khi chạy script.
Class weights cho WCE chỉ tính từ mask của tập train:

```bash
python scripts/compute_class_stats.py --data_root ../PhenoBench-v110/PhenoBench --split train --mode merge --out data/processed/class_stats.json
```

## Thí nghiệm

| Thí nghiệm | Kiến trúc | Loss | Config | Epochs | Learning rate |
|---|---|---|---|---:|---:|
| E1 | DeepLabV3+ ResNet50 | CE | `configs/experiments/e1_deeplab_ce.yaml` | 20 | 1e-4 |
| E2 | SegFormer-B0 | CE | `configs/experiments/e2_segformer_ce.yaml` | 20 | 6e-5 |
| E3 nếu chọn DeepLab | DeepLabV3+ ResNet50 | WCE | `configs/experiments/e3_deeplab_wce.yaml` | 20 | 1e-4 |
| E3 nếu chọn SegFormer | SegFormer-B0 | WCE | `configs/experiments/e3_segformer_wce.yaml` | 20 | 6e-5 |

`plan20` dùng seed 42, batch size 4, crop 512, AdamW, weight decay 0.01 và AMP,
không có scheduler. Trainer cập nhật trọng số sau mỗi batch; loss của epoch là
trung bình loss các batch, kể cả batch cuối nhỏ hơn. Validation dùng cùng batch size
với train. Seed kiểm soát augmentation và shuffle theo epoch; kết quả vẫn có thể
khác khi thay môi trường hoặc phần cứng.

Đánh giá E1/E2 trên validation, chọn theo mIoU; hòa điểm dùng Weed IoU.
Chỉ chạy một E3 trên kiến trúc được chọn, huấn luyện mới từ pretrained với WCE.
Trọng số WCE dùng nghịch đảo căn bậc hai tần suất lớp, chuẩn hóa trung bình bằng 1.
Sau E3, so sánh cả ba thí nghiệm theo mIoU và các chỉ số phun.
Kết luận chỉ áp dụng cho những cấu hình đã chạy; WCE không bảo đảm cải thiện mIoU.

Các YAML trong `configs/` giữ ngân sách gốc: CE 3 epochs, DeepLab WCE 5 epochs,
SegFormer WCE 10 epochs. Notebook chọn chúng qua `PROFILE='original'`;
mặc định `PROFILE='plan20'` dùng bảng trên. So sánh riêng ảnh hưởng loss cần
cùng ngân sách huấn luyện. Ngân sách 20 epochs không tự chứng minh model đã hội tụ.

## Huấn luyện trên Kaggle / Colab

Đóng gói từ thư mục gốc của repo:

```bash
python scripts/package_cloud.py --overwrite
python scripts/package_cloud.py --include-data --data-root ../PhenoBench-v110/PhenoBench --overwrite
```

Lệnh đầu tạo `cloud_upload/DO_AN_AI_cloud.zip`; lệnh sau tạo thêm
`PhenoBench_trainval.zip`. Source ZIP chứa code hiện tại, cấu hình, notebook và test.
`source_manifest.json` lưu commit, trạng thái Git và SHA-256 của từng file.

Mở [train_cloud.ipynb](notebooks/train_cloud.ipynb), bật GPU và Internet.
Kaggle cần code và dataset trong Input. Colab đọc hai ZIP từ
`MyDrive/DO_AN_AI_cloud`. Chạy các cell từ trên xuống với cấu hình:

```python
MODEL = 'deeplab'
LOSS = 'ce'
PROFILE = 'plan20'
RUN_TAG = 'plan20_original_loop_v1'
RESUME = False
RESTORE_FROM = None
RUN_SMOKE = True
```

`RUN_TAG` định danh lần chạy và phải giữ nguyên khi resume hoặc tìm checkpoint
để đánh giá. Notebook chỉ thay đường dẫn data/output trong YAML.
Smoke test chạy 2 train batches và 1 validation batch, lưu riêng với hậu tố `_smoke`.
Checkpoint smoke được loại khỏi kết quả cuối. Train chính dùng toàn bộ train/validation.

Chạy trực tiếp bằng script:

```bash
python scripts/train.py --config configs/experiments/e1_deeplab_ce.yaml --model_name deeplab --data-root ../PhenoBench-v110/PhenoBench --device cuda
```

## Checkpoint và resume

Mỗi thư mục thí nghiệm chứa `best.pt`, `last.pt`, `config.yaml`,
`training_log.csv`, `environment.json` và `requirements-lock.txt`.
`best.pt` có validation loss thấp nhất; `last.pt` lưu epoch hoàn tất gần nhất.
Checkpoint ghi trọng số, optimizer, AMP scaler, epoch, `best_val_loss` và metadata.

Trong notebook, đặt `RESUME=True`, giữ model/loss/profile/tag và cấu hình cũ.
Nếu chuyển sang phiên mới, thêm thư mục kết quả vào Input; `RESTORE_FROM` trỏ
tới thư mục chứa trực tiếp checkpoint/config/log. Trainer kiểm tra tính tương thích
trước khi tiếp tục. Checkpoint thiếu `best_val_loss` được đối chiếu với `best.pt`
hoặc log bên cạnh; thiếu cả hai thì không thể khôi phục đầy đủ lịch sử best.

`epochs` là tổng ngân sách: resume từ cuối epoch 7 với ngân sách 20 sẽ chạy
epoch 8 đến 20. Epoch chưa hoàn tất phải chạy lại. Đổi CE sang WCE là thí nghiệm mới.

Tải ZIP từ bảng Output của Kaggle hoặc lưu phiên kèm output trước khi đóng.
`Save & Run All` chạy lại notebook trong phiên mới. Colab lưu checkpoint vào Drive.
Giữ cả cấu hình và log khi bàn giao checkpoint cho evaluation.

## Evaluation và dự đoán

Dùng [evaluate_cloud.ipynb](notebooks/evaluate_cloud.ipynb) với code, dataset và
thư mục checkpoint trong Input. Chọn đúng model/loss/tag hoặc điền trực tiếp
`CHECKPOINT_PATH`. `LIMIT=None` đánh giá đủ 772 ảnh validation;
`LIMIT=2` chỉ kiểm tra pipeline và xuất kết quả `_preview`.

```bash
python scripts/evaluate.py --checkpoint outputs/plan20/deeplab/ce/best.pt --data-root ../PhenoBench-v110/PhenoBench --output-dir results/E1 --device cuda
```

Output phải mới hoặc rỗng. Kết quả gồm `metrics.json`, confusion matrix (NPY/CSV/PNG),
`per_image_metrics.csv`, predictions, spray masks, error maps, hình minh họa và
`selected_cases.csv`. `plan20` dùng full-image inference; giữ cùng protocol khi so sánh.
`best.pt` được chọn theo validation loss thấp nhất, còn mIoU dùng để so sánh thí nghiệm.

Metric tổng hợp tính từ số pixel cộng trên toàn tập, với hàng là ground truth và
cột là prediction:

- IoU của lớp = TP / (TP + FP + FN); mIoU là trung bình IoU của ba lớp.
- Weed Recall = Weed dự đoán đúng / toàn bộ GT Weed.
- Spray mask là vùng dự đoán Weed; Weed Coverage bằng Weed Recall.
- Crop Wrongly Sprayed Rate = Crop dự đoán thành Weed / toàn bộ GT Crop.
- Spray Area Ratio = pixel phun / toàn bộ pixel GT hợp lệ.
- Estimated Pixel-Area Reduction = `1 - Spray Area Ratio`, chỉ đo diện tích pixel.

Pixel GT có ID ignore được loại khỏi phép tính. JSON lưu tỉ lệ từ 0 đến 1;
metric có mẫu số bằng 0 là `null`, mIoU yêu cầu đủ IoU của ba lớp.
Lượng thuốc thực tế cần được đo riêng.

Tập test công khai thiếu GT nên dùng để xem dự đoán local; không tính IoU hoặc
Weed Coverage khi chưa có nhãn tham chiếu. Xuất prediction và spray mask bằng:

```bash
python scripts/predict.py --checkpoint outputs/plan20/deeplab/ce/best.pt --input-dir ../PhenoBench-v110/PhenoBench/test/images --output-dir results/test_predictions --device cuda
```

## Phân tích kết quả

[result_analysis.ipynb](notebooks/result_analysis.ipynb) chạy trên CPU.
`RESULT_DIRS` trỏ tới thư mục chứa trực tiếp `metrics.json` của từng thí nghiệm.
Ví dụ ZIP giải nén thành `evaluation_E1/E1/metrics.json` thì chọn `evaluation_E1/E1`.
Trên Kaggle, đặt output phân tích trong `/kaggle/working`.

Notebook kiểm tra split, danh sách ảnh, mapping, normalization, inference và các
thông số train trước khi tạo bảng chung. Preview/smoke hoặc split thiếu ảnh bị loại.
Chọn kiến trúc cho E3 cần log xác nhận E1/E2 đã hoàn tất cùng ngân sách.
`TRAIN_LOGS` cho phép chỉ định lại vị trí log để vẽ đường cong loss.

`comparison.csv` và biểu đồ tổng hợp được lưu trong output phân tích.
Đối chiếu ảnh trong `selected_cases.csv` rồi điền `manual_category`, `manual_notes`
vào bản CSV có quyền ghi. Các ảnh này được chọn theo mức lỗi, không phải mẫu ngẫu nhiên.
Nhận xét CE/WCE cần đọc cùng Weed Coverage và Crop Wrongly Sprayed Rate.

## Kiểm tra trước khi commit

```bash
python -m pytest -q
git diff --check
```

Xóa output notebook bằng Clear All Outputs trước khi commit. Dataset, checkpoint,
ZIP và kết quả sinh tự động được bỏ qua trong `.gitignore`; bàn giao các file này
riêng với cấu hình tương ứng. Sau khi cập nhật code, tạo lại source ZIP cho cloud.
