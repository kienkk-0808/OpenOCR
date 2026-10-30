# Fine-tune SVTRv2-B nhận dạng biển số

Cấu hình: [svtrv2_b_plate.yml](../configs/rec/svtrv2/svtrv2_b_plate.yml) — SVTRv2-B (dims 128/256/384, depths 6/6/6) + RCTC decoder.

- Pretrained: `pretrained/svtrv2_b/best.pth` (SVTRv2-B, Union14M-L-Filter, tải từ [Google Drive](https://drive.google.com/drive/folders/11u11ptDzQ4BF9RRsOYdZnXl6ell2h4jN)). Chỉ encoder được nạp; lớp head bị bỏ qua vì từ điển khác.
- Ký tự: `0-9 A-Z - .` ([plate_dict.txt](../tools/utils/plate_dict.txt)); sửa file này nếu cần thêm ký tự (head khởi tạo lại tự động).
- Ảnh đầu vào: crop biển số, kích thước cơ sở 144x48, giữ tỉ lệ khung hình (tối đa tỉ lệ 6).

## 1. Chuẩn bị dữ liệu

File nhãn, mỗi dòng `đường_dẫn_ảnh<TAB>nhãn` (tương đối với `--data_dir`):

```
imgs/0001.jpg	59A-123.45
imgs/0002.jpg	30F-999.99
```

```shell
python tools/prepare_plate_dataset.py --data_dir ./raw_plates --label_file ./raw_plates/labels.txt --output_dir ./plate_dataset --val_ratio 0.1
```

Nhãn được chuẩn hoá in hoa, bỏ khoảng trắng; mẫu chứa ký tự ngoài từ điển bị loại. Kết quả: `plate_dataset/train`, `plate_dataset/val` (LMDB).

## 2. Cài đặt

```shell
pip install -r requirements.txt   # cần thêm PyTorch + torchvision bản GPU
```

## 3. Train / đánh giá / suy luận

Xem [svtrv2_b_plate.sh](../train_script/svtrv2_b_plate.sh). Đổi đường dẫn dữ liệu bằng `--o Train.dataset.data_dir_list=[...] Eval.dataset.data_dir_list=[...]`.

`lr` mặc định (0.0002) ứng với 1 GPU, batch 128; khi đổi tổng batch size hãy điều chỉnh `Optimizer.lr` tỉ lệ tương ứng.
