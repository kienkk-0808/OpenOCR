# Train SVTRv2-B nhận dạng biển số

Công cụ fine-tune mô hình **SVTRv2-B** (CTC) để đọc biển số xe, xây dựng trên [OpenOCR](https://github.com/Topdu/OpenOCR) (Fudan University, giấy phép xem [LICENSE](./LICENSE)).

## Cấu trúc

| Đường dẫn | Mô tả |
| --- | --- |
| `configs/rec/svtrv2/svtrv2_b_plate.yml` | Cấu hình train/eval SVTRv2-B cho biển số |
| `tools/utils/plate_dict.txt` | Bộ ký tự: `0-9 A-Z - .` |
| `tools/prepare_plate_dataset.py` | Tạo LMDB train/val từ file nhãn |
| `tools/train_rec.py`, `eval_rec.py`, `infer_rec.py`, `toonnx.py` | Train, đánh giá, suy luận, xuất ONNX |
| `train_script/svtrv2_b_plate.sh` | Các lệnh mẫu |
| `pretrained/svtrv2_b/best.pth` | Checkpoint SVTRv2-B pretrained (Union14M-L-Filter) |
| `docs/license_plate.md` | Hướng dẫn chi tiết |

## Cài đặt

```shell
pip install -r requirements.txt   # cài thêm PyTorch + torchvision bản GPU
```

## Cấu trúc dataset

Ảnh là crop biển số, nhãn chữ in hoa (`0-9 A-Z - .`), mỗi dòng của file nhãn là `đường_dẫn_ảnh<TAB>nhãn`:

```
raw_plates/                 # --data_dir
├── labels.txt              # 59A-123.45 ... (đường dẫn tương đối với raw_plates/)
└── imgs/
    ├── 0001.jpg
    └── 0002.jpg
```

```
# labels.txt
imgs/0001.jpg	59A-123.45
imgs/0002.jpg	30F-999.99
```

Chuyển sang LMDB (tự chia train/val, chuẩn hoá nhãn, loại mẫu ngoài bộ ký tự):

```shell
python tools/prepare_plate_dataset.py --data_dir ./raw_plates --label_file ./raw_plates/labels.txt --output_dir ./plate_dataset --val_ratio 0.1
```

```
plate_dataset/              # config đọc trực tiếp thư mục này
├── train/                  # LMDB train
│   ├── data.mdb
│   └── lock.mdb
└── val/                    # LMDB val
    ├── data.mdb
    └── lock.mdb
```

## Train

```shell
# 1 GPU
CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml

# Nhiều GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml
```

Kết quả lưu tại `output/rec/plate/svtrv2_b_plate/` (`best.pth`, `latest.pth`). Các tuỳ chọn khác (train từ đầu, resume, đổi lr/batch size): xem [docs/license_plate.md](./docs/license_plate.md).

## Đánh giá / suy luận / ONNX

```shell
python tools/eval_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
python tools/infer_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.infer_img=./imgs Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
python tools/toonnx.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.device=cpu Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
```

## Trích dẫn

```bibtex
@inproceedings{svtrv2,
  title     = {SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Xie, Hongtao and Jia, Caiyan and Jiang, Yu-Gang},
  booktitle = {ICCV},
  year      = {2025}
}
```
