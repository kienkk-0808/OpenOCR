"""Chuẩn bị dữ liệu biển số: file nhãn (img<TAB>label) -> LMDB train/val.

Ví dụ:
    python tools/prepare_plate_dataset.py --data_dir ./raw_plates --label_file ./raw_plates/labels.txt \
        --output_dir ./plate_dataset --val_ratio 0.1
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.create_lmdb_dataset import createDataset, get_datalist  # noqa: E402


def load_charset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip('\r\n') for line in f if line.strip('\r\n')}


def normalize(data_list, charset):
    """Chuẩn hoá nhãn về chữ in hoa, bỏ khoảng trắng; loại mẫu chứa ký tự ngoài từ điển."""
    out, dropped = [], 0
    for path, label in data_list:
        label = label.strip().replace(' ', '').upper()
        if label and all(c in charset for c in label):
            out.append([path, label])
        else:
            dropped += 1
    return out, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='thư mục gốc chứa ảnh')
    parser.add_argument('--label_file', required=True, nargs='+',
                        help='file nhãn, mỗi dòng: đường_dẫn_ảnh<TAB>nhãn (tương đối với data_dir)')
    parser.add_argument('--output_dir', default='./plate_dataset')
    parser.add_argument('--dict_path', default='./tools/utils/plate_dict.txt')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = get_datalist(args.data_dir, args.label_file, args.max_len)
    data, dropped = normalize(data, load_charset(args.dict_path))
    print(f'Hợp lệ: {len(data)} mẫu, loại bỏ: {dropped} mẫu (ký tự ngoài từ điển/nhãn rỗng)')
    if not data:
        raise SystemExit('Không có mẫu hợp lệ.')

    random.Random(args.seed).shuffle(data)
    n_val = int(len(data) * args.val_ratio) if args.val_ratio > 0 else 0
    splits = {'val': data[:n_val], 'train': data[n_val:]}
    for name, items in splits.items():
        if items:
            createDataset(items, os.path.join(args.output_dir, name))


if __name__ == '__main__':
    main()
