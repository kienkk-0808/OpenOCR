# Fine-tune SVTRv2-B (pretrained ./pretrained/svtrv2_b/best.pth) cho biển số - 1 GPU
CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml

# Nhiều GPU (vd 4 GPU)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml

# Train từ đầu (không dùng pretrained)
# CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.pretrained_model=

# Tiếp tục train bị gián đoạn
# CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.checkpoints=./output/rec/plate/svtrv2_b_plate/latest.pth

# Đánh giá / suy luận / xuất ONNX
# python tools/eval_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
# python tools/infer_rec.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.infer_img=./imgs Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
# python tools/toonnx.py --c configs/rec/svtrv2/svtrv2_b_plate.yml --o Global.device=cpu Global.pretrained_model=./output/rec/plate/svtrv2_b_plate/best.pth
