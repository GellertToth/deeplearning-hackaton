#!/bin/bash

source .venv/bin/activate

MODEL_ID="$1"
DEVICE_ID=${2:-0}

# Pretraining
python3 main_vgae.py --train_path "./datasets/A/train.json.gz ./datasets/D/train.json.gz ./datasets/C/train.json.gz ./datasets/B/train.json.gz" \
    --pretraining=True --model_id "$MODEL_ID" \
    --n_folds 8 --train_folds_to_use 5 \
    --device $DEVICE_ID

# Finetuning / Inference
for dataset in A B C D; do
    python3 main_vgae.py \
        --train_path "./datasets/$dataset/train.json.gz" \
        --pretrained_path "./checkpoints/model_pretraining_${MODEL_ID}_best.pth" \
        --model_id "$MODEL_ID" \
        --device $DEVICE_ID
done