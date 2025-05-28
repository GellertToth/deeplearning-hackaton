source .venv/bin/activate
for dataset in A B C D; do
    python3 main_vgae.py --train_path ./datasets/A/train.json.gz \
        --test_path ./datasets/$dataset/test.json.gz \
        --models "./checkpoints/model_${dataset}_0_best.pth ./checkpoints/model_${dataset}_1_best.pth ./checkpoints/model_${dataset}_2_best.pth ./checkpoints/model_${dataset}_3_best.pth ./checkpoints/model_${dataset}_4_best.pth ./checkpoints/model_${dataset}_5_best.pth ./checkpoints/model_${dataset}_6_best.pth ./checkpoints/model_${dataset}_7_best.pth ./checkpoints/model_${dataset}_6_best.pth ./checkpoints/model_${dataset}_8_best.pth"
done