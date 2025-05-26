source .venv/bin/activate
python3 main_vgae.py --train_path "./datasets/A/train.json.gz ./datasets/D/train.json.gz ./datasets/C/train.json.gz ./datasets/B/train.json.gz" --pretraining=True --model_id 1
python3 main_vgae.py --train_path ./datasets/A/train.json.gz --test_path ./datasets/A/test.json.gz --pretrained_path "./checkpoints/model_pretraining_1_best.pth" --model_id 1
python3 main_vgae.py --train_path ./datasets/B/train.json.gz --test_path ./datasets/B/test.json.gz --pretrained_path "./checkpoints/model_pretraining_1_best.pth" --model_id 1
python3 main_vgae.py --train_path ./datasets/C/train.json.gz --test_path ./datasets/C/test.json.gz --pretrained_path "./checkpoints/model_pretraining_1_best.pth" --model_id 1
python3 main_vgae.py --train_path ./datasets/D/train.json.gz --test_path ./datasets/D/test.json.gz --pretrained_path "./checkpoints/model_pretraining_1_best.pth" --model_id 1
