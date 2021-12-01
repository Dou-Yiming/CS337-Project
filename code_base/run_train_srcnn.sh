# python train_srcnn.py --train_file "./data/srcnn/train/masked_91-image_x3.h5" \
python train_srcnn.py --train_file "./data/srcnn/train/91-image_x3.h5" \
                --eval_file "./data/srcnn/val/Set5_x3.h5" \
                --outputs_dir "exp/scrnn" \
                --config_path "configs/srcnn.yml" \
                --scale 3 \
                --lr 1e-3 \
                --batch_size 1024 \
                --num_epochs 400 \
                --num_workers 0 \