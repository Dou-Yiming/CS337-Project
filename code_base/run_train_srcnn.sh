python train_srcnn.py --train_file "./data/srcnn/train/91-image_x3.h5" \
                --eval_file "./data/srcnn/val/Set5_x3.h5" \
                --outputs_dir "exp/scrnn" \
                --config_path "configs/srcnn.yml" \
                --scale 3 \
                --lr 1e-4 \
                --batch_size 256 \
                --num_epochs 400 \
                --num_workers 0 \