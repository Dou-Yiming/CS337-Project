CUDA_VISIBLE_DEVICES=0 python train.py --outputs_dir "exp/" \
    --config_path "configs/default.yml" \
    --lr 1e-3 \
    --batch_size 1 \
    --num_epochs 1000 \
    --num_workers 0 \
    --down_sample delaunay \
    --scale 3 \
    --model SRCNN \
    --sample_method FFT