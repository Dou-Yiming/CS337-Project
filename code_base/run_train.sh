CUDA_VISIBLE_DEVICES=1 python train.py --outputs_dir "exp/" \
    --config_path "configs/default.yml" \
    --scale 10 \
    --lr 1e-4 \
    --batch_size 1 \
    --num_epochs 1000 \
    --num_workers 0 \
    --down_sample delaunay \
    --model DRRN \
    --ckpt exp/DRRN/delaunay/x_10/best.pth
