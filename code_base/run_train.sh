python train.py --outputs_dir "exp/UNet" \
    --config_path "configs/default.yml" \
    --scale 3 \
    --lr 1e-3 \
    --batch_size 1 \
    --num_epochs 10000 \
    --num_workers 0 \
    --down_sample delaunay --point_num 10000 \
    --model UNet
