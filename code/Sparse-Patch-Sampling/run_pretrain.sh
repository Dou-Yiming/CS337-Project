# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_mae_base_patch16_224'
# path to imagenet-1k train set
DATA_PATH='/path/to/ImageNet_ILSVRC2012/train'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_ratio 0.75 \
    --model pretrain_mae_base_patch16_224 \
    --batch_size 8 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --epochs 1600 \
    --output_dir ${OUTPUT_DIR} \
    --resume weights/pretrain_mae_vit_base_mask_0.75_400e.pth
