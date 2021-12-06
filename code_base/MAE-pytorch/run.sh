# Set the path to save images
OUTPUT_DIR='output/FFT'
# path to image for visualization
IMAGE_PATH='files/butterfly.png'
# IMAGE_PATH='./files/test.png'
# path to pretrain model
MODEL_PATH='weights/pretrain_mae_vit_base_mask_0.75_400e.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH} \
    --mask_ratio 0.5 --mask_strategy FFT
