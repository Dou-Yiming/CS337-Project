CUDA_VISIBLE_DEVICES=0 python test.py --outputs_dir "exp/test" \
    --scale 10 \
    --model DRRN \
    --ckpt exp/FFT/DRRN/x_10/best.pth \
    --test_img ./data/test/test.png