CUDA_VISIBLE_DEVICES=0 python test.py --outputs_dir "exp/test" \
    --scale 3 \
    --model DRRN \
    --ckpt exp/FFT/DRRN/x_3/best.pth \
    --test_img ./data/test/test.png