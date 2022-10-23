python train_video.py --model base \
                      --dataset /workspace/shared/vimeo_septuplet \
                      --cuda \
                      --batch-size 8 \
                      --learning-rate 1e-4 \
                      --epochs 400 \
                      --save \

                      #  -q 7 --lambda 0.0932 --gpu-id 3  
                      # --checkpoint /workspace/lm/TinyLIC/pretrained/tinylic_multistage_cosine_v3/7/checkpoint_best_loss.pth.tar


