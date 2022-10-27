python train_video.py --model base \
                      --lmbda 0.0067 \
                      --dataset /workspace/shared/vimeo_septuplet \
                      --cuda \
                      --batch-size 8 \
                      --learning-rate 1e-4 \
                      --epochs 400 \
                      --save \
                      --checkpoint /workspace/lm/lossyvae-video/checkpoints/base/0.0067/checkpoint_best_loss.pth.tar


