python train_video.py --model base \
                      --lmbda 0.0067 \
                      --dataset /workspace/shared/vimeo_septuplet \
                      --cuda \
                      --batch-size 8 \
                      --learning-rate 1e-4 \
                      --epochs 400 \
                      --save \
                    #   --checkpoint /workspace/lm/NeuralVideoCoding/checkpoint/base/1024/checkpoint.pth.tar


