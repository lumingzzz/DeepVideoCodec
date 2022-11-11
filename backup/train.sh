python train.py -m dmc_context \
                -d /workspace/shared/vimeo_septuplet \
                -e 100 \
                -lr 1e-4 \
                -n 8 \
                -q 6 \
                --lambda 2048 \
                --batch-size 8 \
                --max-frames 2 \
                --cuda \
                --save \
              #   --checkpoint /workspace/lm/NVC/checkpoints/dmc_context_multi/3/checkpoint_best_loss.pth.tar
