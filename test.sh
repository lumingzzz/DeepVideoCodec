python test.py checkpoint \
                     /workspace/lm/data/UVG/YUV/ \
                     results/UVG \
                     -a dmc_v2 \
                     --cuda \
                     -p /workspace/lm/NVC/checkpoints/dmc_context_multi/2/checkpoint_best_loss.pth.tar \
                     --keep_binaries \
                     -v \
                    #  --entropy-estimation \