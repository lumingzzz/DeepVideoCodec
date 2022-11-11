import argparse
import logging
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from timm.utils import AverageMeter
from models.tinylic import TinyLIC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def evaluate_one_video(args, frame_dir, quality):



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-dataset', type=str, default='uvg')
    parser.add_argument('-q', '--quality',      type=int, default=[1,2,3,4,5,6,7,8], nargs='+')
    parser.add_argument('-g', '--gop',          type=int, default=32)
    parser.add_argument('-f', '--num_frames',   type=int, default=96)
    parser.add_argument("--intra", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # init logging
    logging.basicConfig(
        level=logging.INFO, format= '[%(asctime)s] %(message)s', datefmt='%Y-%b-%d %H:%M:%S'
    )

    frames_root = {
        'uvg': '/workspace/lm/data/UVG/PNG',
    }

    suffix = 'allframes' if (args.num_frames is None) else f'first{args.num_frames}'
    results_dir = Path(f'runs/results/{args.test_dataset}-gop{args.gop}-{suffix}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)
    logging.info(f'Saving results to {results_dir}')
    # args.results_dir = results_dir

    video_frame_dirs = list(Path(frames_root[args.test_dataset]).glob('*/'))
    video_frame_dirs.sort()
    logging.info(f'Total {len(video_frame_dirs)} sequences')

    net_intra = TinyLIC()

    # enumerate all quality
    for q in args.quality:
        snapshot = torch.load('checkpoint_q'+str(q)+'.pth.tar',
                        map_location=device)
        net_intra.load_state_dict(snapshot, strict=True)
        net_intra.update(force=True)
        net_intra = net_intra.to(device).eval()

        for i, vfd in enumerate(video_frame_dirs):
            print(i, vfd)
            # evaluate_one_video()





if __name__ == "__main__":
    main()