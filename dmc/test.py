import os
import time
import json
import sys
import struct
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import OrderedDict

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from compressai.zoo import cheng2020_anchor

from models import DMC

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def evaluate_one_video(args, quality, frame_dir):

    net_intra = cheng2020_anchor(quality=6, pretrained=True)
    net_intra.update(force=True)
    net_intra.to(device).eval()

    net_inter = DMC()
    snapshot = torch.load('/workspace/lm/videoCodec/dmc/checkpoints/dmc/2048.0/checkpoint_best_loss.pth.tar')

    # new_state_dict = OrderedDict()
    # for k, v in snapshot["state_dict"].items():
    #     name = k[7:] # remove module
    #     new_state_dict[name] = v
    # net_inter.load_state_dict(new_state_dict)

    net_inter.load_state_dict(snapshot['state_dict'])
    net_inter.update(force=True)
    net_inter.to(device).eval()

    # sanity check
    results_dir = Path(args.results_dir)
    assert results_dir.is_dir(), f'results_dir={results_dir} does not exist'
    save_path = results_dir / f'q{quality}.json'

    logging.info(f'starting q={quality}, frame_dir={frame_dir}, save_path={save_path}')
    tic = time.time()

    frame_dir = Path(frame_dir)
    _str = f'{args.test_dataset}-q{quality}-gop{args.gop}-num{args.num_frames}'
    save_bit_path = Path(f'cache/{_str}/{frame_dir.stem}.bits')
    if not save_bit_path.parent.is_dir():
        save_bit_path.parent.mkdir(parents=True, exist_ok=True)

    f = save_bit_path.open("wb")

    # # compute metrics
    ori_frame_paths = list(Path(frame_dir).glob('*.png'))
    ori_frame_paths.sort()
    if args.num_frames == None:
        num_frames = len(ori_frame_paths)
    else:
        num_frames = args.num_frames
    img_height, img_width = cv2.imread(str(ori_frame_paths[0])).shape[:2]

    # compute psnr
    sum_psnr = 0.0
    for fi, ori_fp in enumerate(ori_frame_paths[:num_frames]):
        # read an original frame
        x = cv2.cvtColor(cv2.imread(str(ori_fp)), cv2.COLOR_BGR2RGB) / 255.
        assert x.shape == (img_height, img_width, 3)

        x = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(device)

        p = 64  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
        x_pad = pad(x, p)

        if fi % 32 == 0:
            with torch.no_grad():
                out_enc = net_intra.compress(x_pad)
                out = net_intra.decompress(out_enc["strings"], out_enc["shape"])
                rec_pad = out["x_hat"]
                dpb = {
                        "x_ref": rec_pad,
                        "feature_ref": None,
                        "y_ref": None,
                        "y_mv_ref": None,
                }
                rec = crop(rec_pad, (img_height, img_width))
            
            write_body(f, out_enc["shape"], out_enc["strings"])

            # dpb = {
            #         "x_ref": x_pad,
            #         "feature_ref": None,
            #         "y_ref": None,
            #         "y_mv_ref": None,
            # }
            # rec = crop(x_pad, (img_height, img_width))

        else:
            with torch.no_grad():
                out_enc = net_inter.encode_inter(x_pad, dpb)
                rec_pad, info = net_inter.decode_inter(out_enc["strings"], out_enc["shape"], dpb)
                rec_pad = rec_pad.clamp(0, 1)
            dpb = {
                    "x_ref": rec_pad,
                    "feature_ref": info["feature_ref"],
                    "y_ref": info["y_ref"],
                    "y_mv_ref": info["y_mv_ref"],
                }
            rec = crop(rec_pad, (img_height, img_width))
            # rec_ = transforms.ToPILImage()(rec.squeeze().cpu())
            # rec_.save('/workspace/lm/videoCodec/dmc/test.png', format="PNG")

            for shape, out in zip(
                out_enc["shape"].items(), out_enc["strings"].items()
            ):
                write_body(f, shape[1], out[1])

        mse = torch.mean((x - rec)**2).item()
        psnr = -10 * np.log10(mse)
        print(psnr)

        sum_psnr += psnr
    
    f.close()
    
    # compute bpp
    num_pixels = img_height * img_width * num_frames
    # avg_bpp = sum_bpp * 8.0 / float(num_pixels)
    avg_bpp = float(filesize(save_bit_path)) * 8 / num_pixels
    avg_psnr = sum_psnr / num_frames

    stats = OrderedDict()
    stats['video']   = str(frame_dir)
    stats['quality'] = quality
    stats['bpp']     = avg_bpp
    stats['psnr']    = avg_psnr

    # save results
    if save_path.is_file():
        with open(save_path, mode='r') as f:
            all_seq_results = json.load(fp=f)
        assert isinstance(all_seq_results, list)
        all_seq_results.append(stats)
    else:
        all_seq_results = [stats]
    with open(save_path, mode='w') as f:
        json.dump(all_seq_results, fp=f, indent=2)

    elapsed = time.time() - tic
    msg = f'q={quality}, time={elapsed:.1f}s. '
    msg += f'evaluate {num_frames} out of {len(ori_frame_paths)} frames. '
    msg += f'frame_dir.stem={frame_dir.stem}, bpp={avg_bpp}, psnr={avg_psnr}'
    logging.info('\u001b[92m' + msg + '\u001b[0m')
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test-dataset', type=str, default='uvg')
    parser.add_argument('-q', '--quality',      type=int, default=[6], nargs='+')
    parser.add_argument('-g', '--gop',          type=int, default=32)
    parser.add_argument('-f', '--num-frames',   type=int, default=96)
    parser.add_argument("--intra", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    frames_root = {
        'uvg': '/workspace/lm/data/UVG/PNG',
    }

    suffix = 'allframes' if (args.num_frames is None) else f'first{args.num_frames}'
    results_dir = Path(f'runs/results/{args.test_dataset}-gop{args.gop}-{suffix}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)

    # init logging
    setup_logger(str(results_dir) + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')

    logging.info(f'Saving results to {results_dir}')
    args.results_dir = results_dir

    video_frame_dirs = list(Path(frames_root[args.test_dataset]).glob('*/'))
    video_frame_dirs.sort()
    logging.info(f'Total {len(video_frame_dirs)} sequences')

    mp_results = []
    # enumerate all quality
    for q in args.quality:
        sum_psnr = 0.0
        sum_bpp = 0.0
        for i, vfd in enumerate(video_frame_dirs):
            mp_results = evaluate_one_video(args, q, vfd)
            sum_psnr += mp_results['psnr']
            sum_bpp += mp_results['bpp']
        
        avg_bpp = sum_bpp / len(video_frame_dirs)
        avg_psnr = sum_psnr / len(video_frame_dirs)
        logging.info(f'AVERAGE BPP: {avg_bpp}, AVERAGE PSNR: {avg_psnr}')


if __name__ == "__main__":
    main()