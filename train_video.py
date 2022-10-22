import argparse
import logging
import os
import random
import time
import sys
import shutil
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from benchmark.datasets import VideoFolder
from benchmark.models import Bench


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--seed", type=float)
    parser.add_argument("--model", type=str, default="benchmark")
    parser.add_argument("--quality-level", type=int, default=3)
    parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate",type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)

    # parser.add_argument("--epochs", type=int, default=400)
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--gpu-id",
    #     type=str,
    #     default=0,
    #     help="GPU ids (default: %(default)s)",
    # )
    
    args = parser.parse_args(argv)
    return args


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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True:
            aux_loss.backward()

    return aux_loss_sum


def train_one_epoch(
    model, train_dataloader, optimizer, aux_optimizer, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, batch in enumerate(train_dataloader):
        d = [frames.to(device) for frames in batch]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        dpb = {
                    "ref_frame": d[0],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }

        result = model(d[1], dpb)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        aux_optimizer.step()
        
        if i*len(d) % 5000 == 0:
            logging.info(
                f'[{i*len(d)}/{len(train_dataloader.dataset)}] | '
                # f'Loss: {stats["loss"]:.3f} | '
                # f'MSE loss: {stats["mse"]:.5f} | '
                # f'Bpp loss: {stats["kl"]:.4f}'


                # f"\tLoss: {out_criterion["loss"].item():.3f} |"
                # f"\tMSE loss: {out_criterion["mse_loss"].item():.3f} |"
                # f"\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |"
                # f"\tAux loss: {aux_loss.item():.2f}"
            )


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            stats = model(d)
            # out_criterion = criterion(out_net, d)

            bpp_loss.update(stats["kl"])
            loss.update(stats["loss"])
            mse_loss.update(stats["mse"])

    logging.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"Bpp loss: {bpp_loss.avg:.4f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    base_dir = f'./checkpoint/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    setup_logger(base_dir + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    # Warning, the order of the transform composition should be kept.
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = VideoFolder(
        args.dataset,
        rnd_interval=True,
        rnd_temp_order=True,
        split="train",
        transform=train_transforms,
    )
    test_dataset = VideoFolder(
        args.dataset,
        rnd_interval=False,
        rnd_temp_order=False,
        split="test",
        transform=test_transforms,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    model = Bench()
    model = model.to(device)

    optimizer, aux_optimizer = configure_optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,], gamma=0.1)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            aux_optimizer,
            args.clip_max_norm,
        )
        # loss = test_epoch(epoch, test_dataloader, model)
        # lr_scheduler.step(loss)

        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

        # if args.save:
        #     save_checkpoint(
        #         {
        #             "epoch": epoch,
        #             "state_dict": model.state_dict(),
        #             "loss": loss,
        #             "optimizer": optimizer.state_dict(),
        #             "aux_optimizer": aux_optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #         },
        #         is_best,
        #     )

if __name__ == '__main__':
    main(sys.argv[1:])
