import argparse
import sys

from .train import TrainerBuilder


def train(args=None):
    args = sys.argv[1:] if args is None else args
    desc = 'Self-supervised super-resolution.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-p', '--patch-size', default=[32, 32], type=int, nargs=2)
    parser.add_argument('-s', '--slice-profile', default='gaussian')
    parser.add_argument('-d', '--num-blocks', default=8, type=int)
    parser.add_argument('-w', '--num-channels', default=64, type=int)
    parser.add_argument('-l', '--learning-rate', default=2e-4, type=float)
    parser.add_argument('-e', '--num-epochs', default=4, type=int)
    parser.add_argument('-b', '--num-batches', default=20000, type=int)
    parser.add_argument('-B', '--batch-size', default=32, type=int)
    parser.add_argument('-S', '--patch-save-step', default=10000, type=int)
    parser.add_argument('-z', '--patch-save-zoom', default=4, type=int)
    parser.add_argument('-g', '--num-groups', default=2, type=int)
    parser.add_argument('-f', '--following-num-batches', default=2000, type=int)
    parser.add_argument('-O', '--optim', default='adam')
    parser.add_argument('-L', '--loss-func', default='l1')
    parser.add_argument('-m', '--model', default='rcan')
    parser.add_argument('-v', '--valid-step', type=int, default=100)
    parser.add_argument('-V', '--num-valid-samples', type=int, default=128)
    parser.add_argument('-P', '--pred-batch-step', type=int, default=1000)
    parser.add_argument('-F', '--pred-following-batch-step', type=int, default=1000)
    parser.add_argument('-E', '--pred-epoch-step', type=int, default=1)
    parser.add_argument('-Z', '--pred-batch-size', type=int, default=64)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-r', '--set-random-seed', action='store_true')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-C', '--checkpoint-save-step', type=int, default=1)
    parsed_args = parser.parse_args(sys.argv[1:] if args is None else args)

    if parsed_args.set_random_seed:
        import torch
        import random
        import numpy as np
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

    trainer = TrainerBuilder(parsed_args).build().trainer
    trainer.train()
