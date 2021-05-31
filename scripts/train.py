#!/usr/bin/env python

import argparse


desc = 'Self-supervised super-resolution.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--image')
parser.add_argument('-o', '--output-dir')
parser.add_argument('-p', '--patch-size', default=[32, 32], type=int, nargs=2)
parser.add_argument('-s', '--slice-profile', default='gaussian')
parser.add_argument('-d', '--num-blocks', default=8, type=int)
parser.add_argument('-w', '--num-channels', default=256, type=int)
parser.add_argument('-l', '--learning-rate', default=0.0001, type=float)
parser.add_argument('-e', '--num-epochs', default=3, type=int)
parser.add_argument('-b', '--num-batches', default=10000, type=int)
parser.add_argument('-B', '--batch-size', default=16, type=int)
parser.add_argument('-S', '--patch-save-step', default=50, type=int)
parser.add_argument('-z', '--patch-save-zoom', default=4, type=int)
parser.add_argument('-g', '--num-groups', default=4, type=int)
parser.add_argument('-f', '--following-num-batches', default=100, type=int)
parser.add_argument('-O', '--optim', default='adam')
parser.add_argument('-L', '--loss-func', default='l1')
parser.add_argument('-m', '--model', default='rcan')
parser.add_argument('-v', '--valid-step', type=int, default=100)
parser.add_argument('-V', '--num-valid-samples', type=int, default=128)
parser.add_argument('-P', '--pred-batch-step', type=int, default=float('inf'))
parser.add_argument('-F', '--pred-following-batch-step', type=int,
                    default=float('inf'))
parser.add_argument('-E', '--pred-epoch-step', type=int, default=1)
parser.add_argument('-Z', '--pred-batch-size', type=int, default=64)
parser.add_argument('-D', '--debug', action='store_true')
parser.add_argument('-r', '--set-random-seed', action='store_true')
parser.add_argument('-c', '--checkpoint')
parser.add_argument('-C', '--checkpoint-save-step', type=int, default=float('inf'))
args = parser.parse_args()


from sssr.train import TrainerBuilder

if args.set_random_seed:
    import torch
    import random
    import numpy as np
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

trainer = TrainerBuilder(args).build().trainer
trainer.train()
