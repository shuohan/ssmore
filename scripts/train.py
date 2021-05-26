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
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-e', '--num-epochs', default=10000, type=int)
parser.add_argument('-n', '--num-iters', default=3, type=int)
parser.add_argument('-I', '--image-save-step', default=50, type=int)
parser.add_argument('-P', '--use-padding', action='store_true')
parser.add_argument('-g', '--num-groups', default=4, type=int)
parser.add_argument('-f', '--following-num-epochs', default=100, type=int)
parser.add_argument('-S', '--iter-save-step', default=10, type=int)
parser.add_argument('-O', '--optim', default='adam')
parser.add_argument('-L', '--loss-func', default='l1')
parser.add_argument('-N', '--network', default='rcan')
parser.add_argument('-v', '--valid-step', type=int, default=100)
args = parser.parse_args()


from sssr.train import build_trainer


trainer = build_trainer(args)
trainer.train()
