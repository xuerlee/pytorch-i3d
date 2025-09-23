import torch
import torch.utils.data
import torchvision

from .dataset import build

def build_dataset(args):
    if args.img_path:
        return build(args)
    raise ValueError(f'image path {args.img_path} not supported')
