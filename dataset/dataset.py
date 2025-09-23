from pathlib import Path

import torch
import torch.utils.data as data
import torchvision

from torchvision import datasets, transforms
import videotransforms

from .collective import collective_path, collective_read_dataset, collective_all_frames, Collective_Dataset

def build(args):
    img_root = Path(args.img_path)
    assert img_root.exists(), f'provided image path {img_root} does not exist'
    ann_root = Path(args.ann_path)
    assert ann_root.exists(), f'provided bbox path {ann_root} does not exist'

    num_frames = args.num_frames

    if args.feature_file == 'collective':
        train_img_file, train_ann_file, test_img_file, test_ann_file = collective_path(img_root, ann_root)

        train_anns = collective_read_dataset(train_ann_file)  # ann dictionary
        train_frames = collective_all_frames(train_anns, num_frames)  # frame and sec ids: (s, f)
        # print(train_frames)

        test_anns = collective_read_dataset(test_ann_file)
        test_frames = collective_all_frames(test_anns, num_frames)

        # train_transform = visiontransforms.Compose([
        # visiontransforms.RandomHorizontalFlip(),
        # visiontransforms.Resize((args.img_h, args.img_w)),  # bbox resize is integrated in roialingn part
        # visiontransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # visiontransforms.ToTensor(),
        # visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        #
        # test_transform = visiontransforms.Compose([
        # visiontransforms.Resize((args.img_h, args.img_w)),
        # visiontransforms.ToTensor(),
        # visiontransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])

        train_transform = transforms.Compose([videotransforms.RandomCrop(224),
                                               videotransforms.RandomHorizontalFlip(),
        ])
        test_transform = transforms.Compose([videotransforms.CenterCrop(224)])

        train_dataset = Collective_Dataset(train_anns, train_frames, args.img_path, train_transform,
                                          num_frames=args.num_frames, is_training=args.is_training)
        test_dataset = Collective_Dataset(test_anns, test_frames, args.img_path, test_transform,
                                         num_frames=args.num_frames, is_training=args.is_training)

    else:
        ValueError("Invalid dataset.")

    return train_dataset, test_dataset

