import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default='output_dir/test', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--feature_file', default='collective',
                    help='choose the dataset: collective or volleyball')
parser.add_argument('--img_path',
                    default='/home/jiqqi/data/new-new-collective/ActivityDataset', type=str)
parser.add_argument('--ann_path',
                    default='/home/jiqqi/data/social_CAD/anns', type=str)
parser.add_argument('--img_w', default=224, type=int,
                    help='width of resized images')
parser.add_argument('--img_h', default=224, type=int,
                    help='heigh of resized images')
parser.add_argument('--is_training', default=True, type=bool,
                    help='data preparation may have differences')
parser.add_argument('--num_frames', default=17, type=int,
                    help='number of stacked frame features')

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np
import util.misc as utils
from util.misc import accuracy
from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset

from dataset import build_dataset



def run(init_lr=0.01, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip(),
    # ])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    os.makedirs(save_model, exist_ok=True)
    dataset, val_dataset = build_dataset(args=args)

    # dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=utils.collate_fn)

    # val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=utils.collate_fn)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}


    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        # i3d = InceptionI3d(400, in_channels=3)
        i3d = InceptionI3d(157, in_channels=3)
        # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        i3d.load_state_dict(torch.load('models/rgb_charades.pt'))
    i3d.replace_logits(6)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [90, 50000])
    writer_dir = args.save_model.split('/')[-1]
    writer = SummaryWriter(log_dir=f'runs/{writer_dir}')

    num_steps_per_update = 4 # accum gradient (update the parameters after accumulated steps)
    steps = 0
    # train it
    while steps < max_steps: # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            error = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            phase_probs = []
            phase_labels = []
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels, meta = data
                inputs, _ = inputs.decompose()  # B, T, H, W, C (original transform); B, T, C, H, W (own transform)
                labels = labels[3]

                # wrap them in Variable
                # inputs = Variable(inputs.cuda())
                inputs = inputs.to('cuda')
                # inputs = inputs.permute(0, 4, 1, 2, 3).contiguous()  # B, C, T, H, W
                inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W
                t = inputs.size(2)

                # labels = Variable(labels.cuda())
                labels = labels.to('cuda')


                logits = i3d(inputs)
                logits = logits.mean(dim=-1)
                # logits = torch.squeeze(logits)  # only 224 * 224 can be -> bs * num_cls

                # upsample to input size (we only have label for the whole clip)
                # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                # loc_loss = F.binary_cross_entropy_with_logits(logits, labels)
                # tot_loc_loss += loc_loss.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                cls_loss = F.cross_entropy(logits, labels)

                tot_cls_loss += cls_loss.item()
                # loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                loss = cls_loss/num_steps_per_update
                tot_loss += loss.item()

                if num_iter == num_steps_per_update and phase == 'train':
                    loss.backward()
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        print('{} Tot Loss: {:.4f}'.format(phase, tot_loss/10))
                        writer.add_scalar("Train/Loss", tot_loss / 10, steps)
                        tot_loss  = tot_cls_loss = 0.
                    if steps % 100 == 0:
                        torch.save(i3d.module.state_dict(), save_model + '/'+ str(steps).zfill(6) + '.pt')

                with torch.no_grad():
                    logits_bt = logits
                    labels_bt = labels
                    phase_probs.append(logits_bt.detach().cpu())
                    phase_labels.append(labels_bt.detach().cpu())
            logits_all = torch.cat(phase_probs, dim=0)  # [N, C]
            labels_all = torch.cat(phase_labels, dim=0)  # [N, C]
            error = 100 - accuracy(logits_all, labels_all)[0]

            if phase == 'train':
                writer.add_scalar("Train/Error", error, steps)

            if phase == 'val':
                # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                print('{} Tot Loss: {:.4f}'.format(phase, (tot_loss*num_steps_per_update)/num_iter))
                writer.add_scalar("Test/Loss", (tot_loss*num_steps_per_update)/num_iter, steps)
                writer.add_scalar("Test/Error", error, steps)
    writer.close()


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
