import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', default=1, type=int)
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-root_dir', default='/media/jiqqi/新加卷/dataset/Charades_v1_rgb', type=str)
parser.add_argument('-train_split', default='/media/jiqqi/新加卷/dataset/charades.json', type=str)
parser.add_argument('-save_model', default='output_dir/test', type=str)
parser.add_argument('-root', type=str)

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

from sklearn.metrics import average_precision_score
from pytorch_i3d import InceptionI3d

from charades_dataset import Charades as Dataset


def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/media/jiqqi/新加卷/dataset/Charades_v1_rgb', train_split='/media/jiqqi/新加卷/dataset/charades.json', batch_size=8*5, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    os.makedirs(save_model, exist_ok=True)
    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(157)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    writer_dir = args.save_model.split('/')[-1]
    writer = SummaryWriter(log_dir=f'runs/{writer_dir}')


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])  # sigmoid + BCE
                tot_cls_loss += cls_loss.item()

                # compute total loss
                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        writer.add_scalar("Train/cls_loss", tot_cls_loss / (10*num_steps_per_update), steps)
                        writer.add_scalar("Train/loc_loss", tot_loc_loss / (10*num_steps_per_update), steps)
                        writer.add_scalar("Train/Loss", tot_loss / 10, steps)

                        # compute classification errors
                        probs = torch.sigmoid(per_frame_logits).amax(dim=2)  # [B, C]
                        labels = labels.float().amax(
                            dim=2)  # [B, C]  # get the maximum logits/lables for each cls over time dimension  (if one cls appears for 1 frame, it is positive in this clip)
                        preds = (probs > 0.5).float()
                        mAP = average_precision_score(labels.cpu().numpy(),
                                                      probs.detach().cpu().numpy(),
                                                      average='macro')
                        writer.add_scalar("Train/Average_precision", mAP, steps)

                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
                    if steps % 100 == 0:
                        torch.save(i3d.module.state_dict(), save_model + '/'+ str(steps).zfill(6) + '.pt')
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
                writer.add_scalar("Test/cls_loss", tot_cls_loss / num_iter, steps)
                writer.add_scalar("Test/loc_loss", tot_loc_loss / num_iter, steps)
                writer.add_scalar("Test/Loss", (tot_loss*num_steps_per_update)/num_iter, steps)
                writer.add_scalar("Test/Average_precision", mAP, steps)
    writer.close()


if __name__ == '__main__':
    # need to add argparse
    run(root=args.root_dir, train_split=args.train_split, save_model=args.save_model, batch_size=args.batch_size)
