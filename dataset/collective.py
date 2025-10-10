import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path
from collections import Counter

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720),
               6: (480, 720), 7: (480, 720), 8:  (480, 720), 9:  (480, 720), 10: (480, 720),
               11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800),
               16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800),
               21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720),
               26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720),
               31: (480, 720),  32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720),
               36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720),
               41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}

def collective_path(img_root, ann_root):
    # train_seqs = [str(i + 1) for i in range(32)]
    # val_seqs = [str(i + 1) for i in range(32)]
    # val_seqs = [str(i + 33) for i in range(12)]
    # val_seqs = [str(44)]

    # for testing code runing one seq
    # train_seqs = [str(i + 1) for i in range(1)]
    # val_seqs = [str(i + 1) for i in range(1)]

    # random seqs
    all_seqs = [str(i + 1) for i in range(44)]
    random.shuffle(all_seqs)
    train_seqs = all_seqs[:32]
    val_seqs = all_seqs[32:]

    train_seq_path = [img_root / ('seq' + train_seq.zfill(2)) for train_seq in train_seqs]
    val_seq_path = [img_root / ('seq' + val_seq.zfill(2)) for val_seq in val_seqs]
    train_ann_path = [ann_root / (train_seq + '_annotations.txt') for train_seq in train_seqs]
    val_ann_path = [ann_root / (val_seq + '_annotations.txt') for val_seq in val_seqs]

    train_img_path = [file for seq in train_seq_path for file in seq.rglob("*")]
    val_img_path = [file for seq in val_seq_path for file in seq.rglob("*")]  # find all folders and files inside

    PATHS = {
        "train": (train_img_path, train_ann_path),
        "val": (val_img_path, val_ann_path),
    }

    train_img_file, train_ann_file = PATHS['train']
    test_img_file, test_ann_file = PATHS['val']  # imgs and anns paths

    return train_img_file, train_ann_file, test_img_file, test_ann_file


def collective_read_annotations(ann_file):
    annotations = {}  # annotations for each frame
    with open(ann_file, 'r') as ann_txt:
        se_anns = ann_txt.readlines()
        for se_ann in se_anns:
            se_ann = se_ann.rstrip()
            frame_id = int(se_ann.split('\t')[0])
            if frame_id not in annotations:
                annotations[frame_id] = {}
                annotations[frame_id]['groups'] = []
                annotations[frame_id]['persons'] = []
            x1 = float(se_ann.split('\t')[1])
            y1 = float(se_ann.split('\t')[2])
            x2 = float(se_ann.split('\t')[3])
            y2 = float(se_ann.split('\t')[4])  # absolute coord
            if x1 < 0:
                x1 = 0.0
            bbox = [x1, y1, x2, y2]
            action = int(se_ann.split('\t')[5]) - 1  # start by 0
            person_id = int(se_ann.split('\t')[7]) - 1
            group_id = int(se_ann.split('\t')[8]) - 1

            if any(person.get('group_id') == group_id for person in annotations[frame_id]['persons']):
                group = [group for group in annotations[frame_id]['groups'] if group.get('group_id') == group_id][0]
                group['include_id'].append(person_id)
            else:
                activity = int(se_ann.split('\t')[6])
                annotations[frame_id]['groups'].append({
                                'group_id': group_id,
                                'activity': activity,
                                'include_id': [person_id]
                            })

            annotations[frame_id]['persons'].append({
                            'person_id': person_id,
                            'bbox': bbox,
                            'action': action,
                            'group_id': group_id
                        })
            annotations[frame_id]['persons'].sort(key=lambda x: x['person_id'])
    # print(annotations)
    return annotations


def collective_read_dataset(ann_files):
    data = {}
    for ann_file in ann_files:
        sid = int(str(ann_file).split('/')[-1].split('_')[0])
        data[sid] = collective_read_annotations(ann_file)  # data for each seq
    return data


'''
data stucture:
data: {1: {ann1}, 2: {ann2}, ... , seq: {annseq}}
ann1(seq): {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1(seq)-1(frame): {persons: [{person_id: 1, bbox: [], action:, 1, group_id: 1}, ..., {...}], 
        groups: [{group_id: 1, activity: 1, person_ids: []}, ..., {...}]}
'''


def collective_all_frames(anns, num_frames):
    half_left = num_frames // 2
    half_right = num_frames - half_left
    return [(s, f) for s in anns for f in anns[s] if f != 1 and f != max(anns[s]) and f + half_right <= max(anns[s]) and f - half_left >= 1]
    # (sid, fid) with anns (every 10 frames: eg. 11, 21, 31, ...)
    # filtered the first and the last anns


class Collective_Dataset(data.Dataset):
    def __init__(self, anns, frames, img_path, transform, num_frames=10, is_training=True):
        """
        Args:
            Characterize collective dataset based on feature maps.
        """
        self.anns = anns
        self.frames = frames
        self.img_path = img_path

        self.num_frames = num_frames  # number of stacked frame features
        self.is_training = is_training

        self.transform = transform

    def __len__(self):
        return len(self.frames)  # number of frames with anns (filtered the first and thea last ones)
        # TODO: put key frame in middle and take 10 frames currently
        #  put key frames at the third and seventh location later

    def __getitem__(self, idx):
        # Load feature map

        select_frames = self.get_frames(self.frames[idx])  # 10 frames
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]
            # normal training: each training loading 10 frames
        else:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]            # normal testing: each test loading 10 frames

    def load_samples_sequence(self, select_frames):
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         device = torch.device('cpu')

        sid = select_frames[0]
        src_fid = select_frames[1]

        person_ids = []
        bboxes = []
        actions = []
        # p_group_ids = []

        group_ids = []
        activities = []
        include_ids = []

        for person in self.anns[sid][src_fid]['persons']:
            person_id = person['person_id']  # to connect the prediction and build pred matrix for iou group-level loss of group members
            person_ids.append(person_id)  # it is not in order in the tensor, following the order of group id, and some disappear
            bbox = person['bbox']  # for ROI Align and positional encoding
            bboxes.append(bbox)
            action = person['action']  # gt individual actions
            actions.append(action)
            # p_group_id = person['group_id']  # gt for person-level cross-entropy loss of group members
            # p_group_ids.append(p_group_id)

        for group in self.anns[sid][src_fid]['groups']:
            activity = group['activity']  # gt for group activity
            activities.append(activity)
            # group_id = group['group_id']
            # group_ids.append(group_id)
            include_id = group['include_id']
            include_ids.append(include_id)

        # get the activity id for the whole img (for debug and comparison)
        wimg_groups = self.anns[sid][src_fid]['groups']
        wimg_activities = [g["activity"] for g in wimg_groups]
        counter = Counter(wimg_activities)
        wimg_activity, freq = counter.most_common(1)[0]

        num_persons = len(person_ids)
        num_groups = len(activities)
        one_hot_matrix = np.zeros((num_persons, num_groups), dtype=float)
        # TODO: check if the order of column and the row should change
        person_to_index = {p: i for i, p in enumerate(person_ids)}

        for group, persons in enumerate(include_ids):
            for person in persons:
                one_hot_matrix[person_to_index[person], group] = 1

        imgs = []
        bbox = bboxes.copy()
        bbox = np.array(bbox, dtype=np.float64).reshape(-1, 4)
        for i, (sid, src_fid, fid) in enumerate(select_frames[2]):  # 10 frames for 1 item
            img = cv2.imread(self.img_path + '/seq%02d/frame%04d.jpg' % (sid, fid))[:, :, [2, 1, 0]]  # BGR -> RGB  # H, W, 3
            img = Image.fromarray(img)
            img, bboxes = self.transform(img, bbox)
            imgs.append(img)

        # labels for the whole video clip (the label of the key frame)
        meta = {}
        imgs = np.stack(imgs)

        # imgs = self.transform(imgs)  # t, h, w, c = img.shape
        # bboxes = np.array(bbox, dtype=np.float64).reshape(-1, 4)

        actions = np.array(actions, dtype=np.int32)
        activities = np.array(activities, dtype=np.int32)
        one_hot_matrix = np.array(one_hot_matrix, dtype=np.int32)

        imgs = torch.from_numpy(imgs).float()
        imgs = torch.squeeze(imgs, 1)
        bboxes = torch.from_numpy(bboxes).float()
        # bboxes = torch.unsqueeze(bboxes[0], 0)
        actions = torch.from_numpy(actions).long()
        # actions = torch.unsqueeze(actions[0], 0)
        activities = torch.from_numpy(activities).long()
        one_hot_matrix = torch.from_numpy(one_hot_matrix).int()

        meta['sid'] = sid
        meta['src_fid'] = src_fid
        meta['frame_size'] = FRAMES_SIZE[sid]

        return imgs, bboxes, actions, activities, wimg_activity, one_hot_matrix, meta
