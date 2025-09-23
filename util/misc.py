# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from packaging import version
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',  # occupation of space log_msg.format
                'eta: {eta}',  # keywords parameters
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:  # print every 10 video clips (including num_frames(10) in per clip)
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def collate_fn(batch):
    featuremaps, bboxes, actions, activities, wimg_activity, one_hot_matrix, meta = list(zip(*batch))  # featuremap: tuple?
    featuremaps = torch.stack(featuremaps)
    fm_mask = torch.zeros_like(featuremaps)
    featuremaps = NestedTensor(featuremaps, fm_mask)

    bboxes = nested_tensor_from_tensor_list(bboxes)

    padded_actions = pad_sequence(actions, batch_first=True, padding_value=-1)
    action_mask = (padded_actions == -1)  # the area for the real tensor in the mask is FALSE
    actions = NestedTensor(padded_actions, action_mask)

    padded_activities = pad_sequence(activities, batch_first=True, padding_value=-1)
    activity_mask = (padded_activities == -1)
    activities = NestedTensor(padded_activities, activity_mask)

    wimg_activity = torch.tensor(wimg_activity)

    one_hot_matrix = nested_tensor_from_tensor_list(one_hot_matrix)

    targets = [bboxes, actions, activities, wimg_activity, one_hot_matrix]

    return featuremaps, targets, meta


def _max_by_axis(the_list):  # list of tensor shape list (3 or 4 dimensions)
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]  # the first tensor shape
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):  # the dimensions of tensor shape (3 or 4)
            maxes[index] = max(maxes[index], item)  # compare the current tensor shape and the current maximum one
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    max_size = _max_by_axis([list(ini_tensor.shape) for ini_tensor in tensor_list])  # max size in each dimension
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size
    '''
    if bbox:
    batch_shape: batch size, num_bboxes, 4
    if one hot matrix:
    batch_shape: batch size, num_persons, num_groups
    '''
    b, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)  # tensor with max size
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for ini_tensor, pad_tensor, m in zip(tensor_list, tensor, mask):
        pad_tensor[: ini_tensor.shape[0], : ini_tensor.shape[1]].copy_(ini_tensor)  # fill corresponding positon of pad tensor with ini tensor
        m[: ini_tensor.shape[0], :ini_tensor.shape[1]] = False  # the area for the real tensor in the mask is FALSE
    return NestedTensor(tensor, mask)  # return a class object


def nested_tensor_from_fm_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 4:
        max_size = _max_by_axis([list(fm.shape) for fm in tensor_list])  # fm.shape: nf, c, h, w
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, nf, c, h, w = batch_shape
        # print('b, nf, c, h, w', batch_shape)
        '''
        if img:
        batch_shape: batch size, num_frames, channels, h, w
        '''
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for fm, pad_fm, m in zip(tensor_list, tensor, mask):
            # fm.shape: nf, c, h, w
            pad_fm[: fm.shape[0], : fm.shape[1], : fm.shape[2], : fm.shape[3]].copy_(fm)
            m[: fm.shape[2], :fm.shape[3]] = False  # 0：fm.shape
        # if mask on nf axis is needed, broadcast as follows:
        # m = mask.unsqueeze(1).expand(b, nf, h, w)
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def crop_to_original(mask):
    """
    tensor: [batch size, num_frames, c, h, w]
    mask: [batch size, h, w] for each feature map in the batch
    find the min indices [bs, h, w] of mask where the element is FALSE for each fm in the batch
    """
    valid_areas = []
    for i in range(mask.shape[0]):
        indices = mask[i].eq(False).nonzero(as_tuple=False)
        if mask.dim() == 3:
            y_min, x_min = indices.min(dim=0)[0]  # .min return: values and indices
            y_max, x_max = indices.max(dim=0)[0]
            valid_area = [y_min, y_max + 1, x_min, x_max + 1]
            valid_areas.append(valid_area)
        if mask.dim() == 2:
            min = indices.min()
            max = indices.max()
            valid_area = [min, max + 1]
            valid_areas.append(valid_area)
    valid_areas = torch.tensor(valid_areas)
    return valid_areas

def binary_label_smoothing(target, eps, bicls):
    """
    smoothed labels:
    0 -> eps/(num_classes - 1)，1 -> 1 - eps
    """
    if bicls == True:
        num_classes = target.size(-1)
        return target * (1 - eps) + eps / num_classes
    else:
        return target * (1 - eps) + 0.5 * eps


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # print(output, target)
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def per_class_accuracy(output, target, num_classes):
    _, pred = output.topk(1, dim=1)
    pred = pred.squeeze(1)

    acc_list = []
    for i in range(num_classes):
        mask = target == i
        total = mask.sum()
        if total == 0:
            acc_list.append(float('nan'))
        else:
            correct = (pred[mask] == i).sum()
            acc_list.append(100.0 * correct.item() / total.item())
    return acc_list


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None, normalize=True, figsize=(10, 8), cmap="Blues"):
    """
    make confusion matrix and normalize
    """
    labels = list(range(len(class_names)))  # sort according to the order of classes

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(all='ignore'):  # avoid 0 warning
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # transfer nan to 0

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(include_values=True, cmap=cmap, ax=ax, xticks_rotation=45)
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    save_path = save_path + '/confusion_matrix.jpg'
    plt.savefig(save_path, bbox_inches="tight")
    # plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    # plt.tight_layout()
    # plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    print precision, recall, f1-score of each class
    """
    print(classification_report(y_true, y_pred, target_names=class_names))


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
