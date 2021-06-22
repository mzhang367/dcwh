# coding: utf-8
import torch
import numpy as np
import pickle
import os
import sys
import errno
import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x = np.clip(x, -1, 1)
    return x


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def compute_result(dataloader, net, device):

    """
    return hashing codes of data with shape (N, len_bits) and its labels (N, )
    """
    hash_codes = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs, cls = imgs.to(device), cls.to(device)
        hash_values = net(imgs)
        hash_codes.append(hash_values.data)
        label.append(cls)

    hash_codes = torch.cat(hash_codes)
    B = torch.where(hash_codes > 0.0, torch.tensor([1.0]).cuda(), torch.tensor([-1.0]).cuda())

    return B, torch.cat(label)


def compute_topK(trn_binary, tst_binary, trn_label, tst_label, device, top_list):

    """
    compute mean precision of returned top-k results based on Hamming ranking
    """

    top_p = torch.Tensor(tst_binary.size(0), len(top_list)).to(device)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        for j, top in enumerate(top_list):
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[i, j] = 1.0*N_top/top

    top_pres = top_p.mean(dim=0).cpu().numpy()

    return top_pres


def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device):

    AP = []
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        N = torch.sum(correct)
        Ns = torch.arange(1, N+1).float().to(device)
        index = (torch.nonzero(correct, as_tuple=False)+1)[:, 0].float()
        AP.append(torch.mean(Ns / index))

    mAP = torch.mean(torch.Tensor(AP))
    return mAP


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.cpu().view(-1, 1), 1)
    return target_onehot
