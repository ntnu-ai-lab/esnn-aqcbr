import torch
from torch.utils.data import Dataset
import random
from dataset.makeTrainingData import makeDualSharedArchData
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required
import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnx
from pytorch2keras import pytorch_to_keras


def _pytorch2keras(model, input_var, shape= [(10, 32, 32,)]):
    # input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
    # we should specify shape of the input tensor
    k_model = pytorch_to_keras(model, input_var, shape, verbose=True)
    return k_model


def torch_to_keras(model, input_list, output_list, ex_input, name):
    onnxfile = f"{name}.onnx"
    kerasfile = f"{name}k.h5"
    torch_to_onnx(model, ex_input, input_list, output_list, onnxfile)
    _onnx_to_keras(onnxfile, kerasfile, input_list)


def torch_to_onnx(model, ex_input, input_list, output_list, filename):
    torch.onnx.export(model, ex_input, filename, verbose=True,
                      input_names=input_list, output_names=output_list)


def _onnx_to_keras(onnxfile, kerasfile, inputlist):
    onnx_model = onnx.load(onnxfile,)
    k_model = onnx_to_keras(onnx_model, inputlist)
    keras.models.save_model(k_model, kerasfile,
                            overwrite=True,
                            include_optimizer=True)


def _torch_abs(x1, x2):
    return torch.sqrt(torch.pow(x1-x2, 2))


def _torch_abs2(x):
    return torch.sqrt(torch.pow(x, 2))


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        if torch.isnan(loss_contrastive):
            print("to nan!")
        return loss_contrastive


class ESNNloss(torch.nn.Module):


    def __init__(self, alpha=0.1):
        super(ESNNloss, self).__init__()
        self.alpha = alpha

    def forward(self, distance, label, y1, y2, y1_hat, y2_hat):
        ha = (1-self.alpha)/2
        loss = (self.alpha)*_torch_abs(label-distance)
        loss = loss + (ha * _torch_abs(y1-y1_hat)) + \
            (ha * _torch_abs(y2-y2_hat))
        return loss


class TorchSKDataset(Dataset):
    def __init__(self, sklearndataset, transform=None,should_invert=True):
        super(TorchSKDataset, self).__init__()

        self.sklearandataset = sklearndataset
        self.sklearandatasetDualShared, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeDualSharedArchData(sklearndataset.getFeatures(),
                                       sklearndataset.getTargets(), False)
        self.transform = transform
        self.should_invert = should_invert
        self.featurelength = sklearndataset.featurecolsto - \
            sklearndataset.featurecolsfrom
        self.x1start = 0
        self.x2start = self.featurelength+1

        self.x1stop = self.featurelength+1
        self.x2stop = 2*(self.featurelength+1)

    def __getitem__(self, index):
        return self.getItem(index)

    def getItem(self, index):
        matrix = self.sklearandatasetDualShared
        x1 = torch.from_numpy(matrix[index, self.x1start:self.x1stop].astype('float')).to(torch.float32)
        x2 = torch.from_numpy(matrix[index, self.x2start:self.x2stop].astype('float')).to(torch.float32)
        #l1 = self.y1[index]
        #l2 = self.y2[index]
        #label = torch.from_numpy(np.array(int(np.all(l1 == l2))))
        #label = torch.from_numpy(np.array([int(x1 != x2[1])],
        #dtype=np.float32))
        label = torch.from_numpy(self.targets[index].astype('float')).to(torch.float32)
        return x1, x2, label

    def getOrigItem(self, index):
        randind = random.randint(0, self.sklearandataset.size()-1)
        data_tuple0 = self.sklearandataset.getTuple(randind)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                randind = random.randint(0, self.sklearandataset.size()-1)
                data_tuple1 = self.sklearandataset.getTuple(randind)
                if np.all(data_tuple0[1] == data_tuple1[1]):
                    break
        else:
            while True:
                #keep looping till a different class image is found

                randind = random.randint(0, self.sklearandataset.size()-1)
                data_tuple1 = self.sklearandataset.getTuple(randind)
                if np.all(data_tuple0[1] != data_tuple1[1]):
                    break

    def __len__(self):
        return self.sklearandataset.size()

import math
import torch
from torch.optim.optimizer import Optimizer, required


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
