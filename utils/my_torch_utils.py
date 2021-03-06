import torch
from torch.utils.data import Dataset
import random
from dataset.makeTrainingData import makeDualSharedArchData, makeGabelTrainingData
import numpy as np

# def _pytorch2keras(model, input_var, shape= [(10, 32, 32,)]):
#     # input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
#     # we should specify shape of the input tensor
#     k_model = pytorch_to_keras(model, input_var, shape, verbose=True)
#     return k_model
#
#
# def torch_to_keras(model, input_list, output_list, ex_input, name):
#     onnxfile = f"{name}.onnx"
#     kerasfile = f"{name}k.h5"
#     torch_to_onnx(model, ex_input, input_list, output_list, onnxfile)
#     _onnx_to_keras(onnxfile, kerasfile, input_list)
#
#
# def torch_to_onnx(model, ex_input, input_list, output_list, filename):
#     torch.onnx.export(model, ex_input, filename, verbose=True,
#                       input_names=input_list, output_names=output_list)


# def _onnx_to_keras(onnxfile, kerasfile, inputlist):
#     onnx_model = onnx.load(onnxfile,)
#     k_model = onnx_to_keras(onnx_model, inputlist)
#     keras.models.save_model(k_model, kerasfile,
#                             overwrite=True,
#                             include_optimizer=True)


####
# AUC loss for torch based on https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e
###
# dense = L.Dense(units=1)
# activations = dense(input_layer)
# predictions = tf.sigmoid(activations)
# # This is the new cost function.
# cost = - tf.reduce_mean(tf.sigmoid(activations @ tf.transpose(activations)) * np.maximum(y @ np.ones(y.shape).T - np.ones(y.shape) @ y.T, 0))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# training_op = optimizer.minimize(cost)
#def auc_loss()


def cross_entropy(y_hat, y):
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(- y * logsoftmax(y_hat), 1))

def _torch_abs(x1, x2):
    return torch.sqrt(torch.pow(x1-x2, 2))

sent_value = [2]
sent_value_tensor = torch.tensor(sent_value, device='cuda:0')

sent_value_zero = [0]
sent_value_zero_tensor = torch.tensor(sent_value_zero, device='cuda:0')

sent_value_one = [1]
sent_value_one_tensor = torch.tensor(sent_value_one, device='cuda:0')

sent_value_two = [2]
sent_value_two_tensor = torch.tensor(sent_value_two, device='cuda:0')

sent_value_three = [3]
sent_value_three_tensor = torch.tensor(sent_value_three, device='cuda:0')

def torch_auc_roc(y_true, y_pred, start, stop, delta):
    """

    """
    #true_positives = torch.sum(torch.where(y_pred >= .5, 1, 0))
    #negatives = torch.sum(torch.where(y_pred < .5, 1, 0))
    
    # this holds all the thredholds
    thresholds = torch.arange(start, stop, delta)
    # this holds the indexes of all the threholds for us to store in tables
    inds = torch.arange(0, thresholds.shape[0])
    # these two tensors stores all the true positive rates and false postive rates.
    tprs = torch.zeros(thresholds.shape[0], 1, device='cuda:0')
    fprs = torch.zeros(thresholds.shape[0], 1, device='cuda:0')

    n_same = torch.sum(torch.where(y_true == sent_value_one_tensor, sent_value_one_tensor, sent_value_zero_tensor))

    n_diff = torch.sum(torch.where(y_true == sent_value_zero_tensor, sent_value_one_tensor, sent_value_zero_tensor))
    for t, i in zip(thresholds, inds):
        # which of the network predictions are above the current threshold
        y_pos_thressed = torch.where(y_pred >= t, sent_value_two_tensor, sent_value_zero_tensor)
        # which of the network predictions are below the current threshold
        y_neg_thressed = torch.where(y_pred < t, sent_value_two_tensor, sent_value_zero_tensor)

        # add the ones above the thresholds to y_true so we can check who is over..
        pos_check = torch.add(y_pos_thressed, y_true)
        tp = torch.sum(torch.where(pos_check == sent_value_three_tensor, sent_value_one_tensor, sent_value_zero_tensor))

        #neg_check = torch.add(y_neg_thressed, y_true)
        #tn_corr = torch.sum(torch.where(neg_check == sent_value_zero_tensor, sent_value_one_tensor, sent_value_zero_tensor))

        fp = torch.sum(torch.where(pos_check == sent_value_two_tensor, sent_value_one_tensor, sent_value_zero_tensor))
        #fn = torch.sum(torch.where(neg_check == sent_value_two_tensor, sent_value_one_tensor, sent_value_zero_tensor))
        #tn = torch.sum(torch.where(neg_check == sent_value_three_tensor, sent_value_one_tensor, sent_value_zero_tensor))
        tp = tp.double()
        fp = fp.double()
        #tn = tn.double()
        #fn = fn.double()
        tprs[i] = tp / (n_same)
        fprs[i] = fp / (n_diff)

    #for tpr, fpr in zip(tprs, fprs):
    auc = torch.zeros(1, 1, device='cuda:0')
    #width = torch.zeros(1, 1, device='cuda:0')
    #avg_height = torch.zeros(1, 1, device='cuda:0')
    for i in range(1, tprs.shape[0]):
        width = fprs[i-1]-fprs[i]
        avg_height = (tprs[i]+tprs[i-1])/2.0
        auc += width*avg_height

    return auc

def torch_auc_roc_mm(y_pred, y_true, start, stop, delta):
    """

    """
    true_positives = torch.sum(torch.where(y_pred >= .5, 1, 0))
    negatives = torch.sum(torch.where(y_pred < .5, 1, 0))
    
    thresholds = torch.arange(start, stop, delta)
    inds = torch.arange(0, thresholds.shape(0))
    tprs = torch.zeros(thresholds.shape(0), 1)
    fprs = torch.zeros(thresholds.shape(0), 1)

    # filter = y_pred 

    return 0

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
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.001), 2))
        if torch.isnan(loss_contrastive):
            print("to nan!")
        return loss_contrastive


def xentropy_cost(x_target, x_pred):
    assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    if torch.isnan(cost_value):
        print("heh another nan")
    return cost_value

class ESNNloss(torch.nn.Module):


    def __init__(self, alpha=0.99):
        super(ESNNloss, self).__init__()
        self.alpha = alpha
        self.loss = torch.nn.BCELoss()
        self.xloss = torch.nn.CrossEntropyLoss()
        self.alpha = torch.from_numpy(np.asarray([alpha])).to('cuda:0')
        self.ha = ((1-self.alpha)/2).to('cuda:0')

    """
    y_hat is the output of C(G(x),G(y)) - the distance/similarity estimated by the model
    y is the true distance/sim
    y_x,y_x_hat is the class labels and outputs respectively for the classification outputs
    """
    def forward(self, y_hat, y,
                y1, y2, y1_hat, y2_hat):

        syhat = y_hat#.squeeze()
        sy = y#.squeeze()
        loss = (self.alpha)*self.loss(syhat, sy)
        loss = loss + (self.ha * self.xloss(y1_hat, torch.max(y1, 1)[1])) + \
            (self.ha * self.xloss(y2_hat, torch.max(y2, 1)[1]))
        if torch.isnan(loss):
            print("is naan")
        return loss


class TorchSKDataset(Dataset):
    def __init__(self, sklearndataset, trainidx, transform=None,should_invert=True, ):
        super(TorchSKDataset, self).__init__()

        self.sklearandataset = sklearndataset
        self.loadData(sklearndataset, trainidx)
        self.transform = transform
        self.should_invert = should_invert
        self.featurelength = sklearndataset.featurecolsto - \
            sklearndataset.featurecolsfrom
        self.x1start = 0
        self.x2start = self.featurelength+1

        self.x1stop = self.featurelength+1
        self.x2stop = 2*(self.featurelength+1)
        self.x1s = torch.from_numpy(self.sklearandatasetDualShared[:, self.x1start:self.x1stop].astype('float')).to(torch.float32)
        self.x2s = torch.from_numpy(self.sklearandatasetDualShared[:, self.x2start:self.x2stop].astype('float')).to(
            torch.float32)
        self.labels = torch.from_numpy(self.targets[:].astype('float')).to(torch.float32)

    def loadData(self, sklearndataset, trainidx):
        self.sklearandatasetDualShared, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeDualSharedArchData(sklearndataset.getFeatures()[trainidx],
                                       sklearndataset.getTargets()[trainidx], False)

    def __getitem__(self, index):
        return self.getItem(index)

    def getItem(self, index):
        x1 = self.x1s[index]
        x2 = self.x2s[index]
        l1 = self.y1[index]
        l2 = self.y2[index]
        return x1, x2, l1, l2, self.labels[index]

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
        return self.sklearandatasetDualShared.shape[0]

class GabelTorchDataset(TorchSKDataset):
    def loadData(self, sklearndataset, trainidx):
        self.sklearandatasetDualShared, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeGabelTrainingData(sklearndataset.getFeatures()[trainidx],
                                      sklearndataset.getTargets()[trainidx], False)

    def __init__(self, sklearndataset, trainidx, transform=None,should_invert=True):
        super(GabelTorchDataset, self).__init__(sklearndataset, trainidx)


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
