import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from models.esnn.pytorch_model import ESNNModel
from utils.torch import ContrastiveLoss, RAdam, ESNNloss
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import RMSprop
import numpy as np

def get_linear_warmup_scheduler(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


    # parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--bs', type=int, default=32)
    # parser.add_argument('--epochs', type=int, default=5)
    # parser.add_argument('--gpus', type=int, default=1)
    # parser.add_argument('--distributed-backend', type=str, default=None)
    # parser.add_argument('--use-amp', type=bool, default=True)
    # parser.add_argument('--amp-level', type=str, default='O2')
    # config.betas = (0.9, 0.999)
    # config.eps = 1e-07
    # config.weight_decay = 1e-7
    # config.gradient_clip_val = 15
    # config.warmup_steps = 100
class ESNNSystem(pl.LightningModule):

    def __init__(self, data, X, Y, networklayers=[13, 13],
                 lr=1e-4, betas=(0.9, 0.999), eps=1e-07,
                 weight_decay=1e-7, warmup_steps=100,
                 val_check_interval=250,
                 val_percent_check=0.3,
                 validation_func=None,
                 train_data=None, train_target=None,
                 test_data=None, test_target=None,
                 colmap=None, device=None,
                 dropoutrate=0.2):
        super(ESNNSystem, self).__init__()
        # not the best model...
        self.model = ESNNModel(X, Y, networklayers, dropoutrate).to(torch.float32)
        self.loss = ESNNloss() #ContrastiveLoss()
        #self.esnnloss = ESNNloss()
        self.train_loader = data
        self.dev_loader = data
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.test_loader = data
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.val_check_interval = val_check_interval
        self.val_percent_check = val_percent_check
        self.evalfunc = validation_func
        self.train_data = train_data
        self.train_target = train_target

        self.test_data = test_data
        self.test_target = test_target
        self.colmap = colmap
        self.device = device
        #self.register_backward_hook(self.printgradnorm)

    def forward(self, input1, input2):
        return self.model.forward(input1, input2)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x1, x2, y1, y2, label = batch
        y_hat, inner_output1, inner_output2 = self.forward(x1, x2)
        loss = self.loss.forward(y_hat, label, y1,
                                 y2, inner_output1,
                                 inner_output2)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        #
        # res, errdistvec, truedistvec, \
        #     combineddata, pred_vec = self.evalfunc(self, self.test_data, self.test_target,
        #                                            self.train_data, self.train_target,
        #                                            batch_size=self.train_data.shape[0]*self.test_data.shape[0],
        #                                            anynominal=False, colmap=self.colmap,
        #                                            device=self.device)

        return None#{'val_loss': torch.from_numpy(np.asarray([np.sum(res)/len(res)]))}

    def validation_end(self, outputs):
        # OPTIONAL
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard_logs = {'val_loss': avg_loss}
        #self.current_val_loss = avg_loss
        #return {'avg_val_loss': avg_loss, 'lg': tensorboard_logs}
        res, errdistvec, truedistvec, \
            combineddata, pred_vec = self.evalfunc(self, self.test_data, self.test_target,
                                                   self.train_data, self.train_target,
                                                   batch_size=self.train_data.shape[0]*self.test_data.shape[0],
                                                   anynominal=False, colmap=self.colmap,
                                                   device=self.device)
        avg_loss = torch.from_numpy(np.asarray([1.0-np.sum(res)/len(res)]))
        tensorboard_logs = {'val_loss': avg_loss}
        self.current_val_loss = avg_loss
        return {'avg_val_loss': avg_loss, 'val_loss': avg_loss, 'lg': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    # def configure_optimizers(self):
    #     # REQUIRED
    #     # can return multiple optimizers and learning_rate schedulers
    #     # (LBFGS it is automatically supported, no need for closure function)
    #     return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.dev_loader

    @pl.data_loader
    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # self.opt = RAdam(self.model.parameters(),
        #                  lr=self.lr,
        #                  betas=self.betas,
        #                  eps=self.eps,
        #                  weight_decay=self.weight_decay,
        #                  degenerated_to_sgd=True)

        # self.linear_warmup = \
        #     get_linear_warmup_scheduler(self.opt,
        #                                 num_warmup_steps=self.warmup_steps)
        # self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.opt,
        #     mode='min',
        #     factor=0.1,
        #     patience=5,
        #     verbose=True,
        #     cooldown=5,
        #     min_lr=1e-8,
        # )
        return [self.opt]
        #return [self.opt], [self.linear_warmup, self.reduce_lr_on_plateau]

    def optimizer_step(self, epoch_nb, batch_nb, optimizer,
                       optimizer_i, second_order_closure=None):
        self.opt.step()
        #self.opt.zero_grad()
        #self.linear_warmup.step()
        #if self.trainer.global_step % self.val_check_interval == 0:
        #    self.reduce_lr_on_plateau.step(self.current_val_loss)

    def printgradnorm(self, grad_input, grad_output, fml):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())