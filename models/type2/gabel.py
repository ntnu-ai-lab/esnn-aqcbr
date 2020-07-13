import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from utils.my_torch_utils import ContrastiveLoss, RAdam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import RMSprop
import numpy as np
from argparse import Namespace

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
class GabelTrainer(pl.LightningModule):
    def __init__(self, data, X, Y, networklayers=[13, 13],
                 lr=1e-4, betas=(0.9, 0.999), eps=1e-07,
                 weight_decay=1e-7, warmup_steps=100,
                 val_check_interval=250,
                 val_percent_check=0.3,
                 validation_func=None,
                 train_data=None, train_target=None,
                 test_data=None, test_target=None,
                 colmap=None, device=None,
                 dropoutrate=0.05):
        super(GabelTrainer, self).__init__()
        self.inputwidth = X.shape[1]
        self.outputwidth = Y.shape[1]
        self.hparams = {'lr':lr, 'networklayers': networklayers,
                        'dropoutrate':dropoutrate, 'inputwidth': self.inputwidth,
                        'outputwidth': self.outputwidth}
        self.hparams = Namespace(**self.hparams)
        # not the best model...
        self.model = GabelModel(self.inputwidth, self.outputwidth,
                                networklayers, dropoutrate).to(torch.float32)
        self.loss = torch.nn.BCELoss()
        #self.esnnloss = ESNNloss()
        self.train_loader = data
        self.dev_loader = data
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.opt = RMSprop(self.model.parameters(), lr=0.2)
        self.test_loader = data
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.val_check_interval = val_check_interval
        self.val_percent_check = val_percent_check
        self.colmap = colmap
        self.device = device

    def forward(self, input1, input2):
        return self.model.forward(input1, input2)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x1, x2, y1, y2, label = batch
        y_hat = self.forward(x1, x2)
        loss = self.loss.forward(y_hat, label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x1, x2, y1, y2, label = batch
        y_hat = self.forward(x1, x2)
        return {'val_loss': self.loss.forward(y_hat, label)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.current_val_loss = avg_loss
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

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

def torch_euc_dist(t1, t2):
    epsilon = torch.from_numpy(np.asarray([0.001])).to(torch.float32)
    tmp = torch.sum(torch.pow(t1-t2, 2), axis=1)
    return torch.sqrt(torch.max(tmp, epsilon))

class GabelModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape, networklayers=[13, 13], dropoutrate=0.05):
        """

        """
        super(GabelModel, self).__init__()
        input_shape = input_shape*2
        layers = networklayers
        if isinstance(networklayers[0], list):
            layers = networklayers[0]

        self.L = torch.nn.ModuleList()
        for networklayer in layers:
            self.L.append(torch.nn.Linear(in_features=input_shape,
                                          out_features=networklayer))
            self.L.append(torch.nn.Dropout(dropoutrate))
            input_shape = networklayer

        self.last = torch.nn.Linear(in_features=input_shape,
                                    out_features=1)
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()

    def forward_stack(self, x):
        y = x
        for layer in self.L:
            y = self.sigm(layer(y))
        return y

    def forward_all(self, x):
        y = self.forward_stack(x)
        return self.sigm(self.last(y))

    def forward(self, input1, input2):
        cated = torch.cat([input1, input2], 1)
        return self.forward_all(cated)
