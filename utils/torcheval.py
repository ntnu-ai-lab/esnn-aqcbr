from dataset.makeTrainingData import makeDualSharedArchData, makeGabelTrainingData

from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import os
os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
from dataset.dataset import Dataset
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.training_io import TrainerIOMixin
import torch
import os
import time
from sklearn.metrics import matthews_corrcoef

def runevaler(datasetname, epochs, model, torchevaler, evalfunc,
              networklayers=[13, 13], lr=0.02, dropoutrate=0.5,
              validate_on_k=10, filenamepostfix=""):
    #print("evaling model: ")
    #model.summarize(model,mode="full")
    device = "cpu"
    gpus = None
    if torch.cuda.is_available():
        gpus = [0]
        device = "cuda:0"
    d = Dataset(datasetname)
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
    data = dsl.getFeatures()
    target = dsl.getTargets()
    train, test = next(stratified_fold_generator)

    test_data = data[test]
    test_target = target[test]

    train_data = data[train]
    train_target = target[train]
    datasetinfo = dsl.dataset.datasetInfo
    if "augmentTrainingData" in datasetinfo:
        func = datasetinfo["augmentTrainingData"]
        train_data, train_target = func(train_data, train_target)
        test_data, test_target = func(test_data, test_target)
    batchsize = test_data.shape[0]*train_data.shape[0]

    sys = model(None, train_data, train_target, validation_func=evalfunc,
                train_data=train_data, train_target=train_target,
                test_data=test_data, test_target=test_target,
                colmap=colmap, device=device,
                networklayers=networklayers, lr=lr,
                dropoutrate=dropoutrate)
    sys.cuda(0)
    rootdir = os.getcwd()
    filename = rootdir+"train-"+filenamepostfix+str(os.getpid())
    checkpoint_callback = ModelCheckpoint(
        filepath=filename,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    evaler = torchevaler(dsl, train,
                         modelcheckpoint=checkpoint_callback,
                         validate_on_k=validate_on_k,
                         evalfunc=evalfunc,train_data=train_data,
                         train_target=train_target, test_data=test_data,
                         test_target=test_target, colmap=colmap)
    t0 = time.time()
    torchmodels, \
        jitmodels = evaler.myeval(sys, epochs,
                                  sys.opt, sys.loss, device)
    t1 = time.time()
    diff = t1-t0
    print(f"spent {diff} time training {diff/epochs} per epoch")
    res, errdistvec, truedistvec, \
        combineddata, pred_vec = evalfunc(sys, test_data, test_target, train_data,
                                          train_target, batch_size=batchsize,
                                          anynominal=False, colmap=colmap,
                                          device=device)

    ires = np.asarray(res).astype(int)
    ires = ires-np.ones(ires.shape)
    ires = ires*-1
    ires[ires < 0.1] = -1
    ipred = np.asarray(pred_vec).astype(int)
    ipred[ipred > 0.499] = 1
    ipred[ipred <= 0.499] = -1
    print(f"{np.sum(res)/len(res)}")
    print(f"errdistvec: {errdistvec}")
    print(f"truedistvec: {truedistvec}")
    print(f"MCC: {matthews_corrcoef(ires,ipred)}")
    return sys, test_data, evaler


class TorchEvaler():
    def __init__(self, sklearndataset, trainidx,
                 transform=None, should_invert=True,
                 modelcheckpoint=None, validate_on_k=10,
                 evalfunc=None, train_data=None,
                 train_target=None, test_data=None, test_target=None,
                 colmap=None):

        self.sklearandataset = sklearndataset
        self.loadData(sklearndataset, trainidx)
        self.transform = transform
        self.should_invert = should_invert
        self.featurelength = sklearndataset.featurecolsto - \
            sklearndataset.featurecolsfrom
        self.x1start = 0
        self.evalfunc = evalfunc
        self.x2start = self.featurelength+1
        self.modelcheckpoint = modelcheckpoint
        self.x1stop = self.featurelength+1
        self.x2stop = 2*(self.featurelength+1)
        self.x1s = torch.from_numpy(self.data[:, self.x1start:self.x1stop].astype('float')).to(torch.float32).to('cuda:0')
        self.x2s = torch.from_numpy(self.data[:, self.x2start:self.x2stop].astype('float')).to(
            torch.float32).to('cuda:0')
        self.y1 = torch.from_numpy(self.y1).to(torch.float32).to('cuda:0')
        self.y2 = torch.from_numpy(self.y2).to(torch.float32).to('cuda:0')
        self.labels = torch.from_numpy(self.targets[:].astype('float')).to(torch.float32).to('cuda:0')
        # set the path for the callbacks
        self.trainerio = TrainerIOMixin()
        self.modelcheckpoint.save_function = self.trainerio.save_checkpoint
        self.trainerio.optimizers = []
        self.trainerio.lr_schedulers = []
        self.validate_on_k = validate_on_k
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self.colmap = colmap



    def evalepoch(self, model, criterion, optim):
        """
        This code is for evaluating ESNN and computing the loss..
        """
        y_hat, inner_output1, inner_output2 = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels, self.y1,
                         self.y2, inner_output1, inner_output2)
        return loss


    def myeval(self, model, epochs, optim, criterion, device):
        batch_size = self.train_data.shape[0]*self.test_data.shape[0]
        self.trainerio.model = model
        lastbest = 1
        torchmodels = []
        jitmodels = []
        for epoch in range(epochs):
            #idx = torch.randperm(self.x1s.nelement())
            optim.zero_grad()
            loss = self.evalepoch(model, criterion, optim)
            loss.backward()
            optim.step()
            self.trainerio.current_epoch = epoch
            self.trainerio.global_step = epoch
            if (epoch+1)%self.validate_on_k == 0:
                res, errdistvec, truedistvec, \
                    combineddata, pred_vec = self.evalfunc(model,
                                                           self.test_data,
                                                           self.test_target,
                                                           self.train_data,
                                                           self.train_target,
                                                           batch_size=batch_size,
                                                           anynominal=False,
                                                           colmap=self.colmap,
                                                           device=device)
                val_loss = 1.0-np.sum(res)/len(res)
                if val_loss < lastbest:
                    old_loss = lastbest
                    lastbest = val_loss

                    from collections import OrderedDict
                    import pickle
                    copyed_model = pickle.loads(pickle.dumps(model))
                    best_model_state_dict = {k: v.to('cpu') for k, v in copyed_model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)
                    copyed_model.load_state_dict(best_model_state_dict)
                    smod = torch.jit.script(copyed_model)
                    smod.eval()
                    smod = smod.cpu()
                    filepath = self.modelcheckpoint.filepath+".pt1"
                    print(f" Epoch [{epoch}/{epochs}] saving java model to {filepath} new {val_loss} old {old_loss}")
                    smod.save(filepath)
                    jitmodels.append(smod)
                tensorboard_logs = {'val_loss': val_loss}
                metrics = {'val_loss': val_loss, 'lg': tensorboard_logs}
                self.modelcheckpoint.on_epoch_end(epoch, metrics)

            #print('Epoch [%d/%d], loss: %.4f,'
            #      % (epoch, epochs, loss.data))

    def loadData(self, sklearndataset, trainidx):
        self.data, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeDualSharedArchData(sklearndataset.getFeatures()[trainidx],
                                       sklearndataset.getTargets()[trainidx], False)




class ChopraTorchEvaler(TorchEvaler):
    def __init__(self, *args, **kwargs):
        super(ChopraTorchEvaler, self).__init__(*args, **kwargs)
        self.labels = self.labels.squeeze()

    def evalepoch(self,  model, criterion, optim):
        """
        This code is for evaluating Chopra and computing the loss..
        """
        y_hat = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels)

        return loss

class GabelTorchEvaler(TorchEvaler):
    def loadData(self, sklearndataset, trainidx):
        self.data, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeGabelTrainingData(sklearndataset.getFeatures()[trainidx],
                                      sklearndataset.getTargets()[trainidx], False)

    def __init__(self, *args, **kwargs):
        super(GabelTorchEvaler, self).__init__(*args, **kwargs)


    def evalepoch(self,  model, criterion, optim):
        """
        This code is for evaluating GAbel and computing the loss..
        """
        # inp1 = torch.cat([self.x1s, self.x2s], dim=0).squeeze()
        # inp2 = torch.cat([self.x2s, self.x1s], dim=0).squeeze()
        # ll = torch.cat([self.labels, self.labels], dim=0)
        # llones = torch.ones(ll.shape).to(torch.float32).to("cuda:0")
        # ll = ll-llones
        # ll = torch.pow(ll,2)
        # y_hat = model(inp1, inp2)
        # loss = criterion(y_hat, ll)
        y_hat = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels)

        return loss
