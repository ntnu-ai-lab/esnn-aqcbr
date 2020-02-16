from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import os

from models.type3.chopra import ChopraTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
from dataset.dataset import Dataset
from utils.torch import TorchSKDataset
from pytorch_lightning import Trainer
from models.eval import eval_dual_ann
from torch.utils.data import DataLoader
import numpy as np
from models.esnn.tests.pytorch_esnn_test import ESNNTest
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.torcheval import TorchEvaler, runevaler, ChopraTorchEvaler
import torch
import os
import time
from sklearn.metrics import matthews_corrcoef

def testTorchESNNOPSITU(datasetname, epochs):
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
    evalfunc = eval_dual_ann
    sys = ESNNSystem(None, train_data, train_target, validation_func=evalfunc,
                     train_data=train_data, train_target=train_target,
                     test_data=test_data, test_target=test_target,
                     colmap=colmap, device=device,
                     networklayers=[50, 50, 50], lr=.02,
                     dropoutrate=0.5)
    sys = ESNNSystem(None, train_data, train_target, validation_func=evalfunc,
                     train_data=train_data, train_target=train_target,
                     test_data=test_data, test_target=test_target,
                     colmap=colmap, device=device,
                     networklayers=[50, 50, 50], lr=0.02,
                     dropoutrate=0.05)
    sys.cuda(0)
    filename = "heh"
    rootdir = os.getcwd()
    filename = rootdir+"train-"+filename+str(os.getpid())
    checkpoint_callback = ModelCheckpoint(
        filepath=filename,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    evaler = TorchEvaler(dsl, train,
                         modelcheckpoint=checkpoint_callback,
                         validate_on_k=10,
                         evalfunc=evalfunc,train_data=train_data,
                         train_target=train_target, test_data=test_data,
                         test_target=test_target, colmap=colmap)
    t0 = time.time()
    evaler.myeval(sys,epochs,sys.opt,sys.loss,device=device)
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

def testTorchESNNOPSITU2(datasetname, epochs, eval_on_k, filename):

    runevaler(datasetname, epochs, ESNNSystem, TorchEvaler,
              eval_dual_ann, networklayers=[40, 40, 40],
              lr=.09, dropoutrate=0.5, eval_on_k=eval_on_k,
              filenamepostfix=filename)


if __name__ == "__main__":
    #testTorchESNNOPSITU2("iris",200)
    #runevaler("iris", 200, ESNNSystem, TorchEvaler, eval_dual_ann, networklayers=[40, 40, 40], lr=.02,
    #          dropoutrate=0.05)
    #runevaler("opsitu", 2000, ESNNSystem, TorchEvaler, eval_dual_ann, networklayers=[40, 40, 40], lr=.02,
    #          dropoutrate=0.05)
    runevaler("opsitu", 2000, ChopraTrainer, ChopraTorchEvaler,
              eval_dual_ann, networklayers=[40, 40, 40],
              lr=.05, dropoutrate=0.05, validate_on_k=10,
              filenamepostfix="chopra")
    #testTorchESNNOPSITU("iris",200)
