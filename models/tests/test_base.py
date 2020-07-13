from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import time
from models.type3.chopra import ChopraTrainer
import tensorflow as tf
import os
import unittest
import random
from dataset.dataset import Dataset
from utils.my_torch_utils import GabelTorchDataset, TorchSKDataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import numpy as np
from models.evalfunctions import eval_gabel_ann, eval_dual_ann
from sklearn.metrics import matthews_corrcoef
from models.type2.gabel import GabelTrainer
from models.esnn.pytorch_trainer import ESNNSystem
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


class MethodTestBase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MethodTestBase, self).__init__(*args, **kwargs)
        self.torchdataset = TorchSKDataset
        self.evalfunc = eval_dual_ann
        self.model = ESNNSystem

    @classmethod
    def setupClass(cls):
        print("in esnn iris test")

    @classmethod
    def tearDownClass(cls):
        print("in esnn iris test")

    @staticmethod
    def setSeed(seed):
        os.environ['PYTHONHASHseed'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def load(self, model, device, checkpoint_path):
        """

        :param model:
        :param device: e.g. device = torch.device('cuda')
        :param checkpoint_path: e.g. _ckpt_epoch_899.ckpt
        :return:
        """
        model.load_state_dict(torch.load(
        checkpoint_path, map_location=device), strict=False)

    def method_for_eval(self, datasetname, epochs, valepochs,
                        rootdir=None, filename="",
                        show_progress_bar=False):
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
        torchskdataset = self.torchdataset(dsl,train)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]
        datasetinfo = dsl.dataset.datasetInfo
        if "augmentTrainingData" in datasetinfo:
            func = datasetinfo["augmentTrainingData"]
            train_data, train_target = func(train_data, train_target)
            test_data, test_target = func(test_data, test_target)

        print(f"datasetsize {len(torchskdataset)}")
        dl = DataLoader(torchskdataset, batch_size=len(torchskdataset), num_workers=1)
        sys = self.model(dl, train_data, train_target, validation_func=self.evalfunc,
                         train_data=train_data, train_target=train_target,
                         test_data=test_data, test_target=test_target,
                         colmap=colmap, device=device)
        if rootdir is None:
            rootdir = os.getcwd()
        filename = rootdir+"train-"+filename+str(os.getpid())
        checkpoint_callback = ModelCheckpoint(
            filepath=filename,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        trainer = Trainer(min_epochs=epochs,
                          max_epochs=epochs,
                          show_progress_bar=show_progress_bar,
                          check_val_every_n_epoch=valepochs,
                          gpus=gpus,
                          checkpoint_callback=checkpoint_callback)
        t0 = time.time()
        trainer.fit(sys)
        t1 = time.time()
        diff = t1-t0
        print(f"spent {diff} time training {diff/epochs} per epoch")
        res, errdistvec, truedistvec, \
            combineddata, pred_vec = self.evalfunc(sys, test_data, test_target, train_data,
                                                   train_target, batch_size=train_data.shape[0]*test_data.shape[0],
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

class GabelTestBase(MethodTestBase):
    def __init__(self, *args, **kwargs):
        super(GabelTestBase, self).__init__(*args, **kwargs)
        self.torchdataset = GabelTorchDataset
        self.evalfunc = eval_gabel_ann
        self.model = GabelTrainer
