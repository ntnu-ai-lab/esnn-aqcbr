from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.type3.chopra import ChopraTrainer,ChopraModel
import tensorflow as tf
import os
import unittest
import random
from dataset.dataset import Dataset
from utils.torch import TorchSKDataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import numpy as np
from models.eval import eval_dual_ann
from sklearn.metrics import matthews_corrcoef

"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class ChopraIrisTest(unittest.TestCase):

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

    def testTorchSKDataset(self):
        d = Dataset("iris")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        torchskdataset = TorchSKDataset(dsl)

    def testTorchChopraIRIS(self):
        d = Dataset("iris")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        torchskdataset = TorchSKDataset(dsl)
        data = dsl.getFeatures()
        target = dsl.getTargets()
        train, test = next(stratified_fold_generator)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]
        dl = DataLoader(torchskdataset, batch_size=32)
        sys = ChopraTrainer(dl, train_data, train_target)
        trainer = Trainer(max_epochs=200)
        trainer.fit(sys)
        res, errdistvec, truedistvec, \
            combineddata, pred_vec = eval_dual_ann(sys, data[test], target[test], data[train],
                                         target[train], batch_size=32,
                                         anynominal=False, colmap=colmap)
        print(f"res: {res}")
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


    def testTorchChopraUSE(self):
        d = Dataset("use")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        torchskdataset = TorchSKDataset(dsl)
        data = dsl.getFeatures()
        target = dsl.getTargets()
        train, test = next(stratified_fold_generator)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]
        dl = DataLoader(torchskdataset, batch_size=train.shape[0])
        sys = ChopraTrainer(dl, train_data, train_target)
        trainer = Trainer(min_epochs=1000, max_epochs=1000)
        trainer.fit(sys)
        res, errdistvec, truedistvec, \
            combineddata = eval_dual_ann(sys, data[test], target[test], data[train],
                                         target[train], batch_size=train_data.shape[0],
                                         anynominal=False, colmap=colmap)
        print(f"res: {res}")
        print(f"{np.sum(res)/len(res)}")
        print(f"errdistvec: {errdistvec}")
        print(f"truedistvec: {truedistvec}")
        #def testTrainChopraAdam(self):
        #model = ChopraModel()

    def testTorchChopraOPSITU(self):
        d = Dataset("opsitu")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        torchskdataset = TorchSKDataset(dsl)
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
            #test_data, test_target = func(test_data, test_target)
        dl = DataLoader(torchskdataset, batch_size=train.shape[0])
        sys = ChopraTrainer(dl, train_data, train_target)
        trainer = Trainer(min_epochs=1000, max_epochs=1000)
        trainer.fit(sys)
        res, errdistvec, truedistvec, \
        combineddata = eval_dual_ann(sys, test_data, test_target, train_data,
                                     train_target, batch_size=train_data.shape[0],
                                     anynominal=False, colmap=colmap)
        print(f"res: {res}")
        print(f"{np.sum(res) / len(res)}")
        print(f"errdistvec: {errdistvec}")
        print(f"truedistvec: {truedistvec}")
        # def testTrainChopraAdam(self):
        # model = ChopraModel()
