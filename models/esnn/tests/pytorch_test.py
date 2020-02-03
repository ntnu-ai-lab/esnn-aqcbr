from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
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

"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class ESNNIrisTest(unittest.TestCase):

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

    def testTorchESNNIRIS(self):
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
        sys = ESNNSystem(dl, train_data, train_target)
        trainer = Trainer(max_epochs=200)
        trainer.fit(sys)
        res, errdistvec, truedistvec, \
            combineddata = eval_dual_ann(sys, data[test], target[test], data[train],
                                         target[train], batch_size=32,
                                         anynominal=False, colmap=colmap)
        print(f"res: {res}")
        print(f"{np.sum(res)/len(res)}")
        print(f"errdistvec: {errdistvec}")
        print(f"truedistvec: {truedistvec}")


    def testTorchESNNUSE(self):
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
        dl = DataLoader(torchskdataset, batch_size=32)
        sys = ESNNSystem(dl, train_data, train_target)
        trainer = Trainer(min_epochs=500, max_epochs=500)
        trainer.fit(sys)
        res, errdistvec, truedistvec, \
            combineddata = eval_dual_ann(sys, data[test], target[test], data[train],
                                         target[train], batch_size=32,
                                         anynominal=False, colmap=colmap)
        print(f"res: {res}")
        print(f"{np.sum(res)/len(res)}")
        print(f"errdistvec: {errdistvec}")
        print(f"truedistvec: {truedistvec}")
        #def testTrainESNNAdam(self):
        #model = ESNNModel()
