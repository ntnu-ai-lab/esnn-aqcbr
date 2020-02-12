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
from utils.torch import TorchSKDataset\
    #, _pytorch2keras
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import numpy as np
from models.eval import eval_dual_ann

"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class ESNNIrisTestExport(unittest.TestCase):

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

    def testTorchESNNIRISExport(self):
        d = Dataset("iris")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        df = dsl.df
        df3 = df.idxmax(1)
        df2 = df[["weatherhindrance_1.0", "weatherhindrance_0.0"]]
        df2.to_csv("heh.csv")
        # torchskdataset = TorchSKDataset(dsl)
        # data = dsl.getFeatures()
        # target = dsl.getTargets()
        # train, test = next(stratified_fold_generator)
        # test_data = data[test]
        # test_target = target[test]
        # exinp1, exinp2, label = torchskdataset.getItem(0)
        #
        # train_data = data[train]
        # train_target = target[train]
        # dl = DataLoader(torchskdataset, batch_size=32)
        # model = ESNNSystem(dl, train_data, train_target)
        # trainer = Trainer(max_epochs=10)
        # trainer.fit(model)
        # res, errdistvec, truedistvec, \
        #     combineddata = eval_dual_ann(model, data[test], target[test], data[train],
        #                                  target[train], batch_size=32,
        #                                  anynominal=False, colmap=colmap)
        # print(f"res: {res}")
        # print(f"{np.sum(res)/len(res)}")
        # print(f"errdistvec: {errdistvec}")
        # print(f"truedistvec: {truedistvec}")
        # print("exporting")
        # #torch_to_keras(model, ['input1', 'input2'],
        # #               ['output1', 'output2', 'output3'],
        # #             test_data[0, :], "testoutput")
        # test_input = test_data[0,:]
        # _pytorch2keras(model, [exinp1, exinp2], [exinp1.shape, exinp2.shape])

    def testTorchESNNOpsituExport(self):
        d = Dataset("opsitu")
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        df = dsl.df
        df3 = df.idxmax(1)
        df2 = df[["weatherhindrance_1.0", "weatherhindrance_0.0"]]
        df["class"] = df2.idxmax(1)
        #df2 = df2.drop("weatherhindrance_1.0",axis=1)
        #df.to_csv("heh.csv")
        dfu = dsl.unhotify()
        dfu.to_csv("heh.csv")

        # torchskdataset = TorchSKDataset(dsl)
        # data = dsl.getFeatures()
        # target = dsl.getTargets()
        # train, test = next(stratified_fold_generator)
        # test_data = data[test]
        # test_target = target[test]
        # exinp1, exinp2, label = torchskdataset.getItem(0)
        #
        # train_data = data[train]
        # train_target = target[train]
        # dl = DataLoader(torchskdataset, batch_size=32)
        # model = ESNNSystem(dl, train_data, train_target)
        # trainer = Trainer(max_epochs=10)
        # trainer.fit(model)
        # res, errdistvec, truedistvec, \
        #     combineddata = eval_dual_ann(model, data[test], target[test], data[train],
        #                                  target[train], batch_size=32,
        #                                  anynominal=False, colmap=colmap)
        # print(f"res: {res}")
        # print(f"{np.sum(res)/len(res)}")
        # print(f"errdistvec: {errdistvec}")
        # print(f"truedistvec: {truedistvec}")
        # print("exporting")
        # #torch_to_keras(model, ['input1', 'input2'],
        # #               ['output1', 'output2', 'output3'],
        # #             test_data[0, :], "testoutput")
        # test_input = test_data[0,:]
        # _pytorch2keras(model, [exinp1, exinp2], [exinp1.shape, exinp2.shape])
