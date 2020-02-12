from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn import keras
import tensorflow as tf
import os
import unittest
import random
import numpy as np

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

    @staticmethod
    def makemodel(dataset, epochs, rootpath, optimizer="rprop"):
        d = Dataset(dataset)
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
        data = dsl.getFeatures()
        target = dsl.getTargets()
        train, test = next(stratified_fold_generator)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]

        model, hist, \
            ret_callbacks,\
            embedding_model = keras(o_X=test_data, o_Y=test_target,
                                    X=train_data, Y=train_target,
                                    regression=dsl.isregression,
                                    shuffle=True,
                                    batch_size=1000,
                                    epochs=epochs,
                                    optimizer=optimizer_dict[optimizer],
                                    onehot=True,
                                    multigpu=False,
                                    callbacks=[],
                                    datasetname=dataset,
                                    networklayers=[13, 13],
                                    rootdir=rootpath,
                                    alpha=0.2,
                                    makeTrainingData=makeDualSharedArchData)
        return model, embedding_model, hist, test_data

    def testTrainESNN(self):
        random.seed(42)
        rootpath = "/tmp"
        rootpath = os.path.expanduser(rootpath)
        model, embedding_model, hist, test_data = \
            ESNNIrisTest.makemodel("iris", 200, rootpath)
        loss = hist.history["loss"]
        firstloss = loss[0]
        lastloss = loss[len(loss)-1]
        print(f"RPROP firstloss: {firstloss} lastloss: {lastloss}")
        assert firstloss > lastloss
        # assert lastloss < 0.4  # as of 09.01.20 1954de71e8fc800e81a99d7a9e06750bcd214dad
        model.predict(test_data[0])


    def testTrainESNNAdam(self):
        random.seed(42)
        rootpath = "/tmp"
        rootpath = os.path.expanduser(rootpath)
        model, embedding_model, hist, test_data = \
            ESNNIrisTest.makemodel("iris", 200, rootpath, "adam")
        loss = hist.history["loss"]
        firstloss = loss[0]
        lastloss = loss[len(loss)-1]
        print(f"ADAM firstloss: {firstloss} lastloss: {lastloss}")
        assert firstloss > lastloss
        # assert lastloss < 0.6 # as of 09.01.20 1954de71e8fc800e81a99d7a9e06750bcd214dad

