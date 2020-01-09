from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn import esnn
import tensorflow as tf
import os
import unittest
import random

"""
This test suite tests that the code actually trains a model, 
from high training loss to low training loss
"""


class ESNN_IRIS_TEST(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        print("in esnn iris test")

    @classmethod
    def tearDownClass(cls):
        print("in esnn iris test")

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
            embedding_model = esnn(o_X=test_data, o_Y=test_target,
                                   X=train_data, Y=train_target,
                                   regression=dsl.isregression,
                                   shuffle=True,
                                   batch_size=200,
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
        return model, embedding_model, hist

    def testTrainESNN(self):
        random.seed(42)
        rootpath = "~/research/experiments/annSimilarity/mycbrwrapper/tests/"
        print(f" training eagerly : {tf.executing_eagerly()}")
        rootpath = os.path.expanduser(rootpath)
        model, embedding_model, hist = \
            ESNN_IRIS_TEST.makemodel("iris", 200, rootpath)
        loss = hist.history["loss"]
        firstloss = loss[0]
        lastloss = loss[len(loss)-1]
        print(f"RPROP firstloss: {firstloss} lastloss: {lastloss}")
        assert firstloss > lastloss


    def testTrainESNNAdam(self):
        random.seed(42)
        rootpath = "~/research/experiments/annSimilarity/mycbrwrapper/tests/"
        print(f" training eagerly : {tf.executing_eagerly()}")
        rootpath = os.path.expanduser(rootpath)
        model, embedding_model, hist = \
            ESNN_IRIS_TEST.makemodel("iris", 200, rootpath, "adam")
        loss = hist.history["loss"]
        firstloss = loss[0]
        lastloss = loss[len(loss)-1]
        print(f"ADAM firstloss: {firstloss} lastloss: {lastloss}")
        assert firstloss > lastloss
        
