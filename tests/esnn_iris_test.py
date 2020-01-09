from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn import esnn
import os
import unittest

"""
The model of the case base for the unit tests are simple
id,name,doubleattr1,doubleattr2
"""


class ESNN_IRIS_TEST(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        print("in esnn iris test")

    @classmethod
    def tearDownClass(cls):
        print("in esnn iris test")

    @staticmethod
    def makemodel(dataset, epochs, rootpath):
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
                                   optimizer=optimizer_dict["rprop"],
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
        rootpath = "~/research/experiments/annSimilarity/mycbrwrapper/tests/"
        rootpath = os.path.expanduser(rootpath)
        ESNN_IRIS_TEST.makemodel("iris", 10, rootpath)
