import time
import numpy as np
from models.evalfunctions import eval_dual_ann
from sklearn.metrics import matthews_corrcoef
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from mycbrwrapper.rest import getRequest
from mycbrwrapper.tests.test_dataset_to_cbr import ConvertDataSetTest, defaulthost
from models.esnn.keras import esnn
from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
import logging
import os
import hashlib
import warnings

defaulthost = "localhost:8080"
"""
The model of the case base for the unit tests are simple
id,name,doubleattr1,doubleattr2
"""


class NeuralSimTest(ConvertDataSetTest):

    @classmethod
    def setUpClass(cls):
        print("nnnnn")
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings(action="ignore", message="unclosed",
                                category=ResourceWarning)
        # s.config['keep_alive'] = False

    @classmethod
    def tearDownClass(cls):
        print("in tearDownClass")

    @staticmethod
    def makemodel(dsl, stratified_fold_generator,
                  datasetname, rootpath, epochs, test_data,
                  test_target, train_data, train_target):

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
                                   callbacks=["tensorboard"],
                                   datasetname=datasetname,
                                   networklayers=[13, 13],
                                   rootdir=rootpath,
                                   alpha=0.2,
                                   makeTrainingData=makeDualSharedArchData)
        return model, embedding_model, hist

    def testIRIS(self):
        d = Dataset("iris")
        epochs = 100
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True)
        data = dsl.getFeatures()
        target = dsl.getTargets()
        train, test = next(stratified_fold_generator)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]
        t0 = time.time()
        model, \
            embedding_model, \
            hist = NeuralSimTest.makemodel(dsl, stratified_fold_generator,
                                           "iris", "/tmp", epochs,
                                           test_data, test_target,
                                           train_data, train_target)
        t1 = time.time()
        diff = t1-t0
        print(f" spent {diff} time training {diff/epochs} per epoch")
        res, errdistvec, truedistvec, \
            combineddata, pred_vec = eval_dual_ann(model, data[test],
                                                   target[test], data[train],
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
