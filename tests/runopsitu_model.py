from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn import keras

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import unittest
import random
import numpy as np

"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class run_opsitu(unittest.TestCase):

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
    def readmodel(jsonfile, h5file):
        json_file = open(jsonfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(h5file)
        print("Loaded model from disk")
        return loaded_model

    def testTrainESNN(self):
        random.seed(42)
        jsonfile = "esnn.json"
        h5file = "esnn.h5"
        model = run_opsitu.readmodel(jsonfile, h5file)

        jsonfile_500 = "esnn166.json"
        h5file_500 = "esnn166.h5"
        model_500 = run_opsitu.readmodel(jsonfile_500, h5file_500)

        a1 = [0.9,0.9,0.9,0.9]
        a2 =[0.1,0.1,0.1,0.1]
        ret = model.predict([[a1], [a2]])
        print(f"ret: :{ret}")
        assert ret[0] < 1  # as of 09.01.20 1954de71e8fc800e81a99d7a9e06750bcd214dad

        a1 = [0.5, 0.45, 0.45, 0.5]
        a2 = [0.5, 0.5, 0.5, 0.7]
        ret = model.predict([[a1], [a2]])
        print(f"ret2: :{ret}")
        assert ret[0] < 1

        a1 = [0.5, 0.5, 0.5, 0.7]
        a2 = [0.5, 0.5, 0.5, 0.7]
        ret = model.predict([[a1], [a2]])
        ret4 = model.predict([[a1], [a1]])
        print(f"ret3: :{ret}")
        print(f"ret4: :{ret4}")

        a1 = [0.01, 0.01, 0.01, 0.07]
        a2 = [0.9, 0.9, 0.9, 0.9]
        ret_500 = model_500.predict([[a1], [a2]])
        ret4_500 = model_500.predict([[a1], [a1]])
        print(f"ret3: :{ret_500}")
        print(f"ret4: :{ret4_500}")

