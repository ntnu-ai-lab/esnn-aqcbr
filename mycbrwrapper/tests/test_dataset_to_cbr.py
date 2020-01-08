from mycbrwrapper.concepts import Concepts
from dataset.dataset_to_cbr import fromDatasetToCBR
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import unittest

__name__ = "test_base"

defaulthost = "localhost:8080"
"""
The model of the case base for the unit tests are simple
id,name,doubleattr1,doubleattr2
"""


class ConvertDataSetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("in setupclass")

    @classmethod
    def tearDownClass(cls):
        print("in tearDownClass")

    def __init__(self, *args, **kwargs):
        super(ConvertDataSetTest, self).__init__(*args, **kwargs)

    @staticmethod
    def convert(name):
        c = Concepts(defaulthost)
        # conceptstring = "test_concept_test1"
        #c.addConcept(conceptstring)
        d = Dataset(name)
        dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True)
        concepts, concept = fromDatasetToCBR(d, dsl, colmap,
                                             host=defaulthost,
                                             concepts=c, instances=100)
        # concept = concepts.getConcept(name)
        return d, dsl, colmap, stratified_fold_generator, concepts, concept

    def testconvertopsitu(self):
        ConvertDataSetTest.convert("opsitu")
