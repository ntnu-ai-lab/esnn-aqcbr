from models.tests.test_base import MethodTestBase
"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class ESNNIrisTest(MethodTestBase):

    @classmethod
    def setupClass(cls):
        print("in esnn iris test")

    @classmethod
    def tearDownClass(cls):
        print("in esnn iris test")

    def testTorchESNNIRIS(self):
        self.method_for_eval("iris", 200, 10)

    def testTorchESNNUSE(self):
        self.method_for_eval("use", 200, 10)

    def testTorchESNNOPSITU(self):
        self.method_for_eval("opsitu", 200, 10)
