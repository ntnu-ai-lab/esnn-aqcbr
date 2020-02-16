from models.tests.test_base import MethodTestBase
"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class ESNNTest(MethodTestBase):

    def __init__(self,*args, **kwargs):
        super(ESNNTest, self).__init__(*args, **kwargs)

    @classmethod
    def setupClass(cls):
        print("in esnn iris test")

    @classmethod
    def tearDownClass(cls):
        print("in esnn iris test")

    def testTorchESNNIRIS(self):
        self.method_for_eval("iris", 200, 10, "filename-iris-test", show_progress_bar=True)

    def testTorchESNNUSE(self):
        self.method_for_eval("use", 200, 10)

    def testTorchESNNOPSITU(self):
        self.method_for_eval("opsitu", 200, 10, "filename-opsitu-test", show_progress_bar=True)
