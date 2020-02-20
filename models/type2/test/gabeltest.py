from models.tests.test_base import GabelTestBase
"""
This test suite tests that the code actually trains a model,
from high training loss to low training loss
"""


class GabelIrisTest(GabelTestBase):
    def testTorchGabelIRIS(self):
        self.method_for_eval("iris", 10)

    def testTorchGabelUSE(self):
        self.method_for_eval("use", 100)

    def testTorchGabelOPSITU(self):
        self.method_for_eval("opsitu", 500)
