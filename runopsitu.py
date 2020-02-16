from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
from dataset.dataset import Dataset
from utils.torch import TorchSKDataset
from pytorch_lightning import Trainer
from models.eval import eval_dual_ann
from torch.utils.data import DataLoader
import numpy as np
from models.esnn.tests.pytorch_esnn_test import ESNNTest

def testTorchESNNOPSITU():
    esnnTest = ESNNTest()
    esnnTest.method_for_eval("iris", 200, 10,
                             "filename-iris-test",
                             show_progress_bar=True)


if __name__ == "__main__":
    testTorchESNNOPSITU()
