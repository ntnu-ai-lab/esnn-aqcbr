from dataset.dataset_to_sklearn import fromDataSetToSKLearn
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
from dataset.dataset import Dataset
from utils.torch import TorchSKDataset
from pytorch_lightning import Trainer
from models.eval import eval_dual_ann
from torch.utils.data import DataLoader
import numpy as np

def testTorchESNNOPSITU():
    d = Dataset("opsitu")
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
    torchskdataset = TorchSKDataset(dsl)
    data = dsl.getFeatures()
    target = dsl.getTargets()
    train, test = next(stratified_fold_generator)
    test_data = data[test]
    test_target = target[test]
    dsl.to_csv("opsitutest.csv")
    train_data = data[train]
    train_target = target[train]
    datasetinfo = dsl.dataset.datasetInfo
    if "augmentTrainingData" in datasetinfo:
        func = datasetinfo["augmentTrainingData"]
        train_data, train_target = func(train_data, train_target)
        #test_data, test_target = func(test_data, test_target)
    dl = DataLoader(torchskdataset, batch_size=train.shape[0])
    sys = ESNNSystem(dl, train_data, train_target)
    trainer = Trainer(min_epochs=1000, max_epochs=1000,gpus=1)
    trainer.fit(sys)
    res, errdistvec, truedistvec, \
    combineddata = eval_dual_ann(sys, test_data, test_target, train_data,
                                 train_target, batch_size=train_data.shape[0],
                                 anynominal=False, colmap=colmap)
    print(f"res: {res}")
    print(f"{np.sum(res) / len(res)}")
    print(f"errdistvec: {errdistvec}")
    print(f"truedistvec: {truedistvec}")


if __name__ == "__main__":
    testTorchESNNOPSITU()
