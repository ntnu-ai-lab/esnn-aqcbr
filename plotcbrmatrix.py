
from utils.cbrplotting import makematrixdata, plot2heatmap
import sys
import argparse
import pandas as pd
from utils.pdutils import stat
from utils.plotting import setLateXFonts
from utils.cbrplotting import makematrixdata, plot2heatmap
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from models.esnn.pytorch_trainer import ESNNSystem, ESNNModel
from models.type3.chopra import ChopraTrainer, ChopraModel
from models.type2.gabel import GabelTrainer, GabelModel
import torch
from dataset.dataset import Dataset
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import numpy as np
import random
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='plot stats')
    parser.add_argument('--modelpath', metavar='modelpath', type=str,
                        help='MODELPATH')
    parser.add_argument('--modeltype', metavar='modeltype', type=str,
                        help='modeltype')
    parser.add_argument('--prefix', metavar='prefix', type=str,
                        help='prefix')
    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='dataset')
    parser.add_argument('--seed', metavar='seec', type=int,
                        help='seed')
    args = parser.parse_args()
    if args.seed is not None:
        print(f"setting seed to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(int(args.seed/10000000000))
    else:
        print(f"torch seed: {torch.initial_seed()}")
    device = "cpu"
    gpus = None

    prefix = args.prefix
    if torch.cuda.is_available():
        gpus = [0]
        device = "cuda:0"
    d = Dataset(args.dataset)
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d,
                                                                  True,
                                                                  n_splits=5)
    train, test = next(stratified_fold_generator)
    data = dsl.getFeatures()
    target = dsl.getTargets()
#print the data dist..
    #pairplot(dsl,
    #         filenamepostfix+"pairplot", 10)

    datasetinfo = dsl.dataset.datasetInfo
    if "augmentTrainingData" in datasetinfo:
        func = datasetinfo["augmentTrainingData"]
        data, target = func(data, target)
        dsl.setData(np.concatenate((data, target), axis=1))
        #train_data, train_target = func(train_data, train_target)
        # test_data, test_target = func(test_data, test_target)

    data = dsl.getFeatures()
    target = dsl.getTargets()

    loadedstate = torch.load(args.modelpath)


    if "esnn" in args.modeltype:
        #model = ESNNModel(X=data,Y=target)
        model = ESNNSystem(X=data,Y=target, networklayers=[[40, 6 , 4], [40, 6, 3]])

    elif "chopra" in args.modeltype:
        model = ChopraTrainer(X=data,Y=target, networklayers=[40, 6, 3])
    elif "gabel" in args.modeltype:
        model = GabelTrainer(X=data,Y=target, networklayers=[40, 6, 3])
    #loadedstate = torch.load(args.modelpath)
    model.load_state_dict(loadedstate['state_dict'])
    model = model.to('cuda:0')
    if "esnn" in args.modeltype:
        df = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=0)
    elif "chopra" in args.modeltype:
        df = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=2)
    elif "gabel" in args.modeltype:
        df = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=1)
    df.to_csv(prefix+"-matrixdata.csv")
    setLateXFonts()
    plot2heatmap(df, 10, annot=True, outputfile=prefix+"-matrix.pdf")
    sys.exit(0)
