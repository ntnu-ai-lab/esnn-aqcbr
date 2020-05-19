
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
from utils.runutils import str2bool

def remap(x):
    if '_1.0' in x:
        return 'failure'
    return 'success'

def pairplot(dsl, output, k, font_scale):
    sns.set(font_scale=font_scale)  # crazy big
    labelcols = ["class"]
    wanteddatacols = ["wind speed", "wind from direction", "wind effect"]
    df = dsl.unhotify()
    df = df.rename(columns={'wind_speed':'wind speed', 'wind_from_direction': 'wind from direction', 'wind_effect':'wind effect'})
    # 'weatherhindrance_1.0':'failure', 'weatherhindrance_0.0':'success'})

    df['class'] = df['class'].apply(lambda x: remap(x))
    with sns.plotting_context(font_scale=1, rc={'text.usetex': True}):
        ax = sns.pairplot(df, vars = wanteddatacols, hue="class")

        ax.savefig(f"{output}.pdf", format="pdf", bbox_inches='tight')
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot stats')
    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='dataset')
    parser.add_argument('--prefix', metavar='prefix', type=str,
                        help='prefix')
    parser.add_argument('--fontscale', metavar='fontscale', type=float,
                        help='fontscale')
    parser.add_argument('--augment', metavar='augment', type=str2bool,
                        help='augment')
    args = parser.parse_args()

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

    datasetinfo = dsl.dataset.datasetInfo
    if "augmentTrainingData" in datasetinfo and args.augment:
        func = datasetinfo["augmentTrainingData"]
        data, target = func(data, target)
        dsl.setData(np.concatenate((data, target), axis=1))
        #train_data, train_target = func(train_data, train_target)
        # test_data, test_target = func(test_data, test_target)

    data = dsl.getFeatures()
    target = dsl.getTargets()
    setLateXFonts()
    pairplot(dsl,
             prefix+"pairplot", 10, args.fontscale)

    sys.exit(0)
