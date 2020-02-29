from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import os

from models.type3.chopra import ChopraTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
from dataset.dataset import Dataset
from utils.torch import TorchSKDataset
from pytorch_lightning import Trainer
from models.eval import eval_dual_ann, eval_gabel_ann
from torch.utils.data import DataLoader
import numpy as np
from models.esnn.tests.pytorch_esnn_test import ESNNTest
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.newtorcheval import TorchEvaler, runevaler, ChopraTorchEvaler, GabelTorchEvaler
import torch
from models.type2.gabel import GabelTrainer
import os
import time
from sklearn.metrics import matthews_corrcoef
import numpy as np
import seaborn as sns
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

def setLateXFonts():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

import pandas as pd

def makematrixdata(model, data, targets, k, type=0):
    """

    :param model:
    :param data:
    :param targets:
    :param k:
    :param type: 0: esnn, 1: gabel, 2: chopra
    :return:
    """
    model.eval()
    dsize = data.shape[0]
    tmpdata = np.zeros((k,k))
    max = 0
    maxperrow = 0
    min = 1
    highesti2 = 0
    highesti2s = []
    highestnames = []
    highestname = ""
    #
    for i in range(0, k):
        maxperrow = 0
        for i2 in range(0, k):
            if type == 0:
                t = model(torch.from_numpy(data[i]).to(torch.float32).to('cuda:0'),
                          torch.from_numpy(data[i2]).to(torch.float32).to('cuda:0'))[0]
            elif type == 1:
                t = model(torch.from_numpy(data[i]).to(torch.float32).to('cuda:0').unsqueeze(dim=0),
                          torch.from_numpy(data[i2]).to(torch.float32).to('cuda:0').unsqueeze(dim=0))
            else:
                t = model(torch.from_numpy(data[i]).to(torch.float32).to('cuda:0').unsqueeze(dim=0),
                          torch.from_numpy(data[i2]).to(torch.float32).to('cuda:0').unsqueeze(dim=0))
            diff = t.cpu().detach().numpy().squeeze()
            tmpdata[i,i2] = 1 - diff
            if tmpdata[i,i2] > max:
                max = tmpdata[i,i2]

            if tmpdata[i, i2] > maxperrow and i != i2:
                maxperrow = tmpdata[i, i2]
                highesti2 = i2
                highestname = f"opsitu{i2+1}"

            if tmpdata[i,i2] < min:
                min = tmpdata[i,i2]

        highesti2s.append(highesti2)
        highestnames.append(highestname)



    dfdata = []
    for i in range(0, k):
        name = f"opsitu{i+1}"
        dataline = {"name": name, "class": targets[i][0] == 1, "correct": np.all(targets[i] == targets[highesti2s[i]]).squeeze(), "closestname": highestnames[i]}
        for i2 in range(0, k):
            name2 = f"opsitu{i2+1}"
            dataline[name2] = (tmpdata[i,i2]-min)/(max-min)
        dfdata.append(dataline)
    ret = pd.DataFrame(dfdata)
    print(ret.dtypes)
    print(ret.head(2))
    return ret

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

def plot2heatmap(df, k, annot=True, outputfile=""):
    plt.figure()
    plt.clf()
    df = df.set_index('name')

    label_union = df.index.union(df.columns)
    #print(df.head(10))
    #print(df.dtypes)
    labelcols = ["class"]
    othercols = ["correct", "closestname"]
    datacols = set(list(df))-set(labelcols)
    datacols = datacols-set(othercols)
    datacols = list(datacols)
    datadf = df[sort_human(datacols)]
    label_union = datadf.index.union(datadf.columns)
    datadf = datadf.reindex(index=label_union, columns=label_union)
    #df = df.reindex(index=label_union, columns=label_union)
    labeldf = df[labelcols]
    labeldf.reindex(index=label_union)

    correctdf = df[["correct"]]
    correctdf.reindex(index=label_union)

    closestdf = df[["closestname"]]
    closestdf.reindex(index=label_union)

    fig = plt.figure(figsize=(k+1, k))
    fig.savefig(f"preeverything-{outputfile}")
    ax1 = plt.subplot2grid((k+1, k), (0, 0), colspan=k-1, rowspan=k-1)
    # labe y axis
    ax2 = plt.subplot2grid((k+1, k), (k-1, 0), colspan=k-1, rowspan=1)
    # axis for correct
    ax21 = plt.subplot2grid((k+1, k), (k, 0), colspan=k-1, rowspan=1)

    ret = sns.heatmap(datadf.iloc[:k, :k], ax=ax1, annot=True,
                cmap="YlGnBu",
                linecolor='b', cbar = False)
    ax1.xaxis.tick_top()
    ax1.set_ylim(k, -0.5)
    ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)
    #ax1.set_xticklabels(data.columns,rotation=40)

    sns.heatmap(labeldf.transpose().iloc[:,:k],
                ax=ax2, annot=True, cmap="YlGnBu",
                cbar=False, xticklabels=False, yticklabels=False)

    closestnames = closestdf.transpose().iloc[:,:k]
    correctvalues = correctdf.transpose().iloc[:, :k].astype(int).values
    #closestnames = np.squeeze(closestdf.transpose().iloc[:,:k].values)
    #correctvalues = np.squeeze(correctdf.transpose().iloc[:, :k].astype(int).values)
    #closestnames = np.asarray(["{0} {1}".format(string,value)
    #                            for string, value in zip(closestnames, correctvalues)]).T
    #closestnames = np.expand_dims(closestnames,axis=1)
    #correctvalues = np.expand_dims(correctvalues, axis=1)
    correctvalues = correctdf.transpose().iloc[:,:k].astype(int).values
    closestnamesvalues = closestnames.values
    sns.heatmap(correctvalues,
                ax=ax21,  annot=closestnamesvalues, cmap="YlGnBu",
                cbar=False, xticklabels=False, fmt="",
                yticklabels=False)
    plt.show()
    fig = ax1.get_figure()
    fig.savefig(f"{outputfile}")
    fig.savefig(f"post1{outputfile}")
    plt.show()
    fig.savefig(f"final{outputfile}")
    plt.clf()
    plt.cla()
    plt.close()
    #ax21.set_xlim(k, -0.5)

    #sns.heatmap(closestdf.transpose().iloc[:,:k],
    #            ax=ax22,  annot=True, cmap="YlGnBu",
    #            cbar=False, xticklabels=False, yticklabels=False)

    #sns.heatmap(labeldf.iloc[:k,:],
    #            ax=ax3,  annot=False, cmap="YlGnBu",
    #            cbar=False, xticklabels=False, yticklabels=False)
    #ax3.set_ylim(k, -0.5)
    #ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)

def plotESNNOpsitu(lr=0.2, networklayers=[50, 30, 3], dropout=0.05, epochs=200, validate_on_k=10, n=5):
    model, \
    dsl, \
    train, \
    test, \
    evaler, \
    best_model,\
    losses,\
    lossdf = runevaler("opsitu", epochs, ESNNSystem, TorchEvaler,
                           eval_dual_ann, networklayers=networklayers, lr=lr,
                           dropoutrate=dropout, validate_on_k=validate_on_k, n=n, filenamepostfix="esnn")

    df1 = makematrixdata(best_model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=0)
    plot2heatmap(df1, 10, annot=True, outputfile="esnn-matrix.pdf")

    df2 = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=0)
    plot2heatmap(df2, 10, annot=True, outputfile="training-esnn-matrix.pdf")
    return losses, lossdf

def plotCHOPRAOpsitu(lr=0.2, networklayers=[50, 30, 3], dropout=0.05, epochs=200, validate_on_k=10, n=5):
    model, \
    dsl, \
    train, \
    test, \
    evaler, \
    best_model, \
    losses,\
    lossdf  = runevaler("opsitu", epochs, ChopraTrainer, ChopraTorchEvaler,
                           eval_dual_ann, networklayers=networklayers, lr=lr,
                           dropoutrate=dropout, validate_on_k=validate_on_k, n=n, filenamepostfix="chopra")

    df = makematrixdata(best_model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=2)
    plot2heatmap(df, 10, annot=True, outputfile="chopra-matrix.pdf")

    df = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=2)
    plot2heatmap(df, 10, annot=True, outputfile="training-chopra-matrix.pdf")
    return losses, lossdf

def plotGableOpsitu(lr=0.03, networklayers=[50, 30, 3], dropout=0.05, epochs=200, validate_on_k=10, n=5):
    model, \
    dsl, \
    train, \
    test, \
    evaler, \
    best_model, \
    losses, \
    lossdf = runevaler("opsitu", epochs, GabelTrainer, GabelTorchEvaler,
              eval_gabel_ann, networklayers=networklayers,
              lr=lr, dropoutrate=dropout, validate_on_k=validate_on_k, n=n,
              filenamepostfix="gabel")
    df1 = makematrixdata(best_model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=1)
    plot2heatmap(df1, 10, annot=True, outputfile="gabel-matrix.pdf")

    df2 = makematrixdata(model, dsl.getFeatures()[train], dsl.getTargets()[train], 10, type=1)
    plot2heatmap(df2, 10, annot=True, outputfile="training-gabel-matrix.pdf")
    return losses, lossdf

import sys
if __name__ == "__main__":
    lossdfs = []

    dsl, \
        trainedmodels,\
        validatedmodels,\
        losses,\
        lossdf = runevaler("opsitu", 500, [ESNNSystem, ChopraTrainer, GabelTrainer],
                           [TorchEvaler, ChopraTorchEvaler, GabelTorchEvaler],
                           [eval_dual_ann, eval_dual_ann, eval_gabel_ann],
                           networklayers=[[40, 6, 3],[40, 6, 3],[40, 6, 3]],
                           lrs=[0.08, 0.08, 0.02],
                           dropoutrates=[0.005, 0.005, 0.005],
                           validate_on_k=10, n=5,
                           filenamepostfixes=["esnn", "chopra", "gabel"])
    #allavgdf = pd.concat(lossdfs)
    plt.clf()
    plt.cla()
    plt.close()
    setLateXFonts()
    lossdf.to_csv("lossdf.csv")
    sns_plot = sns.lineplot(x="epoch", y="signal", hue="label", data=lossdf)
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig("allavg.pdf")

    sys.exit(0)
