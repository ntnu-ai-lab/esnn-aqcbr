from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import os

from models.type3.chopra import ChopraTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
from models.esnn.pytorch_model import ESNNModel
from models.esnn.pytorch_trainer import ESNNSystem
from dataset.dataset import Dataset
from utils.my_torch_utils import TorchSKDataset
from pytorch_lightning import Trainer
from models.evalfunctions import eval_dual_ann, eval_gabel_ann
from torch.utils.data import DataLoader
import numpy as np
from models.esnn.tests.pytorch_esnn_test import ESNNTest
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.torcheval import TorchEvaler, runevaler, ChopraTorchEvaler, GabelTorchEvaler
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

def testTorchESNNOPSITU(datasetname, epochs):
    device = "cpu"
    gpus = None
    if torch.cuda.is_available():
        gpus = [0]
        device = "cuda:0"
    d = Dataset(datasetname)
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d, True, n_splits=5)
    data = dsl.getFeatures()
    target = dsl.getTargets()
    train, test = next(stratified_fold_generator)

    test_data = data[test]
    test_target = target[test]

    train_data = data[train]
    train_target = target[train]
    datasetinfo = dsl.dataset.datasetInfo
    if "augmentTrainingData" in datasetinfo:
        func = datasetinfo["augmentTrainingData"]
        train_data, train_target = func(train_data, train_target)
        test_data, test_target = func(test_data, test_target)
    batchsize = test_data.shape[0]*train_data.shape[0]
    evalfunc = eval_dual_ann
    sys = ESNNSystem(None, train_data, train_target, validation_func=evalfunc,
                     train_data=train_data, train_target=train_target,
                     test_data=test_data, test_target=test_target,
                     colmap=colmap, device=device,
                     networklayers=[50, 50, 50], lr=.02,
                     dropoutrate=0.5)
    sys = ESNNSystem(None, train_data, train_target, validation_func=evalfunc,
                     train_data=train_data, train_target=train_target,
                     test_data=test_data, test_target=test_target,
                     colmap=colmap, device=device,
                     networklayers=[50, 50, 50], lr=0.02,
                     dropoutrate=0.05)
    sys.cuda(0)
    filename = "heh"
    rootdir = os.getcwd()
    filename = rootdir+"train-"+filename+str(os.getpid())
    checkpoint_callback = ModelCheckpoint(
        filepath=filename,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    evaler = TorchEvaler(dsl, train,
                         modelcheckpoint=checkpoint_callback,
                         validate_on_k=10,
                         evalfunc=evalfunc,train_data=train_data,
                         train_target=train_target, test_data=test_data,
                         test_target=test_target, colmap=colmap)
    t0 = time.time()
    evaler.myeval(sys,epochs,sys.opt,sys.loss,device=device)
    t1 = time.time()
    diff = t1-t0
    print(f"spent {diff} time training {diff/epochs} per epoch")
    res, errdistvec, truedistvec, \
        combineddata, pred_vec = evalfunc(sys, test_data, test_target, train_data,
                                          train_target, batch_size=batchsize,
                                          anynominal=False, colmap=colmap,
                                          device=device)

    ires = np.asarray(res).astype(int)
    ires = ires-np.ones(ires.shape)
    ires = ires*-1
    ires[ires < 0.1] = -1
    ipred = np.asarray(pred_vec).astype(int)
    ipred[ipred > 0.499] = 1
    ipred[ipred <= 0.499] = -1
    print(f"{np.sum(res)/len(res)}")
    print(f"errdistvec: {errdistvec}")
    print(f"truedistvec: {truedistvec}")
    print(f"MCC: {matthews_corrcoef(ires,ipred)}")

def testTorchESNNOPSITU2(datasetname, epochs, eval_on_k, filename):

    runevaler(datasetname, epochs, ESNNSystem, TorchEvaler,
              eval_dual_ann, networklayers=[40, 40, 40],
              lr=.09, dropoutrate=0.5, eval_on_k=eval_on_k,
              filenamepostfix=filename)

def choprairis():
    runevaler("iris", 200, ChopraTrainer, ChopraTorchEvaler,
               eval_dual_ann, networklayers=[40, 40],
               lr=.05, dropoutrate=0.05, validate_on_k=10,
               filenamepostfix="chopra")

def gabeliris():
    runevaler("iris", 2000, GabelTrainer, GabelTorchEvaler,
              eval_dual_ann, networklayers=[40, 40, 40],
              lr=.01, dropoutrate=0.00, validate_on_k=10,
              filenamepostfix="gabel")

def gabelopsitu():
    runevaler("opsitu", 2000, GabelTrainer, GabelTorchEvaler,
              eval_gabel_ann, networklayers=[70, 50, 20],
              lr=.03, dropoutrate=0.3, validate_on_k=10,
              filenamepostfix="gabel")
def esnnopsitu():
    model, \
        test_data, \
        test_target, \
        evaler,\
        best_model = runevaler("opsitu", 200, ESNNSystem, TorchEvaler,
                               eval_dual_ann, networklayers=[70, 20, 3], lr=.02,
                               dropoutrate=0.05, validate_on_k=10, filenamepostfix="esnn")
    return model, best_model, test_data, test_target

def chopraopsitu():
    model, \
        test_data, \
        test_target, \
        evaler, \
        best_model = runevaler("opsitu", 2000, ChopraTrainer, ChopraTorchEvaler,
                               eval_dual_ann, networklayers=[30, 10, 2],
                               lr=.01, dropoutrate=0.02, validate_on_k=10,
                               filenamepostfix="chopra")
    print(model(evaler.x1s[0:1,], evaler.x1s[0:1,]))
    print(model(evaler.x1s[1:2, ], evaler.x1s[1:2, ]))
    print(model(evaler.x1s[1:2,], evaler.x1s[0:1,]))


def esnniris():
    runevaler("iris", 200, ESNNSystem, TorchEvaler, eval_dual_ann, networklayers=[40, 10, 2], lr=.02,
              dropoutrate=0.05, validate_on_k=10, filenamepostfix="esnn")

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
    # axis for closestname
    # ax22 = plt.subplot2grid((k+2, k), (k+1, 0), colspan=k - 1, rowspan=1)
    #ax3 = plt.subplot2grid((k+1, k), (0, k-1), colspan=1, rowspan=k-1)


    #mask = np.zeros_like(data)
    #mask[np.tril_indices_from(mask)] = True

    #labeldata = data[labelcols]

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
    #gabeliris()
    #gabelopsitu()
    #chopraopsitu()
    #choprairis()
    #chopralosses, chopralossdf = plotCHOPRAOpsitu(lr=0.08, networklayers=[40, 6, 3], dropout=0.005, epochs=50,n=1)
    #setLateXFonts()
    #sns_plot = sns.lineplot(x="epoch", y="signal", hue="label", data=chopralossdf)
    #plt.show()
    #fig = sns_plot.get_figure()
    #fig.savefig("chopraloss.pdf")
    #gabellosses, gabelossdf = plotGableOpsitu(lr=0.02, networklayers=[40, 6, 3], dropout=0.005, epochs=500)
    #lossdfs.append(gabelossdf)
    #gabeldf = pd.DataFrame(gabellosses)

    gabellosses, gabelossdf = plotGableOpsitu(lr=0.02, networklayers=[40, 6, 3], dropout=0.005, epochs=500)
    lossdfs.append(gabelossdf)
    #esnndf = pd.DataFrame(esnnlosses)
    chopralosses,chopralossdf = plotCHOPRAOpsitu(lr=0.08, networklayers=[40, 6, 3], dropout=0.005, epochs=500)
    lossdfs.append(chopralossdf)
    esnnlosses, esnnlossdf = plotESNNOpsitu(lr=0.08, networklayers=[40, 6, 3], dropout=0.1, epochs=500)
    lossdfs.append(esnnlossdf)
    #chopradf = pd.DataFrame(chopralosses)
    # sns_plot = sns.lineplot(x="epoch", y="signal", hue="label", data=esnndf)
    # plt.show()
    # fig = sns_plot.get_figure()
    # fig.savefig("esnnloss.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # plt.clf()
    # plt.cla()
    # plt.close()
    # sns_plot2 = sns.lineplot(x="epoch", y="signal", hue="label", data=chopradf)
    # plt.show()
    # fig = sns_plot2.get_figure()
    # fig.savefig("chopraloss.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # plt.clf()
    # plt.cla()
    # plt.close()
    # sns_plot3 = sns.lineplot(x="epoch", y="signal", hue="label", data=gabeldf)
    # plt.show()
    # fig = sns_plot3.get_figure()
    # fig.savefig("gabelloss.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close()

    allavgdf = pd.concat(lossdfs)
    plt.clf()
    plt.cla()
    plt.close()
    setLateXFonts()
    sns_plot = sns.lineplot(x="epoch", y="signal", hue="label", data=allavgdf)
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig("allavg.pdf")

    sys.exit(0)

    #esnniris()
    #testTorchESNNOPSITU2("iris",200)



    #testTorchESNNOPSITU("iris",200)
