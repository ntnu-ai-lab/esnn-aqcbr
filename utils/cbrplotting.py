def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l

import numpy as np
import seaborn as sns
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import pandas as pd
from utils.plotting import setLateXFonts
import torch

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
            if type == 0: #esnn
                t = model(torch.from_numpy(data[i]).to(torch.float32).to('cuda:0'),
                          torch.from_numpy(data[i2]).to(torch.float32).to('cuda:0'))[0]
            elif type == 1: # gabel
                t = model(torch.from_numpy(data[i]).to(torch.float32).to('cuda:0').unsqueeze(dim=0),
                          torch.from_numpy(data[i2]).to(torch.float32).to('cuda:0').unsqueeze(dim=0))
            else: #chopra
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
    #print(ret.head(2))
    return ret

def plot2heatmap(df, k, annot=True, outputfile="", plotclasscolor=False):
    plt.clf()
    plt.cla()
    plt.close()
    #plt.figure()
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
    #fig.savefig(f"preeverything-{outputfile}")
    ax1 = plt.subplot2grid((k+1, k), (0, 0), colspan=k-1, rowspan=k-1,fig=fig)
    # labe y axis
    ax2 = plt.subplot2grid((k+1, k), (k-1, 0), colspan=k-1, rowspan=1,fig=fig)
    # axis for correct
    ax21 = plt.subplot2grid((k+1, k), (k, 0), colspan=k-1, rowspan=1,fig=fig)

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
    if plotclasscolor is True:
        resulaxisnot = closestnames.values
    else:
        resulaxisnot = closestnames
    sns.heatmap(correctvalues,
                ax=ax21,  annot=resulaxisnot, cmap="YlGnBu",
                cbar=False, xticklabels=False, fmt="",
                yticklabels=False, vmin=0, vmax=1)
    uval = np.squeeze(correctvalues)
    print(f"correct: {np.sum(uval)/uval.shape[0]}")
    ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax21.set_ylabel('')
    ax21.set_xlabel('')
    import matplotlib._pylab_helpers
    figures = [manager.canvas.figure
               for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    print(figures)
    plt.tight_layout()
    plt.show()
    figures[1].savefig(f"{outputfile}")
    figures[1].savefig(f"2{outputfile}")
    # #plt.show()
    # fig = ax1.get_figure()
    # fig.savefig(f"{outputfile}")
    # fig.savefig(f"post1{outputfile}")
    # plt.show()
    # fig.savefig(f"final{outputfile}")
    # plt.clf()
    # plt.cla()
    # plt.close()
    #ax21.set_xlim(k, -0.5)

    #sns.heatmap(closestdf.transpose().iloc[:,:k],
    #            ax=ax22,  annot=True, cmap="YlGnBu",
    #            cbar=False, xticklabels=False, yticklabels=False)

    #sns.heatmap(labeldf.iloc[:k,:],
    #            ax=ax3,  annot=False, cmap="YlGnBu",
    #            cbar=False, xticklabels=False, yticklabels=False)
    #ax3.set_ylim(k, -0.5)
    #ret.set_yticklabels(labels=ret.get_yticklabels(), rotation=0)
