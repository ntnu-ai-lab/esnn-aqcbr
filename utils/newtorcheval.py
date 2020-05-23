from dataset.makeTrainingData import makeDualSharedArchData, makeGabelTrainingData

from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from dataset.dataset import Dataset
from sklearn.metrics import roc_auc_score
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from models.type1 import sim_def_lin
import torch
import os
import time
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
# from models.eval import roc
from utils.torch import torch_auc_roc

def getpearsons(df, col1, col2):
    labelarr = df.as_matrix(columns=col1).squeeze()
    newarr = np.zeros_like(labelarr)
    for i in range(0,labelarr.shape[0]):
        newarr[i] = int(labelarr[i] == "failure")

    effectarr = df.as_matrix(columns=col2).squeeze()
    pc = pearsonr(effectarr, newarr)
    return pc


def remap(x):
    if '_1.0' in x:
        return 'failure'
    return 'success'


def pairplot(dsl, output, k):
    labelcols = ["class"]
    wanteddatacols = ["wind speed", "wind from direction", "wind effect"]
    df = dsl.unhotify()
    df = df.rename(columns={'wind_speed': 'wind speed',
                            'wind_from_direction': 'wind from direction',
                            'wind_effect': 'wind effect'})
    # 'weatherhindrance_1.0':'failure', 'weatherhindrance_0.0':'success'})

    df['class'] = df['class'].apply(lambda x: remap(x))

    ax = sns.pairplot(df, vars=wanteddatacols, hue="class")

    pc = getpearsons(df, labelcols, ["wind effect"])
    print(f"pearsonscorr: {pc[0]} , {pc[1]}")
    ax.savefig(f"{output}.pdf", format="pdf", bbox_inches='tight')
    return df


# TODO: this function needs to be split up!

def runevaler(datasetname, epochs, models, torchevalers, evalfuncs,
              networklayers, lrs, dropoutrates,
              validate_on_k=10, filenamepostfixes=[], n_splits=5,
              n=5, removecoverage=True, prefix="", augmentData=True):
    #print("evaling model: ")
    #model.summarize(model,mode="full")
    device = "cpu"
    gpus = None
    if torch.cuda.is_available():
        gpus = [0]
        device = "cuda:0"
    d = Dataset(datasetname)
    dsl, colmap, stratified_fold_generator = fromDataSetToSKLearn(d,
                                                                  True,
                                                                  n_splits=n_splits)

    data = dsl.getFeatures()
    target = dsl.getTargets()
#print the data dist..
    #pairplot(dsl,
    #         filenamepostfix+"pairplot", 10)

    datasetinfo = dsl.dataset.datasetInfo
    # if "augmentTrainingData" in datasetinfo:
    #     func = datasetinfo["augmentTrainingData"]
    #     data, target = func(data, target)
    #     dsl.setData(np.concatenate((data, target), axis=1))
        #train_data, train_target = func(train_data, train_target)
        # test_data, test_target = func(test_data, test_target)

    data = dsl.getFeatures()
    target = dsl.getTargets()
    tlossmean = [np.zeros((epochs, n)) for filname in filenamepostfixes]

    trainedmodels = [[] for filname in filenamepostfixes]
    vlossmean = [np.zeros((int(epochs/validate_on_k), n)) for filname in filenamepostfixes]
    aucmean = [np.zeros((int(epochs/validate_on_k), n)) for filname in filenamepostfixes]
    validatedmodels = [[] for filname in filenamepostfixes]
    knndata = []
    truedata = [np.squeeze(np.zeros((1, n)),axis=0) for filname in filenamepostfixes]
    errdata = [np.squeeze(np.zeros((1, n)),axis=0) for filname in filenamepostfixes]
    for i in range(0, n):
        thisk = 0
        tlossdata = [np.zeros((epochs, n_splits)) for filname in filenamepostfixes]
        aucvaldata = [np.zeros((int(epochs/validate_on_k),n_splits)) for filname in filenamepostfixes]
        vlossdata = [np.zeros((int(epochs/validate_on_k), n_splits)) for filname in filenamepostfixes]
        knnresults = 0
        stratified_fold_generator = dsl.getSplits(n_splits=n_splits)
        splittruedata = [np.squeeze(np.zeros((1, n_splits))) for filname in filenamepostfixes]
        spliterrdata = [np.squeeze(np.zeros((1, n_splits))) for filname in filenamepostfixes]
        kerrratios = 0
        ktrueratios = 0
        for train, test in stratified_fold_generator:
            ec = 0
            thisk = thisk +1
            test_data = data[test]
            test_target = target[test]

            train_data = data[train]
            train_target = target[train]
            batchsize = test_data.shape[0] * train_data.shape[0]
            if "augmentTrainingData" in datasetinfo and augmentData:
                func = datasetinfo["augmentTrainingData"]
                train_data, train_target = func(train_data, train_target)
                #dsl.setData(np.concatenate((ata, target), axis=1))
            onennres = sim_def_lin(None, test_data, test_target,
                                   train_data, train_target,
                                   batchsize, False, None)
            knnresults += np.sum(onennres)/len(onennres)
            
            for torchevaler in torchevalers:
                #train, test = next(stratified_fold_generator)

                sys = models[ec](None, train_data, train_target,
                                 validation_func=evalfuncs[ec],
                                 train_data=train_data, train_target=train_target,
                                 test_data=test_data, test_target=test_target,
                                 colmap=colmap, device=device,
                                 networklayers=networklayers[ec], lr=lrs[ec],
                                 dropoutrate=dropoutrates[ec])

                sys.cuda(0)
                rootdir = os.getcwd()
                filename = rootdir + prefix \
                    + f"train-{epochs}e-{thisk}k-{i+1}n-{removecoverage}rc"\
                    + filenamepostfixes[ec]+str(os.getpid())
                #print(f"filename: {filename}")
                checkpoint_callback = ModelCheckpoint(
                    filepath=filename,
                    verbose=False,
                    monitor='val_loss',
                    mode='min',
                    prefix=''
                )
                evaler = torchevalers[ec](dsl, train,
                                          modelcheckpoint=checkpoint_callback,
                                          validate_on_k=validate_on_k,
                                          evalfunc=evalfuncs[ec], train_data=train_data,
                                          train_target=train_target, test_data=test_data,
                                          test_target=test_target, colmap=colmap)
                t0 = time.time()
                methodlabel = filenamepostfixes[ec]
                tmodel, \
                    torchmodels, \
                    jitmodels,\
                    best_model,\
                    losses, \
                    tlosses, \
                    vlosses,\
                    aucvals, \
                    besterrdict,\
                    besttruedict = evaler.myeval(sys, epochs,
                                            sys.opt, sys.loss, device,
                                            datalabel=methodlabel,
                                            fileprefix=filename)
                biggest=0
                biggestkey=None
                dictsum = 0
                for k,v in  besttruedict.items():
                    if v > biggest:
                        biggest = v
                        biggestkey = k
                    dictsum += v
                trueratio = biggest/dictsum
                biggest = 0
                biggestkey = None
                dictsum = 0
                for k, v in besterrdict.items():
                    if v > biggest:
                        biggest = v
                        biggestkey = k
                    dictsum += v
                errratio = biggest / dictsum
                ktrueratios += trueratio
                kerrratios += errratio


                trainedmodels[ec].append(tmodel)
                validatedmodels[ec].append(best_model)
                tlossdata[ec][:, thisk-1] = np.asarray(tlosses)
                lasttloss = tlosses[len(tlosses)-1]
                lastvloss = vlosses[len(vlosses)-1]
                vlossdata[ec][:, thisk-1] = np.asarray(vlosses)
                aucvaldata[ec][:, thisk - 1] = np.asarray(aucvals)
                splittruedata[ec][thisk-1] = trueratio
                spliterrdata[ec][thisk-1] = errratio
                t1 = time.time()
                diff = t1-t0
                print(f"n:[{i+1}/{n}] k:[{thisk}/{n_splits}] running {filenamepostfixes[ec]}: {lasttloss} t {lastvloss} v")
                #print(f"spent {diff} time training {diff/epochs} per epoch")
                ec = ec + 1
                
            # ktrueratios = ktrueratios / n_splits
            # kerrratios = kerrratios / n_splits
            #truedata[]
        knnresults = knnresults/n_splits
        knndata.append(knnresults)

        for evalcounter in range(0, len(filenamepostfixes)):
            vlossmean[evalcounter][:,i] = np.mean(vlossdata[evalcounter], axis=1)
            aucmean[evalcounter][:, i] = np.mean(aucvaldata[evalcounter], axis=1)
            tlossmean[evalcounter][:,i] = np.mean(tlossdata[evalcounter], axis=1)
            truedata[evalcounter][i] = np.mean(splittruedata[evalcounter], axis=0)
            errdata[evalcounter][i] = np.mean(spliterrdata[evalcounter], axis=0)

    dfdata = []
    i = 0
    for evalcounter in range(0,len(filenamepostfixes)):
        for row in tlossmean[evalcounter]:
            i = i + 1
            for cel in row:
                dfdata.append({'epoch': i,
                               'label': f'{filenamepostfixes[evalcounter]}.train',
                               'loss': float(cel)})
        i=0
    for evalcounter in range(0,len(filenamepostfixes)):
        for row in vlossmean[evalcounter]:
            i = i + 1
            for cel in row:
                dfdata.append({'epoch': int(i*validate_on_k),
                               'label': f'{filenamepostfixes[evalcounter]}.val',
                               'loss': float(cel)})
        i=0

    for evalcounter in range(0,len(filenamepostfixes)):
        for row in aucmean[evalcounter]:
            i = i + 1
            for cel in row:
                dfdata.append({'epoch': int(i*validate_on_k),
                               'label': f'{filenamepostfixes[evalcounter]}.auc',
                               'loss': float(cel)})
        i=0

    ratiodata = []
    for evalcounter in range(0,len(filenamepostfixes)):
        for i in range(0,n):
            ratiodata.append({'n:': i, 'method': f'{filenamepostfixes[evalcounter]}',
                              'type': 'true', 'value': truedata[evalcounter][i]})
            ratiodata.append({'n:': i, 'method': f'{filenamepostfixes[evalcounter]}',
                              'type': 'err', 'value': errdata[evalcounter][i]})
    # for row in vlossmean:
    #     i = i + 1
    #     for cel in row:
    #         dfdata.append({'epoch': int(i*validate_on_k), 'label': f'{filenamepostfix}.val', 'signal': float(cel)})

    lossdf = pd.DataFrame(dfdata)
    ratiodf = pd.DataFrame(ratiodata)
    return dsl, trainedmodels, validatedmodels, losses, \
                             lossdf, knndata, ratiodf


class TorchEvaler():
    def __init__(self, sklearndataset, trainidx,
                 transform=None, should_invert=True,
                 modelcheckpoint=None, validate_on_k=10,
                 evalfunc=None, train_data=None,
                 train_target=None, test_data=None, test_target=None,
                 colmap=None):

        self.sklearandataset = sklearndataset
        self.loadData(sklearndataset, trainidx)
        self.transform = transform
        self.should_invert = should_invert
        self.featurelength = sklearndataset.featurecolsto - \
            sklearndataset.featurecolsfrom
        self.x1start = 0
        self.evalfunc = evalfunc
        self.x2start = self.featurelength+1
        self.modelcheckpoint = modelcheckpoint
        self.x1stop = self.featurelength+1
        self.x2stop = 2*(self.featurelength+1)
        self.x1s = torch.from_numpy(self.data[:, self.x1start:self.x1stop].astype('float')).to(torch.float32).to('cuda:0')
        self.x2s = torch.from_numpy(self.data[:, self.x2start:self.x2stop].astype('float')).to(
            torch.float32).to('cuda:0')
        self.y1 = torch.from_numpy(self.y1).to(torch.float32).to('cuda:0')
        self.y2 = torch.from_numpy(self.y2).to(torch.float32).to('cuda:0')
        self.labels = torch.from_numpy(self.targets[:].astype('float')).to(torch.float32).to('cuda:0')
        # set the path for the callbacks
        self.trainerio = TrainerIOMixin()
        self.modelcheckpoint.save_function = self.trainerio.save_checkpoint
        self.trainerio.optimizers = []
        self.trainerio.lr_schedulers = []
        self.validate_on_k = validate_on_k
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        # self.alldata = np.concatenate((self.train_data,self.test_data), axis=0)
        # self.alltarget = np.concatenate((self.train_target,self.test_target), axis=0)
        # self.firsthalfdata, self.secondhalfdata = np.split(self.alldata,2)
        # self.firsthalftarget, self.secondhalftarget = np.split(self.alltarget,2)
        self.colmap = colmap



    def evalepoch(self, model, criterion, optim):
        """
        This code is for evaluating ESNN and computing the loss..
        """
        y_hat, inner_output1, inner_output2 = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels, self.y1,
                         self.y2, inner_output1, inner_output2)
        y_hatc = y_hat.clone()
        y_hatc = y_hatc.detach().cpu().numpy()
        rocs = roc_auc_score(self.labels.cpu().numpy(), y_hatc)
        #return loss+(2*(1-rocs))
        return 1.0-rocs


    def myeval(self, model, epochs, optim, criterion,
               device, datalabel="", fileprefix=""):
        batch_size = self.train_data.shape[0]*self.test_data.shape[0]
        self.trainerio.model = model
        lastbest = 1
        torchmodels = []
        jitmodels = []
        losses = []
        tlosses = []
        vlosses = []
        aucvals = []
        besterrdict = None
        besttruedict = None

        for epoch in range(epochs):
            #idx = torch.randperm(self.x1s.nelement())
            optim.zero_grad()
            loss = self.evalepoch(model, criterion, optim)
            loss.backward()
            lossc = loss.clone()
            tloss = lossc.detach().cpu().numpy().squeeze()
            losses.append({'epoch': epoch, 'label': f'{datalabel}.train', 'loss': float(tloss)})
            tlosses.append(tloss)
            optim.step()
            self.trainerio.current_epoch = epoch
            self.trainerio.global_step = epoch
            #print('Epoch [%d/%d], loss: %.4f,'
            #      % (epoch, epochs, loss.data))

            if (epoch+1) % self.validate_on_k == 0:
                res, errdistvec, truedistvec, \
                    combineddata, y_pred, y_true = self.evalfunc(model,
                                                           # self.firsthalfdata,
                                                           # self.firsthalftarget,
                                                           # self.secondhalfdata,
                                                           # self.secondhalftarget,
                                                           self.test_data,
                                                           self.test_target,
                                                           self.train_data,
                                                           self.train_target,
                                                           batch_size=batch_size,
                                                           anynominal=False,
                                                           colmap=self.colmap,
                                                           device=device)

                
                labelwidth = self.test_target.shape[1]
                label_indexes=[(combineddata.shape[1]-2)-2*labelwidth,
                               (combineddata.shape[1]-2)-labelwidth]
                # df = roc(combineddata, combineddata.shape[1]-2, label_width=labelwidth,
                #          label_indexes=[(combineddata.shape[1]-2)-2*labelwidth,
                #             (combineddata.shape[1]-2)-labelwidth],
                #          start=0.01, stop=0.8, delta=0.05)

                y_scores = y_pred
                rocs = roc_auc_score(y_true, y_scores)
                auc = None#np.trapz(y=df.tpr.values,x=df.fpr.values)
                val_loss = 1.0-np.sum(res)/len(res)
                losses.append({'epoch': epoch, 'label': f'{datalabel}.val', 'loss': val_loss, 'auc': rocs})
                vlosses.append(val_loss)
                aucvals.append(rocs)
                if val_loss < lastbest:
                    old_loss = lastbest
                    lastbest = val_loss

                    from collections import OrderedDict
                    import pickle
                    copyed_model = pickle.loads(pickle.dumps(model))
                    best_model = pickle.loads(pickle.dumps(model))
                    best_model_state_dict = {k: v.to('cpu') for k, v in copyed_model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)
                    copyed_model.load_state_dict(best_model_state_dict)
                    smod = torch.jit.script(copyed_model)
                    smod.eval()
                    smod = smod.cpu()
                    filepath = self.modelcheckpoint.filepath+".pt1"
                    besterrdict = errdistvec
                    besttruedict = truedistvec
                    #print(f" Epoch [{epoch}/{epochs}] saving java model to {filepath} new {val_loss} old {old_loss}")
                    smod.save(filepath)
                    jitmodels.append(smod)
                tensorboard_logs = {'val_loss': val_loss}
                metrics = {'val_loss': val_loss, 'lg': tensorboard_logs}
                self.modelcheckpoint.on_epoch_end(epoch, metrics)
        return model, torchmodels, jitmodels, best_model, \
            losses, tlosses, vlosses, aucvals, besterrdict, besttruedict

    def loadData(self, sklearndataset, trainidx):
        self.data, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeDualSharedArchData(sklearndataset.getFeatures()[trainidx],
                                       sklearndataset.getTargets()[trainidx], False)




class ChopraTorchEvaler(TorchEvaler):
    def __init__(self, *args, **kwargs):
        super(ChopraTorchEvaler, self).__init__(*args, **kwargs)
        self.labels = self.labels.squeeze()

    def evalepoch(self,  model, criterion, optim):
        """
        This code is for evaluating Chopra and computing the loss..
        """
        y_hat = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels)

        return loss

class GabelTorchEvaler(TorchEvaler):
    def loadData(self, sklearndataset, trainidx):
        self.data, \
            self.targets, \
            self.y1,\
            self.y2 = \
                makeGabelTrainingData(sklearndataset.getFeatures()[trainidx],
                                      sklearndataset.getTargets()[trainidx], False)

    def __init__(self, *args, **kwargs):
        super(GabelTorchEvaler, self).__init__(*args, **kwargs)


    def evalepoch(self,  model, criterion, optim):
        """
        This code is for evaluating GAbel and computing the loss..
        """
        # inp1 = torch.cat([self.x1s, self.x2s], dim=0).squeeze()
        # inp2 = torch.cat([self.x2s, self.x1s], dim=0).squeeze()
        # ll = torch.cat([self.labels, self.labels], dim=0)
        # llones = torch.ones(ll.shape).to(torch.float32).to("cuda:0")
        # ll = ll-llones
        # ll = torch.pow(ll,2)
        # y_hat = model(inp1, inp2)
        # loss = criterion(y_hat, ll)
        y_hat = model(self.x1s, self.x2s)
        loss = criterion(y_hat, self.labels)

        return loss
