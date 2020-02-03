import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow.keras.models
import random
import torch

import scipy.spatial.distance as scdist
# input is a matrix concated from two matrices of the series of outputs of chopranet for pairs if
# input datapoints
def choprascore(y_pred):
    #g_shape = tf.shape(y_pred)[1]
    #g_length = tf.constant(tf.round(tf.divide(g_shape,tf.constant(2))))
    y1 = np.zeros((y_pred.shape[0],int(y_pred.shape[1]/2)))
    y2 = np.zeros((y_pred.shape[0],int(y_pred.shape[1]/2)))
    ret = np.zeros((y_pred.shape[0],1))
    for i in range(0,y_pred.shape[0]):
        ret[i] = scdist.cityblock(y1[i],y2[i])
    return ret

def eval_normal_ann_l2(model, test_data, test_target, train_data,
                    train_target, batch_size, anynominal=False, colmap=None):

    error_vec = list()
    if test_data.shape[0] != test_target.shape[0]:
        return
    #for i in range(0,test_data.shape[0]):
    if type(model) == "Sequential":
        test_pred_vec = model.predict(test_data, batch_size=batch_size)
        train_pred_vec = model.predict(train_data, batch_size=batch_size)
    else:
        test_pred_vec = model.predict(test_data)
        train_pred_vec = model.predict(train_data)
    #iterate through the test instances (Queries)
    for q_instance,q_target_instance in zip(test_pred_vec, test_target):
        allSame = True
        mindist = 1
        closest = None
        #iterate through the training instances (the case base)
        for c_instance,c_target_instance in zip(train_pred_vec, train_target):
            dist = np.linalg.norm(c_instance-q_instance) #euclidian norm
            # maybe add https://docs.scipy.org/doc/scipy/reference/spatial.distance.html for manhattan, to test?
            if dist < mindist:
                mindist = dist
                closest = c_target_instance

        for target_dim in range(0,test_target.shape[1]):
            if not np.rint(closest[target_dim]) == np.rint(q_target_instance[target_dim]):
                    allSame = False
        error_vec.append(1.0 if allSame else 0.0)

    return error_vec

def eval_normal_ann_l1(model, test_data, test_target, train_data,
                    train_target, batch_size, anynominal=False, colmap=None):
    error_vec = list()
    if test_data.shape[0] != test_target.shape[0]:
        return
    #for i in range(0,test_data.shape[0]):
    if type(model) == "Sequential":
        test_pred_vec = model.predict(test_data, batch_size=batch_size)
        train_pred_vec = model.predict(train_data, batch_size=batch_size)
    else:
        test_pred_vec = model.predict(test_data)
        train_pred_vec = model.predict(train_data)
    #iterate through the test instances (Queries)
    for q_instance,q_target_instance in zip(test_pred_vec, test_target):
        allSame = True
        mindist = 1
        closest = None
        #iterate through the training instances (the case base)
        for c_instance,c_target_instance in zip(train_pred_vec, train_target):
            dist = scdist.cityblock(c_instance,q_instance) #euclidian norm
            # maybe add https://docs.scipy.org/doc/scipy/reference/spatial.distance.html for manhattan, to test?
            if dist < mindist:
                mindist = dist
                closest = c_target_instance

        for target_dim in range(0,test_target.shape[1]):
            if not np.rint(closest[target_dim]) == np.rint(q_target_instance[target_dim]):
                    allSame = False
        error_vec.append(1.0 if allSame else 0.0)

    return error_vec

def getAscOrDescIndex(shape, asc):
    if asc:
        return shape-1
    else:
        return 0

def eval_dual_ann(model, test_data, test_target, train_data,
                  train_target, batch_size, anynominal=False,
                  colmap=None, gabel=False):
    return _eval_dual_ann(model, test_data, test_target, train_data,
                          train_target, batch_size, anynominal, colmap,
                          gabel=gabel, distance=True)


def eval_gabel_ann(model, test_data, test_target, train_data,
                   train_target, batch_size, anynominal=False, colmap=None):
    return _eval_dual_ann(model, test_data, test_target, train_data,
                         train_target, batch_size, anynominal, colmap,
                         gabel=True, distance=False)


def eval_chopra_ann(model, test_data, test_target, train_data,
                    train_target, batch_size, anynominal=False, colmap=None,
                    distance = True):
    return _eval_dual_ann(model, test_data, test_target, train_data,
                         train_target, batch_size, anynominal, colmap,
                         gabel=False, distance=True)


def roc(matrix, distindex, label_width,
        label_indexes, start, stop, delta):
    """This function calculates ROC/AUC and F2

    Args:
       matrix the datamatrix
       distindex the column index of the distance in the matrix
       true_label the true label
       label_index index of the label in the data martix
       start start point for the ROC curve
       stop stop point for the ROC curve
       delta delta of how much to move for each iteration in the ROC curve

    Returns: ROC/AUC and F2
    yup

    """

    iters = (stop-start)/delta
    datalist = []
    for t in np.arange(start, stop, delta):
        tp, fp, tn, fn, totn, totp = auc(matrix, t, distindex,
                                         label_width, label_indexes)
        datalist.append({"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                         "totn": totn, "totp": totp, "t": t,
                         "tpr": tp/totp, "fpr": fp/totn})
    return pd.DataFrame(data=datalist)


def auc(matrix, threshold, distindex, label_width, label_indexes):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    totp = 0
    totn = 0
    for i in range(0, matrix.shape[0]):
        left_label = matrix[i, label_indexes[0]:label_indexes[0]+label_width]
        right_label = matrix[i, label_indexes[1]:label_indexes[1]+label_width]
        if matrix[i, distindex] < threshold:
            if np.all(left_label == right_label):
                tp += 1
                totp += 1
            else:
                fp += 1
                totn += 1
        else:
            if np.all(left_label != right_label):
                tn += 1
                totn += 1
            else:
                fn += 1
                totp += 1
    return tp, fp, tn, fn, totn, totp


def old_auc(matrix, threshold, distindex, true_label, label_index):
    belowtindexes = [np.where(r[distindex] < threshold) for r in matrix]
    tp = 0
    fp = 0
    for i in range(0,len(belowtindexes)):
        label = matrix[belowtindexes[i],label_index]
        if label == true_label:
            tp += 1
        else:
            fp += 1
    abovetindexes = [np.where(r[distindex] >= threshold) for r in matrix]
    totp = 0
    fotn = 0
    for i in range(0,len(abovetindexes)):
        label = matrix[abovetindexes[i],label_index]
        if label == true_label:
            tp += 1
        else:
            fp += 1
    return tp, fp

def _eval_dual_ann(model, test_data, test_target, train_data, train_target,
                   batch_size, anynominal=False, colmap=None,
                   gabel=False, distance=True):
    if test_data.shape[0] != test_target.shape[0]:
        return

    # we stuff everything into this big array, data, Y_pred and Y_true_1 and Y_true2
    combineddata = np.zeros((train_data.shape[0]*test_data.shape[0],
                            (train_data.shape[1]*2)+(test_target.shape[1]*2)+2))
    for i in range(0,test_data.shape[0]):

        # moving from left to right in this array..

        # 1. for this sub-rectangle of the array we set the
        # top-left corner to be the training data R_train
        combineddata[(i*train_data.shape[0]):((i+1)*train_data.shape[0]),
                    0:train_data.shape[1]] = train_data

        # 2. To right of 1. we set R_test to be right of R_train to be equal
        # to a copy of the test data at test_data[i], so combined this is all
        # combinations of test_data[i] and train_data[ı]
        combineddata[
            (i*train_data.shape[0]):((i+1)*train_data.shape[0]),
            train_data.shape[1]:(train_data.shape[1]*2)
        ] = np.tile(test_data[i], (train_data.shape[0], 1))

        # 3. To the right 2. of R test we add columns that describe what
        # class/regression output is the true label/regression output of
        # the training datapoint O_train
        combineddata[(i * train_data.shape[0]):((i + 1) * train_data.shape[0]),
        (train_data.shape[1] * 2):(train_data.shape[1] * 2)+test_target.shape[1]] = train_target

        # 4. To the right of 23. O_train we set to the true label/reg
        # output of  the test_target[i]
        combineddata[
            (i * train_data.shape[0]):((i + 1) * train_data.shape[0]),
            (train_data.shape[1] * 2) + test_target.shape[1]:
            (train_data.shape[1] * 2)+(2*test_target.shape[1])
        ] = np.tile(test_target[i], (train_data.shape[0], 1))

    # Sequential is just used by gabel..
    if type(model) == tensorflow.keras.models.Sequential or gabel:
        pred_vec = model.predict(combineddata[0:combineddata.shape[0], 0:train_data.shape[1] * 2], batch_size=batch_size)
    elif type(model) == tensorflow.keras.models.Model:
        pred_vec = model.predict([combineddata[:, 0:train_data.shape[1]],
                                  combineddata[:,train_data.shape[1]:2*train_data.shape[1]]], batch_size=batch_size)
    elif isinstance(model,torch.nn.Module):
        pred_vec = model(torch.from_numpy(combineddata[:, 0:train_data.shape[1]]).to(torch.float32),
                         torch.from_numpy(combineddata[:,train_data.shape[1]:2*train_data.shape[1]]).to(torch.float32))
        pred_vec = pred_vec.detach().numpy()

    else:
        pred_vec = model.predict(combineddata[0:combineddata.shape[0], 0:train_data.shape[1] * 2])

    if type(pred_vec) is list:
        pred_vec = pred_vec[0]
    if len(pred_vec.shape) == 1:
        pred_vec = pred_vec.reshape(pred_vec.shape[0],-1)

    longtable = np.concatenate(
        (combineddata[:, 0:train_data.shape[1]],
         combineddata[:,train_data.shape[1]:(train_data.shape[1]*2)]
         )
    )
    tot_maximums = np.max(longtable, axis=0)
    tot_minimums = np.min(longtable, axis=0)
    errdict = {}
    #printhisto(pred_vec)
    #for debugoutput we add a colummn that sums the training data at that row (too see to what degree the magnitude of that vector affects the network output)

    #    = combineddata[i,0:train_data.shape[1]].sum()
    # for i in range(0,train_data.shape[0]*test_data.shape[0]):
    #     combineddata[i,(train_data.shape[1] * 2)+(2*test_target.shape[1])+1:(train_data.shape[1] * 2)+(2*test_target.shape[1])+2] \
    #          = global_sim_def_lin(combineddata[i,0:train_data.shape[1]],combineddata[i,train_data.shape[1]:(train_data.shape[1]*2)],tot_maximums,tot_minimums,False,None)
    if pred_vec.shape[1] is not 1:
        pred_vec = choprascore(pred_vec)
    #finally we add the last column which is the networks evaluation of the similiarity of the pair of datapoints at row i.
    combineddata[:,(train_data.shape[1] * 2)+(2*test_target.shape[1]):(train_data.shape[1] * 2)+(2*test_target.shape[1])+1] = pred_vec
    #sortecombineddata = combineddata[combineddata[:,combineddata.shape[1]-1].argsort()]
    #now we have the similarities, we need to sort on the network output
    errdistvec = dict()
    truedistvec = dict()
    errvec = list()
    heh = 0
    for i in range(0, test_data.shape[0]):
        #first the select a subset of the array that corresponds to test_data[i] and all of train_data combined
        subset = combineddata[(i*train_data.shape[0]):((i+1)*train_data.shape[0]), 0:(train_data.shape[1]*2)+(test_target.shape[1]*2)+2]
        #then we sort it so that the smallest network output (most similar) is at the top of the array subset
        sortedsubset = subset[subset[:, subset.shape[1] - 2].argsort()]
        #extract the true labels/reg output of those rows for train and test respectively, compare if they are indeed the same..
        index = getAscOrDescIndex(sortedsubset.shape[0], not distance)
        err = evalSortedsubset(sortedsubset, index,
                               train_data.shape[1],
                               test_target.shape[1])
        true_target_value = sortedsubset[index, (train_data.shape[1] * 2) + test_target.shape[1]:(train_data.shape[1] * 2) + (2 * test_target.shape[1])]
        tv = str(np.where(true_target_value == 1)[0][0])
        if err is False:
            if tv in errdistvec:
                errdistvec[tv] += 1
            else:
                errdistvec[tv] = 0
            heh += 1
            # print(f"boop {heh}")
        else:
            if tv in truedistvec:
                truedistvec[tv] += 1
            else:
                truedistvec[tv] = 0
        errvec.append(err)
        #if np.equal(np.rint(this_train_target),np.rint(this_test_target)).all():
        #    errvec.append(1.0)
        #else:
        #    errvec.append(0)

    return errvec, errdistvec, truedistvec, combineddata


import operator


def topKey(dict):
    sortedict = sorted(dict.items(), key=operator.itemgetter(1),reverse=True)
    ret1 = sortedict[0]
    ret2 = None
    if len(sortedict) > 1:
        ret2 = sortedict[1]
    return ret1, ret2


def targetequal(targets, index1, index2):
    if targets[index1] is targets[index2]:
        return True
    return False


def evalsquare(model, features, targets):
    #distmatrix = np.zeros((features.shape[0],features.shape[0]))
    res = []
    import time
    totaltime = 0
    N = features.shape[0]
    for i in range(0,N):
        distrow = []
        samebiggest = 0
        notsamesmallest = 1000
        start = time.time()
        for i2 in range(0, features.shape[0]):
            dist = model.predict([[features[i,:]],[features[i2,:]]])[0]
            same = targetequal(targets, i, i2)
            if same and dist > samebiggest:
                samebiggest = dist
            elif not same and dist < notsamesmallest:
                notsamesmallest = dist
            distrow.append((dist, same))
        end = time.time()
        res.append(samebiggest < notsamesmallest)
        p = float(i)/float(N)
        delta = end-start
        totaltime = totaltime + delta
        perstep = totaltime / (i+1)
        timeleft = perstep * (N-(i+1))
        print(f"{p} % ({i} of {N}) done, time step: {delta} time left {timeleft}", end='\r')

    return np.sum(res)/len(res)


def sillouettescore(model, features, targets):
    distmatrix = np.zeros((features.shape[0],features.shape[0]))
    import time
    totaltime = 0
    N = features.shape[0]
    for i in range(0,N):
        start = time.time()
        for i2 in range(0, features.shape[0]):
            dist = model.predict([[features[i,:]],[features[i2,:]]])[0]
            distmatrix[i][i2] = dist
        end = time.time()
        p = float(i)/float(N)
        delta = end-start
        totaltime = totaltime + delta
        perstep = totaltime / (i+1)
        timeleft = perstep * (N-(i+1))
        print(f"{p:.2f} % ({i} of {N}) done, time step: {delta:.2f} time left {timeleft:.2f}", end='\r')
    from sklearn.metrics import silhouette_score
    start = time.time()
    sscore = silhouette_score(distmatrix, metric="precomputed", labels=targets)
    end = time.time()
    delta = end-start
    print(f"used {delta} time for sillouettescore")
    return sscore

def evalSortedsubset(sortedsubset, index, datahape, targetshape):
    sortedsubsettruth = sortedsubset[:,-2] == sortedsubset[index,-2]
    true_target_value = sortedsubset[index, (datahape * 2) + targetshape:(datahape * 2) + (2 * targetshape)]
    true_target_reverse = np.argmax(true_target_value,axis=0)
    subsetsubset = sortedsubset[sortedsubsettruth,:]
    unique, counts = np.unique(sortedsubsettruth,return_counts=True)
    unique_counts = dict(zip(unique, counts))
    if unique_counts[True] > 1:
        reverse = np.argmax(subsetsubset[:, (datahape * 2):(datahape * 2) + targetshape],axis=1)
        uniquer, countsr = np.unique(reverse, return_counts=True)
        unique_countsr = dict(zip(uniquer, countsr))
        res1,res2 = topKey(unique_countsr)
        if (res2 is not None) and (res1[1] == res2[1]): # the two top classes are equally numbered..
            return bool(random.getrandbits(1))
        elif res1[0] == true_target_reverse:
            return True
        else:
            return False
        #if unique_countsr[tru]
        #hits = subsetsubset[:, (datahape * 2):(datahape * 2) + targetshape] == true_target_value
        #hits = np.all(hits,axis=1)
        #unique2, counts2 = np.unique(hits, return_counts=True)
        #unique_counts2 = dict(zip(unique2, counts2))
        # if True not in unique_counts2:
        #     return False
        #
        # #if unique_counts2[True] != subsetsubset.shape[0]:
        # #    print("stop")
        #
        # if False not in unique_counts2 or unique_counts2[True] >= unique_counts2[False]:
        #     return True
        # elif unique_counts2[True] < unique_counts2[False]:
        #     return False

    else:
        this_train_target = sortedsubset[index, (datahape * 2):(datahape * 2) + targetshape]
        return all(np.equal(this_train_target,true_target_value))


def _eval_dual_ann_big(model, test_data, test_target, train_data,
    train_target, batch_size, anynominal=False, colmap=None,gabel=False):
    errvec = list()
    if test_data.shape[0] != test_target.shape[0]:
        return

    #we stuff everything into this big array, data, Y_pred and Y_true_1 and Y_true2
    #combinedata = np.zeros((train_data.shape[0]*test_data.shape[0],(train_data.shape[1]*2)+(test_target.shape[1]*2)+2))
    for i in range(0,test_data.shape[0]):
        thistest = np.zeros((train_data.shape[0],(train_data.shape[1]*2)+1))
        thistest[:,0:train_data.shape[1]] = np.tile(test_data[i],(train_data.shape[0],1))
        thistest[:,train_data.shape[1]:train_data.shape[1]*2] = train_data
        #for this sub-rectangle of the array we set the top-left corner to be the training data R_train


        # Sequential is just used by gabel..
        if type(model) == tensorflow.keras.models.Sequential or gabel:
            pred_vec = model.predict(thistest[:, 0:train_data.shape[1] * 2],
                                     batch_size=batch_size)
        elif type(model) == tensorflow.keras.models.Model:
            pred_vec = model.predict([thistest[:, 0:train_data.shape[1]],
                                      thistest[:, train_data.shape[1]:2*train_data.shape[1]]],
                                      batch_size=thistest.shape[0])
        else:
            pred_vec = model.predict(thistest[:, 0:train_data.shape[1] * 2])

        if type(pred_vec) is list:
            pred_vec = pred_vec[0]
        if len(pred_vec.shape) == 1:
            pred_vec = pred_vec.reshape(pred_vec.shape[0],-1)


        if pred_vec.shape[1] is not 1:
            pred_vec = choprascore(pred_vec)
        #finally we add the last column which is the networks evaluation of the similiarity of the pair of datapoints at row i.
        thistest[:,(train_data.shape[1] * 2):(train_data.shape[1] * 2)+1] = pred_vec
        #sortecombineddata = combinedata[combinedata[:,combinedata.shape[1]-1].argsort()]
        #now we have the similarities, we need to sort on the network output


        sortedsubset = thistest[thistest[:, -1].argsort()]
        #extract the true labels/reg output of those rows for train and test respectively, compare if they are indeed the same..
        index = getAscOrDescIndex(sortedsubset.shape[0], gabel)
        this_train_target = sortedsubset[index,
                                         train_data.shape[1]:
                                         train_data.shape[1] * 2]
        this_test_target = sortedsubset[index, 0:train_data.shape[1]]

        if np.equal(np.rint(this_train_target),np.rint(this_test_target)).all():
            errvec.append(1.0)
        else:
            errvec.append(0)

    return errvec

def printhisto(targetmatrix):
    df = pd.DataFrame(data=targetmatrix)
    counts = df.iloc[:,targetmatrix.shape[1]-1].value_counts()
    print(counts)


def test(a, b):
    # Unfortunatly you need to use structured arrays:
    sorted = np.ascontiguousarray(a).view([('', a.dtype)] * a.shape[-1]).ravel()

    # Actually at this point, you can also use np.in1d, if you already have many b
    # then that is even better.

    sorted.sort()

    b_comp = np.ascontiguousarray(b).view(sorted.dtype)
    ind = sorted.searchsorted(b_comp)

    result = sorted[ind] == b_comp
    return result


def test2(a, b):
    matches = list()
    for line in a:
        if any(np.equal(b,line).all(1)):
            matches.append(line)

    return matches
