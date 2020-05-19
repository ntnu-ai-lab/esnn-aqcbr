import numpy as np
import pandas as pd

def stat(df, epoch, method):
    filt = df[df.label.str.contains('val') & (df.epoch == epoch)]
    filt = filt[filt.label.str.contains(method)]
    nparr = np.asarray(filt.loss.values)
    return np.mean(nparr), np.std(nparr)

def aucstat(df, epoch, method):
    filt = df[df.label.str.contains('auc') & (df.epoch == epoch)]
    filt = filt[filt.label.str.contains(method)]
    nparr = np.asarray(filt.loss.values)
    return np.mean(nparr), np.std(nparr)

def ratiostat(df, method):
    filt = df[df.method.str.contains(method)]
    err = filt[filt.type.str.contains("err")]
    true = filt[filt.type.str.contains("true")]
    errarr = np.asarray(err.value.values)
    truearr = np.asarray(true.value.values)
    return (np.mean(errarr), np.std(errarr)),(np.mean(truearr), np.std(truearr))
