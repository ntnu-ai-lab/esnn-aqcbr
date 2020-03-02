import numpy as np
import pandas as pd

def stat(df, epoch, method):
    filt = df[df.label.str.contains('val') & (df.epoch == epoch)]
    filt = filt[filt.label.str.contains(method)]
    nparr = np.asarray(filt.loss.values)
    return np.mean(nparr), np.std(nparr)
