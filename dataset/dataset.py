import pandas as pd
import numpy as np
import collections
import pickle
import os
import json

import requests
import requests_cache
from sklearn import preprocessing

from dataset.datasetlist import datamap

requests_cache.install_cache('bmm_uci_ml')
from io import StringIO
from pandas_datareader.compat import bytes_to_str
import xlrd
from dataset.binarizer import MyLabelBinarizer

__name__ = "dataset"
"""
from https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
"""

def writedatasetobjects(datasetsoboject):
    with open("datasets.pickle", 'wb') as f:
        json.dump(datasetsoboject.__dict__, f)


def readdatasetobjects(picklefilename):
    obj = Datasets()
    if not os.path.isfile(picklefilename):
        print("pickefile: %s is not here" % picklefilename)
        return False
    with open(os.path.join(picklefilename), 'rb') as picklefile:
        obj.__dict__ = pickle.load(picklefile)
    return obj

def glass_preprocess(df):
    colname = "type_of_glass"
    dffff = pd.DataFrame(data={})
    #df.loc[:,colname] = df.loc[:,colname].apply(pd.to_numeric)
    down_query_index = df.query('type_of_glass == "1" or type_of_glass == "2" or type_of_glass == "3" or type_of_glass == "4"').index
    up_query_index = df.query('type_of_glass == "5" or type_of_glass == "6" or type_of_glass == "7"').index


    df.iloc[down_query_index,-1] = "0"
    df.iloc[up_query_index,-1] = "1"
    return df


def mnist_getInfo(df, datasetInfo):
    values = df.values
    features = values[:,:-1]
    targets = values[:,-1:]
    target_names = []
    isregression = False
    targetcols = [features.shape[1]]
    featurecols = [i for i in range(0,features.shape[1])]

    return target_names, isregression, featurecols, targetcols


def mnist_adapt(df, datasetInfo):
    values = df.values
    features = values[:,:-1]
    # features.shape 70000,784
    targets = values[:,-1:]
    mlb = MyLabelBinarizer()
    targets = mlb.fit_transform(targets)
    # features.shape 70000,10
    #targetcols = [i for i in range(features.shape[1],features.shape[1]+targets.shape[1])]
    targetcols = [i for i in range(features.shape[1],features.shape[1]+targets.shape[1])]
    featurecols = [i for i in range(0,features.shape[1])]
    return features, targets, featurecols.extend(targetcols)

def flatten_mnist(x):
    dim = x.shape[1] * x.shape[2]
    output = np.zeros((x.shape[0],dim))
    for i in range(0, x.shape[0]):
        output[i,:] = x[i].flatten()
    return output

def mnist_makedf(url, datasetInfo):

    from keras.utils.data_utils import get_file
    path = get_file("mnist.npz",
                    origin=url)

    f = np.load(path)
    #datasetInfo["classwidth"] = f['y_train'].shape[1]
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    flattentrain = flatten_mnist(x_train)
    flattentest = flatten_mnist(x_test)
    all_train = np.concatenate((flattentrain, y_train.reshape((y_train.shape[0],1))), axis=1)
    all_test = np.concatenate((flattentest, y_test.reshape((y_test.shape[0],1))), axis=1)
    all_data = np.concatenate((all_train, all_test), axis=0)

    feature_values = all_data[:, :-1].astype("float32")
    feature_values /= 255
    target_values = all_data[:,-1]
    target_values = target_values.reshape((target_values.shape[0],1))
    all_normalized_data = np.concatenate((feature_values,target_values),axis=1)
    retdf = pd.DataFrame(data=all_normalized_data)

    return retdf

def ttt_preprocess(df):

    columns = ["top_left", "top_middle", "top_right",
               "middle_left", "middle_middle", "middle_right",
               "bottom_left"]
    for col in columns:
        le = preprocessing.LabelEncoder()
        le.fit()

    return df



def ttt_postprocess(nparr,colmap):
    """
    from https://gist.github.com/mmmayo13/b52cc0e48aa10e1c0eac54e9989d36de
    """
    #nparr = np.delete(nparr, [0,3,6,9,12,15,18,21,24], axis=1)

    return nparr,colmap


# inspiration from https://github.com/pydata/pandas-datareader/blob/master/pandas_datareader/base.py
# and https://github.com/davidastephens/pandas-finance/blob/master/pandas_finance/api.py

def safeGetDF(url):
    httpsession = requests_cache.CachedSession(cache_name='bmm-uciml-cache', backend='sqlite')
    response = httpsession.get(url)
    if response.status_code != requests.codes.ok:
        return None
    text = response.content
    out = StringIO()
    if len(text) == 0:
        #service = self.__class__.__name__
        raise IOError("{} request returned no data; check URL for invalid ")
    if ".xls" in url:
        httpsession.close()
        return xlrd.open_workbook(file_contents=text)
        #out.write(text)
        #out.write(bytes_to_str(text,encoding="ISO-8859-1"))
    elif response.headers["content-type"] != "text/html":
        out.write(bytes_to_str(text,encoding="ISO-8859-1"))
    else:
        out.write(text)
    out.seek(0)
    httpsession.close()
    #decoded_data = out.decode("utf-8")
    return out

def convert(inp):
    if inp is "nominal" or inp is "ordinal" or inp:
        return "str"
    elif inp is "binary":
        return np.int32
    else:
        return inp

def makePosList(inp):
    counter = 0
    ret = []
    for el in inp:
        if el["type"] is "skip":
            counter += 1
        else:
            ret.append(counter)
            counter += 1
    return ret

class Dataset():
    def __init__(self, key):
        self.loadDataset(key)
        self.name = key

    def readFromUrl(self, url, sep=" "):
        self.df = pd.read_csv(url, sep, header=None)


    def loadDataset(self, key):
        self.datasetInfo = datamap[key]
        url = self.datasetInfo["dataUrl"]
        #print("loading dataset {}".format(key))
        if "makedf" in self.datasetInfo:
            self.df = self.datasetInfo["makedf"](url,self.datasetInfo)
            return
        usecols = makePosList(self.datasetInfo["cols"])
        colnames = [c["name"] for c in self.datasetInfo["cols"] if c["type"] is not "skip"]
        dtypes = [c["type"] for c in self.datasetInfo["cols"] if c["type"] is not "skip"]
        dtypes = map(convert,dtypes)
        dtypedict = dict(zip(colnames,dtypes))
        #print(dtypedict)

        try:
            #url = self.datasetInfo["backupUrl"]
            r = requests.get(url)
            #r.raise_for_status()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(f"{e}")
        except requests.exceptions.HTTPError as httperror:
            print(f"got error for dataset {key}")
            url = self.datasetInfo["backupUrl"]


        if "xls" in url:
            sheet_names = self.datasetInfo["sheet_name"]
            excelusecols = ",".join(colnames)
            if "usecols" in self.datasetInfo:
                possibledictofdf = pd.read_excel(safeGetDF(url),
                                                 header=self.datasetInfo["headers"],
                                                 sheet_name=sheet_names,
                                                 names=colnames,
                                                 dtype=dtypedict,
                                                 usecols=self.datasetInfo["usecols"],
                                                 engine="xlrd")
            else:
                possibledictofdf = pd.read_excel(safeGetDF(url),
                                                 header=self.datasetInfo["headers"],
                                                 sheet_name=sheet_names,
                                                 names=colnames,
                                                 dtype=dtypedict, engine="xlrd")
            if isinstance(sheet_names, collections.Sequence) and not isinstance(sheet_names, str):
                firstdf = possibledictofdf[sheet_names[0]]
                for sheet in sheet_names:
                    if sheet is not sheet_names[0]:
                        firstdf.append(possibledictofdf[sheet])
                self.df = firstdf[colnames]
            else:
                self.df = possibledictofdf[colnames]
        else: #this is a csv file
            na_values = None
            if "na_values" in self.datasetInfo:
                na_values = self.datasetInfo["na_values"]
            if "http" in url:
                self.df = pd.read_csv(safeGetDF(url), header=self.datasetInfo["headers"],
                                      names=colnames, dtype=dtypedict,
                                      usecols=usecols,
                                      sep=self.datasetInfo["sep"],
                                      na_values=na_values)
            else: #this is a local file..
                self.df = pd.read_csv(url, header=self.datasetInfo["headers"],
                                      names=colnames, dtype=dtypedict,
                                      usecols=usecols,
                                      sep=self.datasetInfo["sep"],
                                      na_values=na_values)
        if "pre_process" in self.datasetInfo:
            self.df = self.datasetInfo["pre_process"](self.df)
            #print(self.df)

    def getNumberOfRows(self):
        return self.df.shape[0]

    def getNumberOfAttributes(self):
        return self.df.shape[1]-1

    def getTypes(self):
        return self.datasetInfo["cols"]

    def getMaxForCol(self, col):
        return self.df[col].max()

    def getMinForCol(self, col):
        return self.df[col].min()


class Datasets():
    def __init__(self, filename=None):
        self.datasets = {}
        if filename is not None:
            if not os.path.isfile(filename):
                print(f"pickefile: {filename} is not here")
                return
            with open(os.path.join(filename), 'rb') as picklefile:
                self.__dict__ = pickle.load(picklefile)

    def getDataset(self, key):
        if key not in self.datasets:
            ds = Dataset(key)
            self.datasets[key] = ds
            writedatasetobjects(self)
            return ds
        return self.datasets[key]
