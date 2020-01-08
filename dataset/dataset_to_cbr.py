from mycbrwrapper.rest import getRequest
from mycbrwrapper.concepts import Concepts
from dataset.dataset import *
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import json

defaulthost = "localhost:8080"

def getDoubleAttributeParamterJSON(min, max, solution):
    return """
    {{
    "type": "Double",
    "min": "{}",
    "max": "{}",
    "solution": "{}"
    }}
    """.format(min, max, solution)


def getStringAttributeParamterJSON(solution):
    return """
    {{
    "type": "String",
    "solution": "{}"
    }}
    """.format(solution)


def getStringAttributeParamterWithValuesJSON(allowedvalues, solution):
    return """
    {{
    "type": "Symbol",
    "allowedValues": [{}],
    "solution": "{}"
    }}
    """.format(allowedvalues, solution)


def getDoubleParameters(imin, imax, solution):
    return {
        "attributeJSON": getDoubleAttributeParamterJSON(imin, imax, solution)
    }


def getStringParameters(isolution):
    return {"attributeJSON": getStringAttributeParamterJSON(solution=isolution)}


def findColFromValue(colmap, value):
    for key, dict in colmap.items():
        if "possible_values" in dict and value in dict["possible_values"]:
            return key
    return None


def findDatasetInfo(datasetInfo, name):
    for row in datasetInfo["cols"]:
        if row["name"] is name:
            return row
    return None

def sendDf(df, c, casebase):
    jsonstr = df.to_json(orient="records")
    jsono = json.loads(jsonstr)
    datadict = {}
    datadict["cases"] = jsono
    c.addInstances(datadict, casebase)

def fromDatasetToCBR(dataset, sklearndataset, colmap, host=defaulthost, concepts=None, instances=10):
    #print(dataset.df)

    #sklearndataset, colmap = fromDataSetToSKLearn(dataset)

    datadf = sklearndataset.getDataFrame()
    sklearncols = list(datadf)
    if concepts is None:
        concepts = Concepts(host)
    c = concepts.addConcept(dataset.name)
    # create the model in the CBR system
    # for col in dataset.getTypes():
    #     #print("columns \"{}\"".format(dataset.df.columns))
    #     colname = col["name"]
    #     #print("coltype: {}".format(col["type"]))
    #     if colname in nominalcols:
    #
    #     elif (col["type"] is "str" or "nominal") and (isinstance(col["type"],str)):
    #         #print("creating new str attribute from coltype: {}".format(col["type"]))
    #         c.addAttribute(colname,getStringParameters())
    #     else:
    #         #print("creating new double attribute from coltype: {}".format(col["type"]))
    #         cmin = dataset.getMinForCol(colname)
    #         cmax = dataset.getMaxForCol(colname)
    #         c.addAttribute(colname,
    #                        getDoubleParameters(cmin, cmax))
    # for colname,value_list in colmap.items():
    #     #print("columns \"{}\"".format(dataset.df.columns))
    #     #print("coltype: {}".format(col["type"]))
    #
    #     if value_list["type"] is  "nominal":
    #         #print("creating new str attribute from coltype: {}".format(col["type"]))
    #         c.addAttribute(colname,getStringParameters())
    #     else:
    #         #print("creating new double attribute from coltype: {}".format(col["type"]))
    #         cmin = dataset.getMinForCol(colname)
    #         cmax = dataset.getMaxForCol(colname)
    #         c.addAttribute(colname,
    #                        getDoubleParameters(cmin, cmax))
    for col in sklearncols:
        if col not in colmap:  #it has to be a binarized new-column
            #print(f"sending paramstring: {paramstr}")
            originalColName = findColFromValue(colmap, col)
            datasetInfoRow = findDatasetInfo(dataset.datasetInfo,
                                             originalColName)
            classCol = datasetInfoRow["class"]
            paramstr = getDoubleAttributeParamterJSON(0, 1, classCol)
            c.addAttribute(col, paramstr)
        elif colmap[col]["type"] == "number":
            #cmin = dataset.getMinForCol(col)
            #cmax = dataset.getMaxForCol(col)
            datasetInfoRow = findDatasetInfo(dataset.datasetInfo, col)
            classCol = datasetInfoRow["class"]
            c.addAttribute(col,
                           getDoubleAttributeParamterJSON(0, 1.0, classCol))
        elif col in colmap and colmap[col]["type"] == "nominal":
            cmin = datadf[col].min()
            cmax = datadf[col].max()
            datasetInfoRow = findDatasetInfo(dataset.datasetInfo, col)
            classCol = datasetInfoRow["class"]
            paramstr = getDoubleAttributeParamterJSON(cmin, cmax, classCol)
            c.addAttribute(col, paramstr)

    # create the instances that fit into the model

    c.addCaseBase("mydefaultCB")
    df = sklearndataset.getDataFrame()
    df = df.sample(n=instances)
    if instances > 20:
        print("doing piecewise")
        send = df.head(10)
        while len(send) != 0:
            sendDf(df, c, "mydefaultCB")
            df = df.iloc[len(send):]
            send = df.head(10)
    else:
        sendDf(df, c, "mydefaultCB")
    # for case in jsono:
    #     //onlycase = case.copy()
    #     //onlycase.pop("caseID", None)
    #     tempdict = {}
    #
    #
    #     tempdict["case"] = case
    #    c.addInstance(case["caseID"],json.dumps(case),"mydefaultCB")
    counter = 0
    # for row in jsono:
    #     counter += 1
    #     row["caseID"] = f"{prefix}{counter}"

    return concepts, c
