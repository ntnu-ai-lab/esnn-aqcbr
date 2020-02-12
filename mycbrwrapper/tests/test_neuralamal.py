from mycbrwrapper.rest import getRequest
from mycbrwrapper.tests.test_dataset_to_cbr import ConvertDataSetTest, defaulthost
from models.esnn import keras
from dataset.makeTrainingData import makeDualSharedArchData
from utils.runutils import optimizer_dict
import logging
import os
import hashlib
import warnings
__name__ = "test_base"

defaulthost = "localhost:8080"
"""
The model of the case base for the unit tests are simple
id,name,doubleattr1,doubleattr2
"""


class NeuralSimTest(ConvertDataSetTest):

    @classmethod
    def setUpClass(cls):
        print("nnnnn")
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings(action="ignore", message="unclosed",
                                category=ResourceWarning)
        # s.config['keep_alive'] = False

    @classmethod
    def tearDownClass(cls):
        print("in tearDownClass")

    def __init__(self, *args, **kwargs):
        super(NeuralSimTest, self).__init__(*args, **kwargs)

    @staticmethod
    def md5(fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def makemodel(dsl, stratified_fold_generator, datasetname, rootpath, epochs):
        data = dsl.getFeatures()
        target = dsl.getTargets()
        train, test = next(stratified_fold_generator)
        test_data = data[test]
        test_target = target[test]

        train_data = data[train]
        train_target = target[train]



        model, hist, \
            ret_callbacks,\
            embedding_model = keras(o_X=test_data, o_Y=test_target,
                                    X=train_data, Y=train_target,
                                    regression=dsl.isregression,
                                    shuffle=True,
                                    batch_size=200,
                                    epochs=epochs,
                                    optimizer=optimizer_dict["rprop"],
                                    onehot=True,
                                    multigpu=False,
                                    callbacks=["tensorboard"],
                                    datasetname=datasetname,
                                    networklayers=[13, 13],
                                    rootdir=rootpath,
                                    alpha=0.2,
                                    makeTrainingData=makeDualSharedArchData)
        return model, embedding_model, hist


    @staticmethod
    def uploadNeuralSim(filename, h5md5, jsonmd5, datasetname, epochs):
        d, dsl, colmap, stratified_fold_generator, concepts, concept = \
            ConvertDataSetTest.convert(datasetname)

        rootpath = "~/research/experiments/annSimilarity/mycbrwrapper/tests/"
        rootpath = os.path.expanduser(rootpath)

        # basefile = rootpath+f"esnnname-of-model-{epochs-1}"
        basefile = rootpath+f"esnn"
        h5file = basefile+".h5"
        jsonfile = basefile+".json"

        h5exists = os.path.exists(h5file)
        jsonexists = os.path.exists(jsonfile)

        newh5md5 = ""
        newjsonmd5 = ""
        if h5exists and jsonexists:
            newh5md5 = NeuralSimTest.md5(h5file)
            newjsonmd5 = NeuralSimTest.md5(jsonfile)

        # remake models if either of the files does not exist,
        # or either of the md5 checks fails
        if not (h5exists and jsonexists) \
           or not (newh5md5 == h5md5 and
                   newjsonmd5 == jsonmd5):
            print(f"h5 and json md5sums doesnt match: json " +
                  "{newjsonmd5} != {jsonmd5} and h5 {newh5md5} != {h5md5}")
            NeuralSimTest.makemodel(dsl, stratified_fold_generator,
                                    datasetname, rootpath, epochs)
        #NeuralSimTest.makemodel(dsl, stratified_fold_generator,
        #                        datasetname, rootpath, epochs)
        NeuralSimTest.sendNeuralAmal(concept, h5file, jsonfile)
        return concepts

    @staticmethod
    def sendNeuralAmal(concept, h5file, jsonfile):
        amalstring = "neuralamal"
        filesDict = {
            "h5":
            h5file,
            "json":
            jsonfile
        }
        concept.addNeuralAmalgamationFunction(amalstring, filesDict)

    def retrieval(self, filename, h5md5, jsonmd5):
        casebasename = "mydefaultCB"
        conceptname = "opsitu"
        amalname = "neuralamal"
        concepts = NeuralSimTest.uploadNeuralSim(filename, h5md5, jsonmd5, "opsitu", 20)
        concept = concepts.getConcept("opsitu")
        print(len(concept))
        instances = concept.instanceList()
        firstincstance = instances[0]
        print(f"instance 0 params: {firstincstance.instance_parameters}")
        api = getRequest(defaulthost)
        call = api.concepts(conceptname).casebases(casebasename)\
            .amalgamationFunctions(amalname).retrievalByCaseID
        print(call)
        result = call.GET(params={
             "caseID": conceptname+"-"+casebasename+"1"
        })
        print(result.json())

    def test_bad_retrieval(self):
        h5md5 = "a23bf84a38d7338483d1e53239ff501a"
        jsonmd5 = "dfbbe24395322a5817daec2b4a9caa26"
        filename = "esnn"
        self.retrieval(filename, h5md5, jsonmd5)

    def test_good_retrieval(self):
        h5md5 = "5c57e1ac6fdcbdcbd346789b3dd9450e"
        jsonmd5 = "50cee1c63f48306b867956c87b1dc0a7"
        filename = "esnn166"
        self.retrieval(filename, h5md5, jsonmd5)

    def test_just_retrieval(self):
        casebasename = "mydefaultCB"
        conceptname = "opsitu"
        amalname = "neuralamal"
        api = getRequest(defaulthost)
        call = api.concepts(conceptname).casebases(casebasename)\
            .amalgamationFunctions(amalname).retrievalByCaseID
        print(call)
        result = call.GET(params={
             "caseID": conceptname+"-"+casebasename+"1"
        })
        print(result.json())
