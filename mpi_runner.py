from mpi4py import MPI
from models.model_utils import makeANNModel
from utils.keras_utils import set_keras_growth
from utils.runutils import runalldatasetsMPI, getArgs
from utils.storage_utils import createdir, writejson
from datetime import datetime
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import sys
import numpy as np
import pandas as pd
import random
import os
from utils.esnnlogging import LoggingUtility


def main(mpirank, mpisize, mpicomm):
    args = getArgs()
    logger = LoggingUtility.getInstance("mpirunner_main")

    if args.seed is None:
        seed = random.randrange(sys.maxsize)
        args.seed = seed
        print(f"generating new random seed:{seed}")
    random.seed(args.seed)
    datasetlist = args.datasets

    results = {}
    runlist = args.methods

    if "," in args.gpu:
        gpus = args.gpu.split(",")
        mygpu = gpus[mpirank % 2]
        set_keras_growth(int(mygpu))
    else:
        set_keras_growth(args.gpu)

    dataset_results = dict()
    prefix = "runner"
    pid = str(os.getpid())
    if args.prefix is not None:
        prefix = args.prefix
    rootpath = prefix + "_" + pid + o\
        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    if mpirank == 0:
        logger.info(f"in mpirank0 my pid is {pid} the filestring is {rootpath}")
        createdir(rootpath)
        writejson(f"{rootpath}/settings.json", sys.argv[1:])

    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    nresults = list()
    alphalist = [
        0.8
    ]  # this code does not iterate over alpha, see mpi_deesweighting.py
    for i in range(0, args.n):
        dataset_results = runalldatasetsMPI(args,
                                            callbacks,
                                            datasetlist,
                                            mpicomm,
                                            mpirank,
                                            rootpath,
                                            runlist,
                                            alphalist,
                                            n=i,
                                            printcvresults=args.cvsummary,
                                            printcv=args.printcv,
                                            doevaluation=args.doevaluation)
        nresults.append(dataset_results)

        if mpirank == 0:
            writejson(f"{rootpath}/data.json", nresults)
            resdf = pd.DataFrame(results)
            resdf.to_csv(
                f"{rootpath}/results_{args.kfold}kfold_{args.epochs}epochs_{args.onehot}onehot.csv"
            )


if __name__ == "__main__":
    logger = LoggingUtility.getInstance("mpirunner_main")
    comm = MPI.COMM_WORLD
    mpisize = comm.Get_size()
    mpirank = comm.Get_rank()
    logger.info(f"in __main__ my mpirank is {mpirank}")
    main(mpirank, mpisize, comm)
