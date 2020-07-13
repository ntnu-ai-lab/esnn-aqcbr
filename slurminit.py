import argparse as args
from test_tube.hpc import SlurmCluster
from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import numpy as np
# hyperparameters is a test-tube hyper params object
# see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
# hyperparams = args.parse()
from utils.newtorcheval import (ChopraTorchEvaler, GabelTorchEvaler,
    TorchEvaler, runevaler)
from models.esnn.pytorch_trainer import ESNNSystem
from models.evalfunctions import eval_dual_ann, eval_gabel_ann
parser = HyperOptArgumentParser(strategy='random_search')

# init cluster
# cluster = SlurmCluster(
#     hyperparam_optimizer=hyperparams,
#     log_path='/path/to/log/results/to',
#     python_cmd='python3'
# )

# # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
# cluster.notify_job_status(email='bjornmm@ntnu.no', on_done=True, on_fail=True)

# # set the job options. In this instance, we'll run 20 different models
# # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
# cluster.per_experiment_nb_gpus = 1
# cluster.per_experiment_nb_nodes = 1

# # we'll request 50GB of memory per node
# cluster.memory_mb_per_node = 50000

# # set a walltime of 10 minues
# cluster.job_time = '10:00'

# # 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
# # you must provide your own loading and saving function which the cluster object will call
# cluster.minutes_to_checkpoint_before_walltime = 1


# def main(hparams, cluster, return_dict):
#     # do your own generic training code here...
#     # init model
#     model = esnn(hparams)

#     # set the load and save fxs
#     cluster.set_checkpoint_save_function(fx, {})
#     cluster.set_checkpoint_load_function(fx, {})

    # train ...

from utils.pdutils import stat


def train(hparams, *args):
    """Train your awesome model.
    :param hparams: The arguments to run the model with.
    """
    # Initialize experiments and track all the hyperparameters
    # if hparams.disease_model:
    #     save_model_path = hparams.save_model_dir+'/disease'
    # else:
    #     save_model_path = hparams.save_model_dir+'/synthetic'
    # Set seeds
    SEED = hparams.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(hparams)
    print(args)
    exp = Experiment(
        name=hparams.test_tube_exp_name,
        # Location to save the metrics.
        save_dir=hparams.log_path,
        autosave=False,
    )
    exp.argparse(hparams)
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=save_model_path+'/'+hparams.cage_nr +
    #     '/version_'+str(cluster.hpc_exp_number)+'/checkpoints',
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )
    # # Pretend to train.
    # x = torch.rand((1, hparams.x_val))
    # for train_step in range(0, 100):
    #     y = torch.rand((hparams.x_val, 1))
    #     out = x.mm(y)
    #     exp.log({'fake_err': out.item()})

    dsl, \
        trainedmodels,\
        validatedmodels,\
        losses,\
        lossdf,\
        knnres = runevaler("opsitu", hparams.epochs, [ESNNSystem],
                           [TorchEvaler], [eval_dual_ann],
                           networklayers=[hparams.c_layers, hparams.g_layers],
                           lrs=[hparams.lr],
                           dropoutrates=[hparams.dropout],
                           validate_on_k=10, n=1,
                           filenamepostfixes=["esnn"])
    stats = stat(lossdf, hparams.epochs, "esnn")
    print(f"type : {type(stats)}")
    print(f"innertype : {type(stats[0])}")
    print(f"stats : {stats}")
    print(f"stats0 : {stats[0]}")
    exp.log({'loss': stats[0]})
    #exp.log('tng_err': tng_err)
    #exp.log({"loss", stats[0]})
    # Save exp when .
    exp.save()


def str_to_str_tuple(str_value):
    """Convert string to tuple of strings for HyperOptArgumentParser with multiple inputs"""
    if isinstance(str_value, tuple):
        return str_value

    # Remove spaces
    str_value = str_value.replace(' ', '')

    # Remove braces/brackets/parenthesis on outside
    str_value = str_value.strip('[](){}')

    # Split by comma
    return tuple(str_value.split(','))

def str_to_int_tuple(str_value):
    """Convert string to tuple of ints for HyperOptArgumentParser with multiple inputs"""

    # Create a str tuple, then convert to int
    str_tuple = str_to_str_tuple(str_value)
    return tuple([int(str_val) for str_val in str_tuple])


# for hparam_trial in hparams.trials(20):
#     train_network(hparam_trial)
def esnnparms(parser):
    # parser.opt_range('--neurons', default=50, type=int,
    #              tunable=True, low=100, high=800, 
    #              nb_samples=10, log_base=10)
    # [[40,6,3],[2]]
    parser.opt_list('--c_layers', default='40,6,3', type=str_to_int_tuple,
                    tunable=True, options=['40,6,3', '20,3,2', "80,12,6"])
    parser.opt_list('--g_layers', default=2, type=int,
                    tunable=True, options=[6, 4, 2])
    parser.opt_range('--epochs', default=500, type=int,
                     tunable=False, low=50, high=2000,
                     nb_samples=10, log_base=10)
    parser.opt_range('--lr', default=0.08, type=float,
                     tunable=True, low=0.02, high=0.2,
                     nb_samples=10, log_base=10)
    parser.opt_range('--dropout', default=0.005, type=float,
                     tunable=True, low=0.002, high=0.01,
                     nb_samples=10, log_base=10)
    return parser


# module purge
# module load icc/2018.1.163-GCC-6.4.0-2.28
# module load OpenMPI/2.1.2
# module load goolfc/2017b
# module load TensorFlow/1.7.0-Python-3.6.3
# MPIRUNFILE=/share/apps/software/Compiler/intel/2018.1.163-GCC-6.4.0-2.28/OpenMPI/2.1.2/bin/mpirun


if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    hyper_parser = HyperOptArgumentParser(strategy='random_search')
    hyper_parser.add_argument('--test_tube_exp_name', default='my_test')
    hyper_parser.add_argument('--log_path', default='.')
    hyper_parser.add_argument('--seed', default=42, type=int)
    hyper_parser = esnnparms(hyper_parser)
    hyperparams = hyper_parser.parse_args()
    print(hyperparams)
    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python3',
        enable_log_err=True,
        enable_log_out=True
        #test_tube_exp_name=hyperparams.test_tube_exp_name
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='bjornmm@ntnu.no', on_done=False, on_fail=True)

    # SLURM Module to load.

    # Add commands to the non-SLURM portion.
    

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    # cluster.add_slurm_cmd(
    #    cmd='cpus-per-task', value='1', comment='CPUS per task.')

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    # we'll request 10GB of memory per node
    cluster.memory_mb_per_node = 200000

    # set a walltime
    cluster.job_time = '40:00:00'
    cluster.minutes_to_checkpoint_before_walltime = 1
    # cluster.gpu_type = 'gpu:V100:2'
    cluster.add_slurm_cmd(cmd='partition', value='GPUQ',
                          comment='what partition to use')
    cluster.add_slurm_cmd(cmd='account', value='ie-idi',
                          comment='what account to use')
    cluster.add_command('module purge')
    print(cluster.log_path)
    
    cluster.load_modules([
        #'icc/2018.1.163-GCC-6.4.0-2.28',
        'GCC/8.3.0',
        'CUDA/10.1.243',
        'OpenMPI/3.1.4',
    ])
    cluster.add_command(
        'MPIRUNFILE=/share/apps/software/Compiler/intel/2018.1.163-GCC-6.4.0-2.28/OpenMPI/2.1.2/bin/mpirun')
    cluster.add_command('. /home/bjornmm/anaconda3/etc/profile.d/conda.sh')
    cluster.add_command('conda activate')
    cluster.add_command('conda activate bmaitf2')
    cluster.add_command(
        'PYTHON=$HOME/anaconda3/envs/bmtfai2/bin/python')
    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        train,
        # Number of hyperparameter combinations to search:
        nb_trials=24,
        # This is what will display in the slurm queue:
        job_name='first_tt_job')
# run the models on the cluster

# cluster.optimize_parallel_cluster_gpu(main, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')
