import argparse as args
from test_tube.hpc import SlurmCluster

# hyperparameters is a test-tube hyper params object
# see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
hyperparams = args.parse()

# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/path/to/log/results/to',
    python_cmd='python3'
)

# let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
cluster.notify_job_status(email='bjornmm@ntnu.no', on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1

# we'll request 50GB of memory per node
cluster.memory_mb_per_node = 50000

# set a walltime of 10 minues
cluster.job_time = '10:00'

# 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
# you must provide your own loading and saving function which the cluster object will call
cluster.minutes_to_checkpoint_before_walltime = 1


def main(hparams, cluster, return_dict):
    # do your own generic training code here...
    # init model
    model = esnn(hparams)

    # set the load and save fxs
    cluster.set_checkpoint_save_function(fx, {})
    cluster.set_checkpoint_load_function(fx, {})

    # train ...


# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(main, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')
