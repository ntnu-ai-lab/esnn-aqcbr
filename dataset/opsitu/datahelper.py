"""
This function goes through a dataframe, finds a failure data point D -
then makes a new datapoint based on D with a random higher windspeed but
with the same directoin pertubated with +/- 1 degree
"""
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def balancetraining(training_data, training_target, removecovarea=True):
    """This function finds random failures, augments them and adds them
    to the training data so that it becomes balanced.

    Args:
       training_data the datamatrix
       training_target the labels of the datamatrix


    Returns: a balanced datamatrix and the corresponding labels

    """

    # wind_speed wind_from_direction wind_effect sdistance
    wind_speed_ind = 0
    wind_dir_ind = 1
    positive_target = np.asarray([0, 1])
    failure_target = np.asarray([1, 0])
    # here we get the first element to get the indices, but since we compare to another array
    # we get the indices doubled, thus subsample every second element from the zeroth element..
    positives = np.where(training_target[:,:] == positive_target)[0][0::2]

    failures = np.where(training_target[:,:] == failure_target)[0][0::2]
    failure_training_data = training_data[failures]
    #slowfailures = np.where(failure_training_data[:,wind_speed_ind] <= 0.9)[0][0::2]

    #delete the slow data.. this is noise
    #training_data = np.delete(training_data, slowfailures, axis=0)
    #training_target = np.delete(training_target, slowfailures, axis=0)

    failures = np.where(training_target[:,:] == failure_target)[0][0::2]
    ratio = failures.shape[0]/training_data.shape[0]
    print(f"failures: {failures.shape[0]} ratio: {ratio}")

    failure_training_data = training_data[failures]
    fastfailures = np.where(failure_training_data[:,wind_speed_ind] > 0.8)[0][0::2]

    # failures = failures & slowdata

    failuresize = fastfailures.shape[0]
    positivesize = positives.shape[0]

    togenerate = positivesize - failures.shape[0]

    for i in range(0, togenerate):
        failureindex = fastfailures[int(random.uniform(0, 1.0) * failuresize)]
        newfailure = np.copy(failure_training_data[failureindex])
        # add 1 to 10% to wind speed
        newfailure[wind_speed_ind] = newfailure[wind_speed_ind] \
            * random.uniform(1.01, 1.1)
        # change wind direction by +/- 5 degrees
        newdir = newfailure[wind_dir_ind] \
            + random.uniform(-1.0136, 1.0136)
        # no more than 365 and less than 0
        newdir = max(min(newdir, 0), 1)
        newfailure[wind_dir_ind] = newdir
        training_data = np.concatenate((training_data, [newfailure]))
        training_target = np.concatenate((training_target, [failure_target]))

    # we need to rescale the array since we could have
    # added "out of scope" wind speeds
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    # need to reshuffle arrays ..
    assert len(training_data) == len(training_target)
    p = np.random.permutation(len(training_data))
    return training_data[p], training_target[p]
