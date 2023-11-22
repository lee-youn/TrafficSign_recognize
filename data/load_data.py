import os

import h5py
import torch
import numpy as np


def load_data(N_=None, print_=False):
    # Data shape
    # train      (N,     3,48,48)
    # validation (N,     3,48,48)
    # test       (7766,  3,48,48)
    # N=none 일 경우 각 90601, 31063, 7766개

    f = os.path.join(os.getcwd(), "data", "dataset_ts_light_version.hdf5")

    with h5py.File(
        f,
        "r",
    ) as f:
        # Showing all keys in the HDF5 binary file
        # print(list(f.keys()))

        # Extracting saved arrays for training by appropriate keys
        # Saving them into new variables
        x_train = f["x_train"]  # HDF5 dataset
        y_train = f["y_train"]  # HDF5 dataset
        # Converting them into Numpy arrays
        x_train = np.array(x_train)  # Numpy arrays
        y_train = np.array(y_train)  # Numpy arrays

        # Extracting saved arrays for validation by appropriate keys
        # Saving them into new variables
        x_validation = f["x_validation"]  # HDF5 dataset
        y_validation = f["y_validation"]  # HDF5 dataset
        # Converting them into Numpy arrays
        x_validation = np.array(x_validation)  # Numpy arrays
        y_validation = np.array(y_validation)  # Numpy arrays

        # Extracting saved arrays for testing by appropriate keys
        # Saving them into new variables
        x_test = f["x_test"]  # HDF5 dataset
        y_test = f["y_test"]  # HDF5 dataset
        # Converting them into Numpy arrays
        x_test = np.array(x_test)  # Numpy arrays
        y_test = np.array(y_test)  # Numpy arrays

        # Cutting
        x_train = x_train[:N_]
        y_train = y_train[:N_]
        x_validation = x_validation[:N_]
        y_validation = y_validation[:N_]

        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_validation = np.transpose(x_validation, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))

        # Convert to tensor
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_validation = torch.from_numpy(x_validation)
        y_validation = torch.from_numpy(y_validation)
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)

    # Check point
    # Showing shapes of arrays after splitting
    if print_:
        print(x_train.shape)
        print(y_train.shape)

        print(x_validation.shape)
        print(y_validation.shape)

        print(x_test.shape)
        print(y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test
