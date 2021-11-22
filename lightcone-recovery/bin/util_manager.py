import re
import sys
import h5py
import logging

import os
import typing
from typing import Optional, List
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn.preprocessing import normalize

def init_logger(f: str, 
                name: str) -> logging.Logger:
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = init_logger("test.log", __name__)

class UtilManager:
    """Manager class for miscellaneous I/O operations"""

    def __init__(self):
        self.dset_attrs = {}
        self.random_seeds = {}
        self.data = {}
        self.metadata = {}


    def load_all_data_from_h5(self, filepath):
        """Loads all data from h5 file, returns nothing. (Typically used just
        to observe the values in a dataset)"""
        self.filepath = filepath

        with h5py.File(filepath, "r") as hf:

            # Check we have the required datasets
            #datasets = list(hf.keys())
            #assert "wedge_filtered_brightness_temp_boxes" in datasets and \
            #       "brightness_temp_boxes" in datasets and \
            #       "predicted_brightness_temp_boxes" in datasets and \
            #       "ionized_boxes" in datasets and \
            #       "redshifts" in datasets, \
            #       "Failed to extract datasets from h5py file."

            for k in hf.keys():

                # AstroParams are stored in h5py groups
                if isinstance(hf[k], h5py.Group):
                    logger.debug(f"{k} is a group")
                    self.metadata[k] = {}
                    for k2 in hf[k].keys():
                        v = np.array(hf[k][k2], dtype=np.float32)
                        logger.debug(f"\t{k2} created in metadata.")
                        self.metadata[k][k2] = v

                # Lightcones are stored as h5py datasets
                if isinstance(hf[k], h5py.Dataset):
                    v = np.array(hf[k][:], dtype=np.float32)
                    assert np.isnan(np.sum(v)) == False
                    self.data[k] = v
            self.data["redshifts"].reshape(-1) 

            # Load metadata from h5 file
            for k, v in hf.attrs.items():
                self.dset_attrs[k] = v

        # Print success message
        print("\n----------\n")
        print(f"data loaded from {self.filepath}")
        print("Contents:")
        for k, v in self.data.items():
            print("\t{}, shape: {}".format(k, v.shape))
        print("\nMetadata:")
        for k in self.metadata.keys():
            print(f"\t{k}")
        print("\n----------\n")
        print("\nDataset Attributes:")
        for k in self.dset_attrs.keys():
            print(f"\t{k}")
        print("\n----------\n")


    def load_data_from_h5(self,
                          filename: str,
                          augment: bool = False,
                          cube_shape: tuple = (128, 128, 128),
                          start: int = 0,
                          end: Optional[int]= None) -> List[np.ndarray]: 
        """
        Loads data from h5 file. Assumes that each dataset in the h5 file has
        the shape (2, num_samples, *cube_shape). The first element on the 0
        axis are the training samples and the second are their true values.
        -----
        Params:
        :filename: name of h5 data file. Assumes that each dataset in the h5 file
        stores its data as [X, Y], with shape (2, num_samples, *cube_shape).
        :augment: augments the dataset if True
        :cube_shape: shape of a single training sample
        :start: used if you only want a subset of the data
        :end: used if you only want a subset of the data
        -----
        Loads lightcones from h5py file. Assumes h5py file has the datasets:
            'lightcones' --> original lightcones (mean of each slice removed) 
            'wedge_filtered_lightcones' --> lighcones minus wedge
            'redshifts' --> redshifts values for los axis.

        Returns: X and Y data, redshift values for each pixel along the los.
        """

        with h5py.File(filename, "r") as hf:

            # Check we have the required datasets
            datasets = list(hf.keys())
            logger.info(f"Datasets: {datasets}")
            assert "wedge_filtered_lightcones" in datasets and \
                   "lightcones" in datasets and \
                   "redshifts" in datasets and \
                   "random_seeds" in datasets, \
                   "Failed to extract datasets from h5py file."

            _X = np.array(hf["wedge_filtered_lightcones"][:], dtype=np.float32)
            _Y = np.array(hf["lightcones"][:], dtype=np.float32)
            _Z = np.array(hf["redshifts"][:], dtype=np.float32)
            _rs = np.array(hf["random_seeds"][:], dtype=np.float32)

            for k,v in hf.attrs.items():
                self.dset_attrs[k] = v

        # Assert no nan values
        assert np.isnan(np.sum(_X)) == False
        assert np.isnan(np.sum(_Y)) == False
        assert np.isnan(np.sum(_Z)) == False

        assert _X.shape[-3:] == cube_shape, f"expected {cube_shape}, got {_X.shape[-3:]}"
        assert _Y.shape == _X.shape
        assert _X.shape[1] == _Z.shape[0], "Must be one redshift per los-pixel"
        assert _X.shape[0] == _rs.shape[0], "Must be one random seed per lightcone"
        
        X = np.reshape(_X, (-1, *cube_shape))
        Y = np.reshape(_Y, (-1, *cube_shape))
        Z = np.reshape(_Z, (-1))
        self.random_seeds = np.reshape(_rs, (-1))

        return X[start:end], Y[start:end], Z[start:end]


    def load_metadata_from_h5(self, 
                              filename: str):
        """Loads metadata from h5 file. The keys in the hf file should either be
        images, masks, or metadata information."""

        hf = h5py.File(filename, "r")
        metakeys = [k for k in hf.keys() if ("masks" not in k) and "images" not in k]

        metadata = {}

        for k in hf.keys():
            # Only load the metadata
            if ("masks" in k) or ("images" in k):
                continue

            metadata[k] = np.copy(hf[k][:])
            print(f"{k} of shape ", metadata[k].shape)

        return metadata


    def save_lightcones(self, 
                        filename: str, 
                        results: dict) -> None:
        """
        Saves the model predictions (along with input and labels) to an h5 file
        -----
        Params:
        :filename: filename of output file
        :results: key is dataset name, value is an np.ndarray of shape
        (num_saved_samples, *lightcone_size)
        """
        with h5py.File(filename, "w") as hf:

            for dset_name, _data in results.items():
                logger.info(f"Creating {dset_name} dataset.")
                hf.create_dataset(dset_name, data=_data)


    def save_results(self, 
                     filename: str, 
                     results: dict,
                     start: int,
                     end: int) -> None:
        """
        Saves the model predictions (along original data) to an h5 file. By
        specifying a [start:end] range, we also save the original lightcones
        and wedge_filtered lightcones that were used to produce the results.
        -----
        Params:
        :filename: filename of output file
        :results: key is dataset name, value is an np.ndarray of shape
        (num_saved_samples, *lightcone_size)
        :start: starting index of original data to save.
        :end: ending index of original data to save
        """

        with h5py.File(filename, "w") as hf:

            # Save all results generated by model
            for dset_name, _data in results.items():
                assert end - start == _data.shape[0], \
                        f"end-start must match _data shape"

                logger.info(f"Saving {dset_name} dataset.")
                hf.create_dataset(dset_name, data=_data)

            # Save the corresponding data
            for dset_name, _data in self.data.items():
                assert 0<= start and end <= _data.shape[0], \
                        f"end-start must match _data shape"

                logger.info(f"Saving {dset_name} dataset.")
                hf.create_dataset(dset_name, data=_data[start:end])

            # Save the corresponding meta data
            for grp_name, grp_data in self.metadata.items():
                grp = hf.create_group(grp_name)
                logger.info(f"Saving {grp_name} group.")

                for dset_name, _data in grp_data.items():
                    assert 0<= start and end <= _data.shape[0], \
                            f"error: end-start must match _data shape"
                    logger.info(f"\tSaving {dset_name} values.")
                    hf.create_dataset(dset_name, data=_data[start:end])

            # Save the original dataset attributes
            for k, v in self.dset_attrs.items():
                logger.info(f"Saving {k} attribute.")
                hf.attrs[k] = str(v)


    def write_str(self, 
                  s: str, 
                  filename: str) -> None:
        """Writes string to file"""
        assert type(s) is str

        with open(filename,"w") as f:
            f.write(s)
            f.close()


    def normalize_by_frequency_slice(self, 
                  X: np.ndarray) -> np.ndarray: 
        """Performs l2 normalization along the frequency axis
        of each lightcone. Input is of shape (num_samples, *lightcone_shape)"""

        assert X.ndim == 4, \
            "Expected shape (num_samples, *lightcone_shape) but got {X.shape}"

        new_X = tf.math.l2_normalize(X, axis=(2,3)).numpy()

        # for i, lc in enumerate(X): 

        #     new_X[i] = np.copy(normalize(lc.reshape(256, -1)).reshape(256, 128, 128))
        assert (new_X.shape == X.shape)

        return new_X


    def remove_mean(self, 
                    X: np.ndarray) -> np.ndarray:
        """Removes the mean for each array in X"""
        mu = np.array([X[i].flatten().mean() for i in range(X.shape[0])])

        return np.copy(X - mu.reshape(-1, 1, 1, 1))


    def binarize_ground_truth(self, 
                              Y: np.ndarray) -> np.ndarray:
        """Returns binarized lightcone"""
        mins = Y.min(axis=(2,3), keepdims=True)
        Y -= mins

        return (Y > 0).astype(np.float32)


    def split_train_and_val_metadata(self, t: int):
        """
        Split metadata into two dictionaries. Used to separate the metadata
        for training and validation datasets.
        ----------
        :t: index to split at
        """
        self.train_metadata = {}
        self.valid_metadata = {}

        # values in metadata are either np arrays or dictionaries containing np
        # arrays
        for k, v in self.metadata.items():

            if isinstance(v, dict):
                self.train_metadata[k] = {}
                self.valid_metadata[k] = {}

                for k2, v2 in v.items():
                    assert 0 < t < v2.shape[0]
                    self.train_metadata[k][k2] = v2[:t]
                    self.valid_metadata[k][k2] = v2[t:]

            else:
                if isinstance(v, str):
                    self.train_metadata[k] = v
                    self.valid_metadata[k] = v
                    continue

                assert 0 < t < v.shape[0]
                self.train_metadata[k] = v[:t]
                self.valid_metadata[k] = v[t:]

