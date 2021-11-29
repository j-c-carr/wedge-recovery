import re
import h5py
import logging
import pprint

import os
import typing
from typing import Optional, List
import numpy as np


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


    def load_data_from_h5(self, filepath):
        """Loads all data from h5 file, returns nothing. (Typically used just
        to observe the values in a dataset)"""
        self.filepath = filepath

        with h5py.File(filepath, "r") as hf:

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


    def shuffle_data(self, 
                     X: np.ndarray, 
                     Y: np.ndarray, 
                     Z: np.ndarray) -> np.ndarray:
        """
        Wrapper function to shuffle the data, keeping the indices of the samples
        the same (TODO: there is definitely a better implementation for this...)
        """
        assert X.shape == Y.shape
        assert X.shape[0] == Z.shape[0]

        shuffled_X = np.empty(X.shape)
        shuffled_Y = np.empty(Y.shape)
        shuffled_Z = np.empty(Z.shape)

        I = np.random.permutation(X.shape[0])
        for ix, i in enumerate(I):
            shuffled_X[ix] = X[i]
            shuffled_Y[ix] = Y[i]
            shuffled_Z[ix] = Z[i]

        return shuffled_X, shuffled_Y, shuffled_Z


    def save_data_to_h5(self, 
                         filename: str, 
                         results: dict,
                         stats: Optional[str] = None) -> None:
        """
        Saves the model predictions (along with input and labels) to an h5 file
        -----
        Params:
        :filename: filename of output file
        :results: key is dataset name, value is np.ndarray data
        """
        with h5py.File(filename, "w") as hf:

            for dset_name, _data in results.items():
                logger.info(f"Creating {dset_name} dataset.")
                hf.create_dataset(dset_name, data=_data)

            for k, v in self.dset_attrs.items():
                hf.attrs[k] = str(v)

            if stats is not None:
                hf.attrs["stats"] = str(stats)

            if hasattr(self, "xH_boxes"):
                hf.create_dataset("ionized_boxes", data=self.xH_boxes)

        # Print success message
        logger.info("\n----------\n")
        logger.info(f"Validation results saved to {filename}")
        logger.info("Contents:")
        for k in results.keys():
            logger.info("\t'{}', shape: {}".format(k, results[k].shape))
        logger.info("Attributes:")
        logger.info("dset attrs: {}".format(pprint.pformat(self.dset_attrs)))
        if stats is not None:
            logger.info("Validation metrics for ionized bubbles:")
            logger.info("{}".format(pprint.pformat(stats)))
        logger.info("\n----------\n")


    def write_str(self, 
                  s: str, 
                  filename: str) -> None:
        """Writes string to file"""
        assert type(s) is str

        with open(filename,"w") as f:
            f.write(s)
            f.close()
