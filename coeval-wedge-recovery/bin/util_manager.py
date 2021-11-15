import re
import h5py
import logging
import pprint

import os
import typing
from typing import Optional, List
import numpy as np
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

        
    def load_data_from_h5(self,
                          filename: str,
                          cube_shape: tuple,
                          start: int = 0,
                          end: Optional[int] = None) -> List[np.ndarray]:
        """
        Loads coeval boxes from h5py file. Assumes h5py file has 4 datasets,
            'brightness_boxes' --> ground truth brightness temp boxes 
            'wedge_filtered_brightness_temp_boxes' --> brightness temp boxes minus wedge
            'ionized_boxes' --> coeval_boxes minus wedge
            'redshifts' --> redshift of each coeval box
        """

        with h5py.File(filename, "r") as hf:

            # Check we have the required datasets
            datasets = list(hf.keys())
            logger.info(f"Datasets: {datasets}")
            assert "wedge_filtered_brightness_temp_boxes" in datasets and \
                   "brightness_temp_boxes" in datasets and \
                   "ionized_boxes" in datasets and \
                   "redshifts" in datasets, \
                   "Failed to extract datasets from h5py file."

            _X = np.array(hf["wedge_filtered_brightness_temp_boxes"][:], 
                          dtype=np.float32)
            _Y = np.array(hf["brightness_temp_boxes"][:], dtype=np.float32)
            redshifts = np.array(hf["redshifts"][:], dtype=np.float32)
            xh = np.array(hf["ionized_boxes"][:], dtype=np.float32)

            # Load metadata from h5 file
            for k, v in hf.attrs.items():
                if k != "redshifts":
                    self.dset_attrs[k] = v


        # Assert no nan values
        assert np.isnan(np.sum(_X)) == False
        assert np.isnan(np.sum(_Y)) == False
        assert np.isnan(np.sum(redshifts)) == False
        assert np.isnan(np.sum(xh)) == False

        assert _X.shape[-3:] == cube_shape, \
                f"expected {cube_shape}, got {_X.shape[-3:]}"
        assert _Y.shape[-3:] == cube_shape

        X = np.reshape(_X, (-1, *cube_shape))
        Y = np.reshape(_Y, (-1, *cube_shape))
        self.xH_boxes = xh

        assert redshifts.shape[0] == _X.shape[0], \
                f"expected {redshifts.shape[0]}, got {_X.shape[0]}"

        return X[start:end], Y[start:end], redshifts[start:end]


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
