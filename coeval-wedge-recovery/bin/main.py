"""
@author: j-c-carr

Main script for recovering wedge modes using U-Net on coeval boxes.
"""

import os
import yaml
import logging
import argparse
import numpy as np
import tensorflow as tf

from util_manager import UtilManager
from model_manager import ModelManager
from stats_manager import StatsManager 
from coeval_plot_manager import CoevalPlotManager

tf.debugging.set_log_device_placement(False)
# Prints all logging info to std.err
logging.getLogger().addHandler(logging.StreamHandler())


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


def make_parser() -> argparse.ArgumentParser:
    """Makes command line argument parser. Returns ArgumentParser"""

    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="name of the model")
    parser.add_argument("datetime", help="datetime of exectution")
    parser.add_argument("fig_dir", help="name project fig directory")
    parser.add_argument("config_file", help="name of the model config (yaml format) file")
    parser.add_argument("data_file", help="name of data file (hdf5 format)")
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--predict", action="store_true", help="make predictions")
    parser.add_argument("--save_results", action="store_true", 
                        help="store results in h5 file")
    parser.add_argument("--old_model_loc", help="filename of pretrained weights")
    parser.add_argument("--sample_data_only", action="store_true", help="plot data sample and then exit")
    parser.add_argument("--results_dir", help="name of folder to store validation results")
    return parser.parse_args()


def read_params_from_yml_file(filename: str) -> dict:
    """ 
    Reads variables from yml filename. 
    """
    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        return params


def make_out_dir():
    OUT_DIR = args.fig_dir + "/out/" + args.datetime + args.title + "/"
    FIG_DIR = OUT_DIR + "figures/"
    try:
        os.mkdir(OUT_DIR)
        os.mkdir(FIG_DIR)
    except OSError as error:
        print(error)

    return OUT_DIR, FIG_DIR


if __name__ == "__main__":

    # Define constants and hyper-parameters
    args = make_parser()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    model_params = all_params["model_params"]
    CUBE_SHAPE = all_params["cube_shape"]
    CUBE_DIMENSIONS = all_params["cube_dimensions"]

    OUT_DIR, FIG_DIR = make_out_dir()

    np.random.seed(0)
    tf.random.set_seed(0)

    UM = UtilManager()
    CPM = CoevalPlotManager(CUBE_SHAPE, CUBE_DIMENSIONS)

    # Load data
    logger.info(f"Loading data from {args.data_file}...")
    UM.load_data_from_h5(args.data_file)
    Y = UM.data["brightness_temp_boxes"]
    X = UM.data["wedge_filtered_brightness_temp_boxes"]
    redshifts = UM.data["redshifts"]

    # Binarize ground truth, passed as labels to the model.
    B = (Y > 0).astype(np.float32)

    logger.info("Done.")

    MM = ModelManager(X, B, f"{args.datetime}_{args.title}",  model_params, 
                      CUBE_SHAPE)

    if args.sample_data_only:

        CPM.compare_coeval_boxes(f"{FIG_DIR}/data", 
                               {"Original Box": Y, 
                                "Binarized Box": B, 
                                "Wedge-removed Box": X}, 
                               num_samples=10)
        exit()

    # Loads multiple GPUs if available
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    logger.debug("Number of devices: {}".format(
                 strategy.num_replicas_in_sync))

    with strategy.scope():

        # Must compile model inside the scope block
        if args.old_model_loc is not None:
            MM.initialize_model(load_saved_weights=True, old_model=args.old_model_loc)
        else:
            MM.initialize_model()

        if args.train:
            MM.keras_train(strategy.num_replicas_in_sync)
 
        if args.predict:
            logging.debug(f"Making predictions...")
                
            # If the model was trained, save validation results only
            if args.train is True:
                MM.predict_on(MM.X_valid, MM.Y_valid)
                start = MM.X_train.shape[0]
            else:
                MM.predict_on(MM.X, MM.Y)
                start = 0

            CPM.compare_coeval_boxes(f"{FIG_DIR}/predictions", 
                                     {"Original Box": Y[start:],
                                      "Wedge-removed Box": X[start:], 
                                      "Binarized Box": B[start:],
                                      "Predicted Box": MM.preds}, 
                                     num_samples=10)
 
            SM = StatsManager(MM.binary_true, MM.binary_preds)
            SM.analyze_predictions()
            UM.write_str(str(SM.results), f"{OUT_DIR}/stats.txt")
            logging.debug("Done.")

        # Assumes that data has NOT been shuffled during training
        if args.save_results:
            assert args.predict is True, \
                    "Predictions not generated."

            assert args.results_dir is not None, \
                   "Must supply a --results_dir to store the dataset to."

            filename = f"{args.results_dir}/{args.title}_results.h5"
            UM.save_results(filename,
                            {"predicted_brightness_temp_boxes": MM.preds},
                            start=start, end=X.shape[0])

            logger.info("Done")
