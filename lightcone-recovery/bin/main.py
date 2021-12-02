"""
Author: Jonathan Colaco Carr  (jonathan.colacocarr@mail.mcgill.ca)
Main script for recovering wedge modes using U-Net.
"""
import sys
import os
import yaml
import typing
import logging
import argparse
import numpy as np
import tensorflow as tf

from util_manager import UtilManager
from model_manager import ModelManager
from stats_manager import StatsManager 
from lightcone_plot_manager import LightconePlotManager


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
    parser.add_argument("fig_dir", help="name of directory for saving figures")
    parser.add_argument("config_file", help="name of the model config (yaml format) file")
    parser.add_argument("data_file", help="name of data file (hdf5 format)")
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--predict", action="store_true", help="make predictions")
    parser.add_argument("--predict_coeval", action="store_true", 
                        help="make predictions on coeval boxes")
    parser.add_argument("--save_lightcones", action="store_true", help="store results in h5 file")
    parser.add_argument("--save_coeval", action="store_true", help="store results in h5 file")
    parser.add_argument("--old_model_loc", help="filename of pretrained weights")
    parser.add_argument("--sample_data_only", action="store_true", help="plot data sample and then exit")
    parser.add_argument("--results_dir", help="folder to store results")
    return parser.parse_args()


def read_params_from_yml_file(filename: str) -> dict:
    """ 
    Reads variables from yml filename. 
    """
    with open(filename) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        return params


def make_out_dir():
    OUT_DIR = args.fig_dir  + args.datetime + args.title + "/"
    FIG_DIR = OUT_DIR + "figures/"
    try:
        os.mkdir(OUT_DIR)
        os.mkdir(FIG_DIR)
    except OSError as error:
        print(error)

    return OUT_DIR, FIG_DIR


if __name__=="__main__":

    # Define constants and hyper-parameters
    args = make_parser()
    all_params = read_params_from_yml_file(args.config_file)
    logger = init_logger("test.log", __name__)

    model_params = all_params["model_params"]
    LIGHTCONE_SHAPE = all_params["lightcone_shape"]
    LIGHTCONE_DIMENSIONS = all_params["lightcone_dimensions"]

    OUT_DIR, FIG_DIR = make_out_dir()

    np.random.seed(0)
    tf.random.set_seed(0)

    UM = UtilManager()

    # Load data
    logger.info(f"Loading data from {args.data_file}...")
    UM.load_data_from_h5(args.data_file)

    X = UM.data["wedge_filtered_lightcones"]
    Y = UM.data["lightcones"]
    redshifts = UM.data["redshifts"]
    B = UM.binarize_ground_truth(Y)

    logger.info("Done.")

    LPM = LightconePlotManager(redshifts, LIGHTCONE_SHAPE,
                               LIGHTCONE_DIMENSIONS)

    if args.sample_data_only:
        LPM.compare_lightcones(f"{FIG_DIR}/data", 
                               {"Binarized LC": B, 
                                "Original LC": Y, 
                                "Wedge-removed LC": X}, 
                               num_samples=10)
        exit()


    MM = ModelManager(X, B, f"{args.datetime}_{args.title}", model_params, LIGHTCONE_SHAPE)

    # Loads multiple GPUs if available
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    logger.debug("Number of devices: {}".format(
                 strategy.num_replicas_in_sync))

    with strategy.scope():

        # Must compile model inside the scope block
        if args.old_model_loc is not None:
            MM.initialize_model(load_saved_weights=True, 
                                old_model=args.old_model_loc)
        else:
            MM.initialize_model()
 
        if args.train:
            MM.keras_train(strategy.num_replicas_in_sync)
 
        if args.predict:
            logging.debug(f"Making predictions...")

            # Don't save the boxes that the model was trained on
            if args.train is True:
                MM.predict_on(MM.X_valid, MM.Y_valid)
                start=MM.X_train.shape[0]
            else:
                MM.predict_on(MM.X, MM.Y)
                start=0

            LPM.compare_lightcones(f"{FIG_DIR}/predictions", 
                                   {"Original LC": MM.Y_valid,
                                    "Wedge-Removed LC": MM.X_valid, 
                                    "Predicted LC": MM.preds}, 
                                   num_samples=10)

            SM = StatsManager(MM.binary_true, MM.binary_preds)
            SM.analyze_predictions()
            UM.write_str(str(SM.results), f"{OUT_DIR}/stats.txt")
            logging.debug("Done.")
 
        if args.save_lightcones:

            assert args.predict is True, \
                    "No predictions to save. Please specify --predict."

            assert args.results_dir is not None, \
                    "Please specify the --results_dir."

            filename = f"{args.results_dir}/{args.title}_validation.h5"
            logger.info(f"Saving predictions to {filename}")

            UM.save_results(filename,
                            {"predicted_lightcones": MM.preds},
                            start=start,
                            end=MM.X.shape[0])
            logger.info("Done")
        
