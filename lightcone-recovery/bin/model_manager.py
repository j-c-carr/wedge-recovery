"""
Author: Samuel Gagnon-Hartman, modified by Jonathan Colaco Carr (@j-c-carr)

This network will seek to replicate that in https://arxiv.org/pdf/1802.10508.pdf
"""
import os
import typing
from typing import Optional, List, Any
import logging
import numpy as np
from tqdm import tqdm

from isensee2017 import *
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam

from isensee2017 import isensee2017_model
from loss_functions import dice_coefficient_loss

def init_logger(f, name):
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

LOGGER = init_logger("test.log", __name__)

class ModelManager():
    """Manager class for creating, compiling, training, the model"""

    def __init__(self, 
                 X: np.ndarray,
                 Y: np.ndarray,
                 ID: str,
                 LOG_DIR: str,
                 params: dict,
                 CUBE_SHAPE: tuple) -> None:
        """
        Params:
        :X: training samples
        :Y: labels of training samples
        :ID: model label with format: <name>_datetime 
        :LOG_DIR: directory for Tensorboard logs
        :params: dictionary for model params, loaded from yml file
        :CUBE_SHAPE: shape of a single training sample
        """

        self.X = X
        self.Y = Y

        # Model params read from config file in main.py
        for k, v in params.items():
            setattr(self, k.upper(), v)

        # Truncate input data if too large to fit in model
        assert X.shape[1:] == CUBE_SHAPE and Y.shape[1:] == CUBE_SHAPE, \
                f"Model expects samples of shape {CUBE_SHAPE}"+\
                f" but got shape {X.shape[1:]} and {Y.shape}"

        if self.STANDARDIZE:
            Xmu = self.X.mean(axis=(2, 3), keepdims=True)
            Xstd = self.X.std(axis=(2, 3), keepdims=True)
            self.X = (self.X - Xmu) / Xstd

            #Ymu = self.Y.mean(axis=(2, 3), keepdims=True)
            #Ystd = self.Y.std(axis=(2, 3), keepdims=True)
            #self.Y = (self.Y - Ymu) / Ystd


        self.ID = ID
        self.LOG_DIR = LOG_DIR
        self.CUBE_SHAPE = CUBE_SHAPE

        self.MODEL_LOC = f"scratch/model-checkpoints/{self.ID}-checkpoint.h5"

        
        schedule = lambda epoch, lr: 5e-4 * (self.LRF ** epoch)

        self.callbacks = [EarlyStopping(patience=20, verbose=0),
                          LearningRateScheduler(schedule, verbose=0),
                          ModelCheckpoint(self.MODEL_LOC, 
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True)]
                          # TensorBoard(log_dir=LOG_DIR, 
                          #             histogram_freq=1,
                          #             update_freq='epoch',
                          #             write_graph=True,
                          #             profile_batch="2,6")]

        self.X_train, self.X_valid, self.Y_train, self.Y_valid = \
        train_test_split(self.X, self.Y, test_size=params["test_size"],
                         shuffle=params["shuffle"])


    def initialize_model(self, 
                         load_saved_weights: bool = False,
                         old_model: Optional[str] = None) -> None:
        """
        Initializes the isensee2017 U-Net model.
        Params:
        :load_saved_weights: (bool) if True, loads the saved weights in
        <old_model>
        :old_model: location of old model weights
        """
        input_img = Input((*self.CUBE_SHAPE, 1), name='img')
        self.model = isensee2017_model(input_img, 
                                       depth=self.LEVELS,
                                       n_segmentation_levels=self.SEG_LEVELS)
        if load_saved_weights == True:
            self.model.load_weights(old_model)


    def create_tensorflow_datasets(self, 
                                   num_devices: int) -> List[tf.Tensor]:
        # Wrap data in a Dataset object to avoid auto-shard warnings
        train_data = tf.data.Dataset.from_tensor_slices((self.X_train,
            self.Y_train))
        val_data = tf.data.Dataset.from_tensor_slices((self.X_valid,
            self.Y_valid))
        # Batch size must be set on Dataset object
        train_data = train_data.shuffle(buffer_size=1024).batch(self.BATCH_SIZE*num_devices)
        val_data = val_data.shuffle(buffer_size=1024).batch(self.BATCH_SIZE*num_devices)

        # Disable Auto-shard
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
                tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        return train_data, val_data


    def keras_train(self,
                    num_devices: int) -> None:
        """Trains the model using model.fit"""
        LOGGER.info(f"Training on {self.X_train.shape} samples...")
        # Format datasets
        train_dataset, val_dataset = self.create_tensorflow_datasets(num_devices)

        self.results = self.model.fit(train_dataset, epochs=self.EPOCHS,
                callbacks=self.callbacks, validation_data=val_dataset)
        LOGGER.info(f"Done.")
        

    def predict_on(self, 
                   X: tf.Tensor,
                   Y_true: tf.Tensor,
                   threshold: float = 0.9) -> None:
        """
        Load the best model weights and compute the predictions for the
        training and validation sets. Also computes the binary predictions
        Params:
        :threshold: (float) cutoff value for creating binary predictions
        :X: (tf.tensor) lightcones to predict on
        """

        preds = np.empty(X.shape)

        # Single GPU can only handle one prediction at a time
        for i in range(X.shape[0]):
            _preds = self.model.predict(tf.expand_dims(X[i], axis=0), verbose=self.VERBOSE)
            preds[i] = np.squeeze(_preds)

        self.preds = preds
        self.true = Y_true

        # Threshold predictions
        binary_preds = (self.preds > threshold).astype(np.uint8)
        binary_true = (self.true > 0.).astype(np.uint8)

        self.binary_preds = np.copy(binary_preds)
        self.binary_true = np.copy(binary_true)
    
