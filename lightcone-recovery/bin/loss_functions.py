"""
@author: j-c-carr

Collection of loss functions used for training the U-Net in isensee.py
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


def dice_coefficient(y_true, y_pred):
    """
    Computes the dice coefficient between two tensors
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_crossentropy(y_true, y_pred, pos_weight=100):
    """
    pos_weight is a multiplicative coefficient for positive labels in the loss
    term. If pos_weight > 1, false negative count decreases. If pos_weight < 1,
    false positive count decreases. 
    """
    if type(pos_weight) != int:
        w = np.ones((1, 128, 1, 1), dtype=np.float32)
        w[:, 128:, :, :] += 15
        pos_weight = tf.convert_to_tensor(w, dtype=np.float32)

    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)


def tmp_weighted_crossentropy(y_true, y_pred):
    """
    pos_weight is a multiplicative coefficient for positive labels in the loss
    term. If pos_weight > 1, false negative count decreases. If pos_weight < 1,
    false positive count decreases. 
    """

    nfrac = tf.reduce_mean(y_true, axis=(2, 3), keepdims=True)
    pos_weight = nfrac

    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)


def balanced_crossentropy(y_true, y_pred):
    """
    Implements balanced crossentropy as in (Bianco et. al, 2021), eq. 3
    """

    # Calculate neutral fraction
    nfrac = tf.reduce_mean(y_true, axis=(2, 3), keepdims=True)
    pos_weight = (1-nfrac) / nfrac

    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight)


class TverskyLoss(Loss):
    """Tvserky loss generalizes the dice coefficient loss by adding weight to
    false positives and false negatives with the help of a coefficient :beta:"""

    def __init__(self, beta=0.5):
        super().__init__()
        assert 0 <= beta <= 1.
        self.beta = beta

    def call(self, y_true, y_pred):
        """
        Computes the Tversky Loss between two tensors as described in Eq. 10 of
        (Jadon, 2020)
        """
        # Flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)

        # True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))

        return 1 - ((0.5 + TP) / (0.5 + TP + self.beta*FP + (1-self.beta)*FN))


class FocalTverskyLoss(Loss):
    
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Computes the Focal Tversky Loss between two tensors
        """
        # Flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)

        # True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))

        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        focal_tversky = K.pow((1 - tversky), self.gamma)
        
        return focal_tversky
