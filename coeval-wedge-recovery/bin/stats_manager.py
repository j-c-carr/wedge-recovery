import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, MeanIoU 

class StatsManager():

    """Computes statistics for binary lightcone predictions. """

    def __init__(self, Y_true, Y_pred, _t=0.9):
        """
        Params:
        :Y_true: (np.ndarray) list of expected binarized lightcones
        :Y_pred: (np.ndarray) list of predicted binarized lightcones
        """

        assert Y_true.shape == Y_pred.shape
        # Compute accuracy based on ionized bubbles
        self.Y_true = 1 - Y_true
        self.Y_pred = 1 - Y_pred 

        self.metrics = {
                'accuracy': BinaryAccuracy(threshold=_t),
                'precision': Precision(thresholds=_t),
                'recall': Recall(thresholds=_t),
                'meanIoU': MeanIoU(num_classes=2)
                }


    def analyze_predictions(self, idx=None):
        """Compute the metrics for all of the predictions"""
        self.results = {}
        for metric, m in self.metrics.items():
            print(f"Calculating {metric}...")
            for i in range(self.Y_pred.shape[0]):
                m.update_state(self.Y_true[i], self.Y_pred[i])
            self.results[f"{metric}"] = m.result().numpy()
            print(f"Done.")

        self.results["nfrac"] = 1 - (self.Y_true.sum() / self.Y_true.size)
        print(self.results)



