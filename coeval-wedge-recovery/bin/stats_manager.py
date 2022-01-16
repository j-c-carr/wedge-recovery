"""
@author: j-c-carr
"""

import numpy as np
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, MeanIoU


class StatsManager:

    """Computes statistics for binary coeval box predictions. """

    def __init__(self,
                 Y_true: np.ndarray,
                 Y_pred: np.ndarray,
                 threshold=0.9):
        """
        Params:
        :Y_true:    List of expected binarized lightcones
        :Y_pred:    List of predicted binarized lightcones
        :threshold: Threshold value for tensorflow metrics
        """

        assert Y_true.shape == Y_pred.shape
        # Compute accuracy based on ionized bubbles
        self.Y_true = 1 - Y_true
        self.Y_pred = 1 - Y_pred 

        self.results = {}
        self.metrics = {'accuracy': BinaryAccuracy(threshold=threshold),
                        'precision': Precision(thresholds=threshold),
                        'recall': Recall(thresholds=threshold),
                        'meanIoU': MeanIoU(num_classes=2)}

    def analyze_predictions(self):
        """Compute the metrics for all of the predictions"""

        for metric, m in self.metrics.items():
            print(f"Calculating {metric}...")
            for i in range(self.Y_pred.shape[0]):
                m.update_state(self.Y_true[i], self.Y_pred[i])
            self.results[f"{metric}"] = m.result().numpy()
            print(f"Done.")

        self.results["nfrac"] = 1 - (self.Y_true.sum() / self.Y_true.size)
        print(self.results)



