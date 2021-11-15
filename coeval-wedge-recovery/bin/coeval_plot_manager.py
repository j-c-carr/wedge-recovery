import typing
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

eor_colour = colors.LinearSegmentedColormap.from_list(
    "EoR",
    [
        (0, "white"),
        (0.21, "yellow"),
        (0.42, "orange"),
        (0.63, "red"),
        (0.86, "black"),
        (0.9, "blue"),
        (1, "cyan"),
    ],
)
plt.register_cmap(cmap=eor_colour)
plt.style.use("lightcones/plot_styles.mplstyle")

class CoevalPlotManager():

    """Manager class for lightcone plotting functions."""

    def __init__(self, 
                 coeval_shape: tuple,
                 coeval_dimensions: tuple,
                 los_axis:int = 0, 
                 mpc_axis: int = 1) -> None:
        """
        Params:
        :coeval_shape: shape of the lightcone in pixels. LoS axis
        MUST be the first dimension.
        :coeval_dimensions: physical dimensions of the lightcone in
        Mpc. LoS axis MUST be the first dimension.
        :los_axis: line of sight axis of the lightcone. This is constant
        for each dataset and so each instance of the LightconePlotManager
        should have the same los_axis.
        :mpc_axis: any axis other than the los_axis
        """

        assert los_axis != mpc_axis
        assert max(los_axis, mpc_axis) < len(coeval_shape)

        self.coeval_shape = coeval_shape
        self.coeval_dimensions = coeval_dimensions
        self.los_axis = los_axis
        self.mpc_axis = mpc_axis

        self.mpc_labels = np.arange(0, coeval_dimensions[mpc_axis],
                step=coeval_dimensions[mpc_axis]//6)

        self.mpc_ticks = np.arange(0, coeval_shape[mpc_axis], 
                coeval_shape[mpc_axis]/len(self.mpc_labels))

        # Default matplotlib args
        self.kwargs = {
                "origin": "lower",
                "aspect": "auto", 
                "cmap": "coolwarm"
                }

        self.width_ratios = [1,
                coeval_shape[los_axis]//coeval_shape[mpc_axis]]


    def set_ticks_and_labels(self,
                             ax: List[plt.Axes],
                             redshift: float) -> None:

        l = r"$z=$" + str(redshift)

        # Put labels on last plots only
        ax[-1][0].set_xticklabels(self.mpc_labels)
        ax[-1][0].set_xlabel("Mpc")
        ax[-1][1].set_xlabel(l)


    def compare_coeval_boxes(self, 
                           prefix: str,
                           C: dict,
                           T: Optional[List[str]] = None,
                           num_samples: int = 1) -> None:
        """
        Wrapper function for plot_coeval_boxes to plot multiple coeval boxes and
        save them. Used for comparing model predictions and ground truth.
        -----
        Params:
        :prefix: prefix of image filename
        :C: dict of n sets of coeval boxes of shape (batch_size, *coeval_shape)
        :T: table data 
        :num_samples: number of plots to make.
        """

        # Plot random samples
        I = np.random.randint(list(C.values())[0].shape[0], size=num_samples)
        for i in I:
            self.plot_coeval_boxes(np.array([l[i, ...] for l in C.values()]),
                                list(C.keys()))
            plt.tight_layout()
            plt.savefig(f"{prefix}_{i}.png", dpi=400)
            plt.close()


    def plot_coeval_boxes(self, 
                        C: np.ndarray,
                        labels: List[str],
                        mpc_slice_index: Optional[int] = None,
                        los_slice_index: Optional[int] = None,
                        kwargs: Optional[dict] = None) -> None:
        """
        Plots a transverse and LoS slice of each lightcone in C. 
        -----
        Params:
        :C: list of of n coeval boxes of shape (n, *coeval_shape)
        :labels: list of of n labels
        :slice_index: (int) slice index for plotting
        :kwargs: (dict) matplotlib parameters
        """
        
        if kwargs is None:
            kwargs = self.kwargs

        if mpc_slice_index is None:
            mpc_slice_index = np.random.randint(0, self.coeval_shape[self.los_axis])
            
        if los_slice_index is None:
            los_slice_index = np.random.randint(0, self.coeval_shape[self.mpc_axis])

        assert mpc_slice_index < self.coeval_shape[self.los_axis] 
        assert los_slice_index < self.coeval_shape[self.mpc_axis] 

        fig, ax = plt.subplots(nrows=C.shape[0], ncols=2, 
                figsize=((self.width_ratios[1]+1)*2, C.shape[0]*2),
                gridspec_kw={'width_ratios': self.width_ratios})
        

        # Plot tranverse and los slice for each lightcone
        for i in range(C.shape[0]):

            # Transverse slice plot
            trans_slice = np.take(C[i], mpc_slice_index, axis=self.los_axis)
            ax[i][0].imshow(trans_slice, **kwargs)
            ax[i][0].set_ylabel(labels[i])

            
            # LoS slice plot + colorbar
            los_slice = np.take(C[i], los_slice_index, axis=self.mpc_axis).T
            ax[i][1].imshow(los_slice, **kwargs)

        #self.set_ticks_and_labels(ax, labels)

        ax[0][0].set_title("$\Delta T$ (Trans)")
        ax[0][1].set_title(r"$\Delta T$ (LoS)")


