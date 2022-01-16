"""
@author: j-c-carr

Manager class for plotting lightcones
"""
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


class LightconePlotManager:
    """Manager class for creating lightcone plots"""

    def __init__(self, 
                 lightcone_redshifts: np.ndarray,
                 lightcone_shape: tuple,
                 lightcone_dimensions: tuple,
                 los_axis: int = 0,
                 mpc_axis: int = 1) -> None:
        """
        ----------
        Params:
        :lightcone_redshifts:  List of redshifts along the LoS axis
        :lightcone_shape:      Shape of the lightcone in pixels,
                               LoS axis MUST be the first dimension
        :lightcone_dimensions: Physical dimensions of the lightcone in Mpc,
                               LoS axis MUST be the first dimension
        :los_axis:             Line of sight axis of the lightcone
        :mpc_axis:             Any axis other than the los_axis
        """

        assert los_axis != mpc_axis
        assert max(los_axis, mpc_axis) < len(lightcone_shape)
        assert lightcone_redshifts.shape[0] == lightcone_shape[los_axis]

        self.lightcone_redshifts = lightcone_redshifts
        self.lightcone_shape = lightcone_shape
        self.lightcone_dimensions = lightcone_dimensions
        self.los_axis = los_axis
        self.mpc_axis = mpc_axis

        self.redshift_labels = \
            [round(self.lightcone_redshifts[i], 2) for i in
             range(0, self.lightcone_redshifts.shape[0],
                   self.lightcone_redshifts.shape[0]//(3*lightcone_shape[0]//128))]

        self.redshift_ticks = \
            np.arange(0, lightcone_shape[los_axis],
                      step=lightcone_shape[los_axis]/len(self.redshift_labels))

        self.mpc_labels = \
            np.arange(0, lightcone_dimensions[mpc_axis],
                      step=lightcone_dimensions[mpc_axis]//6)

        self.mpc_ticks = \
            np.arange(0, lightcone_shape[mpc_axis],
                      lightcone_shape[mpc_axis]/len(self.mpc_labels))

        self.width_ratios = \
            [1, lightcone_shape[los_axis]//lightcone_shape[mpc_axis]]

        # Default matplotlib args
        self.kwargs = {"origin": "lower",
                       "aspect": "auto",
                       "cmap": "coolwarm"}

    def set_ticks_and_labels(self,
                             ax: List[plt.Axes],
                             labels: List[str],
                             n: int) -> None:
        """
        Set the ticks and labels for the axes in the plot_lightcones function.
        """
        for i in range(n):
            ax[i][0].set_xticks(self.mpc_ticks)
            ax[i][0].set_xticklabels([])
            ax[i][0].set_yticks(self.mpc_ticks)
            ax[i][0].set_yticklabels([])
            ax[i][0].set_ylabel(labels[i])

            ax[i][1].set_xticks(self.redshift_ticks)
            ax[i][1].set_xticklabels([])
            ax[i][1].set_yticks(self.mpc_ticks)
            ax[i][1].set_yticklabels([])
            # ax[i][1].set_ylabel(labels[i])

        # Put labels on last plots only
        ax[-1][0].set_xticklabels(self.mpc_labels)
        ax[-1][0].set_xlabel("Mpc")
        ax[-1][1].set_xticklabels(self.redshift_labels)
        ax[-1][1].set_xlabel(r"$z$")

    def compare_lightcones(self,
                           prefix: str,
                           L: dict,
                           astro_params: Optional[dict] = None,
                           num_samples: int = 1) -> None:
        """
        Wrapper function for plot_lightcones to plot multiple lightcones and
        save them. Used for comparing model predictions and ground truth.
        -----
        Params:
        :prefix:       Prefix for image filenames
        :L:            Dict of n sets of lightcones of shape (batch_size, *lightcone_shape)
        :astro_params: {p21c AstroParam name: (*batch_size)}
        :num_samples:  Number of figures to make
        """

        # Plot random samples
        for i in np.random.randint(list(L.values())[0].shape[0], size=num_samples):
            self.plot_lightcones(np.array([lc[i, ...] for lc in L.values()]),
                                 list(L.keys()))

            # Add AstroParam values to the plot, if specified
            if astro_params is not None:
                astro_param_str = ""
                for k, v in astro_params.items():
                    astro_param_str += r"{}: {:.2e}, ".format(k, v[i])

                plt.suptitle(astro_param_str[:-2].replace("_", "\_"), fontsize="xx-small")

            plt.tight_layout()
            plt.savefig(f"{prefix}_{i}.png", dpi=400)
            plt.close()

    def plot_lightcones(self,
                        L: np.ndarray,
                        labels: List[str],
                        mpc_slice_index: Optional[int] = None,
                        los_slice_index: Optional[int] = None,
                        kwargs: Optional[dict] = None) -> None:
        """
        Plots a transverse and LoS slice of each lightcone in L. 
        -----------
        Params:
        :L:           List of of n lightcones of shape (n, *lightcone_shape)
        :labels:      List of of n labels
        :slice_index: slice index for plotting
        :kwargs:      Extra matplotlib parameters
        """
        
        if kwargs is None:
            kwargs = self.kwargs

        if mpc_slice_index is None:
            mpc_slice_index = np.random.randint(0, self.lightcone_shape[self.los_axis])
            
        if los_slice_index is None:
            los_slice_index = np.random.randint(0, self.lightcone_shape[self.mpc_axis])

        assert mpc_slice_index < self.lightcone_shape[self.los_axis] 
        assert los_slice_index < self.lightcone_shape[self.mpc_axis] 

        fig, ax = plt.subplots(nrows=L.shape[0], ncols=2, 
                               figsize=((self.width_ratios[1]+1)*2, L.shape[0]*2),
                               gridspec_kw={'width_ratios': self.width_ratios})
        
        # Plot tranverse and los slice for each lightcone
        for i in range(L.shape[0]):

            # Transverse slice plot
            trans_slice = np.take(L[i], mpc_slice_index, axis=self.los_axis)
            ax[i][0].imshow(trans_slice, **kwargs)

            # LoS slice plot
            los_slice = np.take(L[i], los_slice_index, axis=self.mpc_axis).T
            ax[i][1].imshow(los_slice, **kwargs)

        self.set_ticks_and_labels(ax, labels, L.shape[0])

        ax[0][0].set_title("$\Delta T$ (Trans), $z=$ {:.2f}".format(
            self.lightcone_redshifts[mpc_slice_index]))
        ax[0][1].set_title(r"$\Delta T$ (LoS)")
