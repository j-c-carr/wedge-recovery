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


class LightconePlotManager():

    """Manager class for lightcone plotting functions."""

    def __init__(self, 
                 lightcone_redshifts: np.ndarray,
                 lightcone_shape: tuple,
                 lightcone_dimensions: tuple,
                 los_axis:int = 0, 
                 mpc_axis: int = 1) -> None:
        """
        Params:
        :lightcone_redshifts: list of redshifts along the LoS
        axis.
        :lightcone_shape: shape of the lightcone in pixels. LoS axis
        MUST be the first dimension.
        :lightcone_dimensions: physical dimensions of the lightcone in
        Mpc. LoS axis MUST be the first dimension.
        :los_axis: line of sight axis of the lightcone. This is constant
        for each dataset and so each instance of the LightconePlotManager
        should have the same los_axis.
        :mpc_axis: any axis other than the los_axis
        """

        assert los_axis != mpc_axis
        assert max(los_axis, mpc_axis) < len(lightcone_shape)
        assert lightcone_redshifts.shape[0] == lightcone_shape[los_axis]

        self.lightcone_redshifts = lightcone_redshifts
        self.lightcone_shape = lightcone_shape
        self.lightcone_dimensions = lightcone_dimensions
        self.los_axis = los_axis
        self.mpc_axis = mpc_axis

        self.redshift_labels = [
                round(self.lightcone_redshifts[i], 2) for i in
                    range(0, self.lightcone_redshifts.shape[0],
                        self.lightcone_redshifts.shape[0]//(3*lightcone_shape[0]//128))]

        self.redshift_ticks = np.arange(0, lightcone_shape[los_axis],
                step=lightcone_shape[los_axis]/len(self.redshift_labels))

        self.mpc_labels = np.arange(0, lightcone_dimensions[mpc_axis],
                step=lightcone_dimensions[mpc_axis]//6)

        self.mpc_ticks = np.arange(0, lightcone_shape[mpc_axis], 
                lightcone_shape[mpc_axis]/len(self.mpc_labels))

        # Default matplotlib args
        self.kwargs = {
                "origin": "lower",
                "aspect": "auto", 
                "cmap": "coolwarm"
                }

        self.width_ratios = [1,
                lightcone_shape[los_axis]//lightcone_shape[mpc_axis]]


    def set_ticks_and_labels(self,
                             ax: List[plt.Axes],
                             labels: List[str],
                             n: int) -> None:
        """
        Set the ticks and labels for the axes in the plot_lightcones
        function.
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
        :prefix: prefix of image filename
        :L: dict of n sets of lightcones of shape (batch_size, *lightcone_shape)
        :astro_params: {p21c AstroParam name: (*batch_size)}
        :num_samples: number of plots to make.
        """

        # Plot random samples
        I = np.random.randint(list(L.values())[0].shape[0], size=num_samples)
        for i in I:
            self.plot_lightcones(np.array([l[i, ...] for l in L.values()]),
                                list(L.keys()))

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
        -----
        Params:
        :L: list of of n lightcones of shape (n, *lightcone_shape)
        :labels: list of of n labels
        :slice_index: (int) slice index for plotting
        :kwargs: (dict) matplotlib parameters
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

            
            # LoS slice plot + colorbar
            los_slice = np.take(L[i], los_slice_index, axis=self.mpc_axis).T
            ax[i][1].imshow(los_slice, **kwargs)
            # cax = fig.add_axes([0.9, 0.9, , height])
            # cbar = fig.colorbar(pos, ax=ax[i][1], pad=0.01)
            # cbar.ax.set_ylabel(labels[i], rotation=270)



        self.set_ticks_and_labels(ax, labels, L.shape[0])

        ax[0][0].set_title("$\Delta T$ (Trans), $z=$ {:.2f}".format(
                                        self.lightcone_redshifts[mpc_slice_index]))
        ax[0][1].set_title(r"$\Delta T$ (LoS)")



    def histogram_slice(self, x1, x2, f, ix=0):

        assert x.ndim == 3

        # the histogram of the data
        n, bins, patches = plt.hist(x1[ix], 50, alpha=0.75, label=f"wedge-removed {ix}")
        n, bins, patches = plt.hist(x2[ix], 50, alpha=0.75, label=f"ground-truth {-1}")

        plt.title("Histogram of frequency slices for h5py data")
        plt.xlabel("Pixel value")
        plt.ylabel("Number of occurences")
        plt.legend()
        plt.savefig(f, dpi=400)


