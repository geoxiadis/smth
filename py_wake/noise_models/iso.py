import numpy as np
from py_wake.site.distance import StraightDistance
from numpy import newaxis as na
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.noise_models.utils import iso_transmission_loss
import warnings


class ISONoiseModel:

    def __init__(self, src_x, src_y, src_h, freqs, sound_power_level,
                 elevation_function=lambda x, y: np.zeros(np.shape(x)),
                 delta_swl_correction=None):
        """ISO 9613-2 noise model for outdoor sound propagation with atmospheric absorption (ISO 9613-1).

        Calculates sound pressure levels including spherical spreading, ground effects,
        and atmospheric absorption. Supports directional sound power level corrections.

        Parameters
        ----------
        src_x, src_y : array_like
            Source coordinates
        src_h : array_like or float
            Source heights (typically hub height)
        freqs : array_like
            Frequencies of sound_power_level
        sound_power_level : array_like
            Emitted sound power level, shape=(n_sources, n_freq)
        elevation_function : callable, optional
            Function returning ground elevation
        delta_swl_correction : callable, optional
            Function for directional sound power level corrections
        """

        self.elevation_function = elevation_function
        self.src_x, self.src_y = src_x, src_y
        self.src_h = np.zeros_like(src_x) + src_h
        self.src_z = elevation_function(src_x, src_y)
        self.n_src = len(src_x)
        self.distance = StraightDistance()
        
        self.sound_power_level = sound_power_level
        self.delta_swl_correction = delta_swl_correction

        # print("SPL in ISONoiseModel:")
        # print("sound_power_level.shape:", sound_power_level.shape)
        # print("delta_swl_correction.shape:", delta_swl_correction.shape)
        
        freqs_init = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 
                               4000.0, 8000.0])

        self.freqs = freqs
        self.freqs_init = freqs_init 

    def __call__(self, rec_x, rec_y, rec_h, patm, Temp, RHum, ground_type, **kwargs):
        """Calculate the sound pressure level at a list of reveicers

        Parameters
        ----------
        rec_x : array_like
            x coordinate, [m], of receivers
        rec_y : array_like
            y coordinate, [m], of receivers
        rec_h : array_like or float
            height, [m], of receivers (typically 2m)
        patm : float
            atmospheric pressure (Pa)
        Temp : float
            Temperature (deg celcius)
        RHum : float
            Relative humidity (%)
        ground_type : float
            Factor describing the amount of ground absorption, 0=hard reflective ground,
            e.g. paving, water, ice and concrete. 1= soft porous absorbing ground,
            e.g. grass, trees, other vegetation and farming land

        Returns
        -------
        total_spl : array_like
            Total sound pressure level at each receiver
        spl : array_like
            Sound pressure level of freqs, dim=(n_receiver, n_freq)
        """
        # frequencies from the supplied WT dataset
        freqs = self.freqs
        # octave band center frequencies for ground effect calculation
        freqs_init = self.freqs_init
        # source heights
        src_h = self.src_h
        # compute distances between sources and receivers
        rec_h = np.zeros_like(rec_x) + rec_h
        dx_ijlk, dy_ijlk, dh_ijlk = self.distance(self.src_x, self.src_y, self.src_h, wd_l=np.array([270]),
                                                  dst_xyh_jlk=[rec_x, rec_y, rec_h])
        # compute the loss in sound pressure level from source to receivers
        Delta_SPL_ijlkf = iso_transmission_loss(freqs_init, freqs, 
                                                dx_ijlk, dy_ijlk, dh_ijlk, 
                                                src_h, rec_h, ground_type, 
                                                patm, Temp, RHum)

        # print("IN ISONoiseModel __call__:")
        # print("Delta_SPL_ijlkf.shape:", Delta_SPL_ijlkf.shape)
        # exit()

        sound_power_ijlkf = np.expand_dims(self.sound_power_level,
                                           tuple(range(1, 6 - len(np.shape(self.sound_power_level)))))

        # print("sound_power_ijlkf.shape:", sound_power_ijlkf.shape)
        # exit()
        
        # apply directional correction if available
        if self.delta_swl_correction is not None:
            correction = self.delta_swl_correction(rec_x, rec_y, rec_h, self.freqs, **kwargs)
            sound_power_ijlkf = sound_power_ijlkf + correction

        # Do not integrate over different frequencies but make a map for diff. freq. 
        spl_jlkf = 10.0 * np.log10(np.sum(10.0**((sound_power_ijlkf + Delta_SPL_ijlkf) / 10.0), axis=0))
        total_spl_jlk = 10.0 * np.log10(np.sum(10.0**(spl_jlkf / 10.0), axis=-1))
        
        # print(spl_jlkf.shape)
        # exit()
        
        return total_spl_jlk, spl_jlkf


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.site._site import UniformSite
        from py_wake.examples.data.noise_example.turbine_noise import TurbineNoise
        from py_wake.utils.layouts import rectangle
        from py_wake.turbulence_models.crespo import CrespoHernandez
        from py_wake.deficit_models.gaussian import ZongGaussian
        from py_wake.flow_map import XYGrid
        from py_wake.utils.plotting import setup_plot

        wt = TurbineNoise()
        site = UniformSite()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The .* model is not representative of the setup used in the literature')
        wfm = ZongGaussian(site, wt, turbulenceModel=CrespoHernandez())

        # coordiantes with a row with 4 turbines
        x, y = rectangle(4, 4, 5 * wt.diameter())

        # sim_res = wfm(x, y, wd=[270, 271], ws=[9, 12], mode=[1], yaw=[0, 10])
        # different wind situations for the turbines
        sim_res = wfm(x, y, wd=[270], ws=[10], mode=[0], yaw=[-20])
        # print("sim_res ds:", wfm.windTurbines.ds)
        # exit()
        rec_x = np.array([x[0], x[-1]])
        rec_y = np.array([1000, 1000])
        rec_h = 2
        
        # get the noise map for the wind speed and direction from the situations
        # specified above

        # 50 points in x, 50 points in y. And then 4 turbines so we have 50*50*4 = 10000 recievers
        noise_map = sim_res.sel(wd=270, ws=10).noise_map(
            noiseModel=ISONoiseModel,
            grid=XYGrid(x=np.linspace(-500, 3500, 50),
                        y=np.linspace(-500, 1500, 50),
                        h=2),
            ground_type=0.0,
            atmopheric_pressure=101325,
            temperature=20,
            relative_humidity=80
        )
        # nm = sim_res.sel(wd=270, ws=8).noise_model(noiseModel=ISONoiseModel)
        nm = sim_res.sel(wd=270, ws=10).noise_model(noiseModel=ISONoiseModel)
        total_sp_jlk, spl_jlkf = nm(rec_x=rec_x, rec_y=rec_y, rec_h=rec_h,
                                    patm=101325, Temp=20, RHum=80, ground_type=0.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

        # sim_res.flow_map(wd=270, ws=8).plot_wake_map(ax=ax1)
        sim_res.flow_map(wd=270, ws=10).plot_wake_map(ax=ax1)
        ax1.plot(rec_x, rec_y, 'ro', markersize=8, label='Receivers')
        ax1.legend()

        ax2.plot(nm.freqs, spl_jlkf[0, 0, 0], 'b-', label='Receiver 1')
        ax2.plot(nm.freqs, spl_jlkf[1, 0, 0], 'r-', label='Receiver 2')
        setup_plot(xlabel='Frequency [Hz]', ylabel='Sound pressure level [dB]', ax=ax2)
        ax2.legend()
        plt.tight_layout()
        plt.savefig("iso_noise_model_receivers.png", dpi=150)
        plt.close()

        fig, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        total_spl_map = noise_map['Total sound pressure level']
        if len(total_spl_map.shape) > 2:
            total_spl_map = total_spl_map.squeeze()

        im = ax3.contourf(noise_map.x, noise_map.y, total_spl_map, levels=20, cmap='viridis')
        ax3.plot(x, y, 'wo', markersize=8, label='Turbines')
        ax3.plot(rec_x, rec_y, 'ro', markersize=8, label='Receivers')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('y [m]')
        ax3.legend()
        ax3.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Total SPL [dB]')

        plt.tight_layout()
        # plt.show()
        fig.savefig("iso_noise_model_map.png", dpi=150)


main()
