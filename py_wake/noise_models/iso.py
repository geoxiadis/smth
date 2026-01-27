import numpy as np
from py_wake.site.distance import StraightDistance
from numpy import newaxis as na
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.noise_models.utils import atmab_alpha
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
        
        freqs_init = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 
                               4000.0, 8000.0])

        self.freqs = freqs
        self.freqs_init = freqs_init

    def iso_ground_eff(self, freqs, freqs_init, ground_distance_ijlk, src_h, rec_h, ground_type):
        """_summary_
        Parameters
        ----------
        ground_distance_ijlk : _type_
            _description_
        distance_ijlk : _type_
            _description_
        ground_type : _type_
            _description_
        Returns
        -------
        _type_
            _description_
        """
        # freqs_init = self.freqs_init
        # Ground effects ISO
        src_h_ilk = np.expand_dims(src_h, tuple(range(len(np.shape(src_h)), 3))) + .0
        src_h_ijlk = src_h_ilk[:, na]
        rec_h_ijlk = rec_h[na, :, na, na]
        hei_check_ijlk = 30.0 * (src_h_ijlk + rec_h_ijlk)
        q_ijlk = np.where(ground_distance_ijlk <= hei_check_ijlk, 0, 1 - hei_check_ijlk / ground_distance_ijlk)
        Gs = Gr = Gm = ground_type
        # for f=63Hz Am =-3*q, for other frequencies Am = -3 * q * (1-Gm)
        fak = np.where(freqs_init == 63.0, 1, (1 - Gm))
        Am_fijlk = -3.0 * q_ijlk[na] * fak[:, na, na, na, na]
        # 125 Hz
        a_source_ijlk = 1.5 + 3.0 * np.exp(-0.12 * (src_h_ijlk - 5.0)**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0)) + \
            5.7 * np.exp(-0.09 * src_h_ijlk**2) * \
            (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ijlk**2)))
        a_rec_ijlk = 1.5 + 3.0 * np.exp(-0.12 * (rec_h_ijlk - 5.0)**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0)) + \
            5.7 * np.exp(-0.09 * rec_h_ijlk**2) * (1.0 - np.exp(-2.8 * (10.0**(-6.0)) * (ground_distance_ijlk**2)))
        # 250 Hz
        b_source_ijlk = 1.5 + 8.6 * np.exp(-0.09 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        b_rec_ijlk = 1.5 + 8.6 * np.exp(-0.09 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        # 500 Hz
        c_source_ijlk = 1.5 + 14.0 * np.exp(-0.46 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        c_rec_ijlk = 1.5 + 14.0 * np.exp(-0.46 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        # 1000 Hz
        d_source_ijlk = 1.5 + 5.0 * np.exp(-0.9 * src_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        d_rec_ijlk = 1.5 + 5.0 * np.exp(-0.9 * rec_h_ijlk**2) * (1.0 - np.exp(-ground_distance_ijlk / 50.0))
        zeros = np.zeros_like(ground_distance_ijlk)
        As_fijlk = np.array([zeros - 1.5,  # 63Hz
                             -1.5 + Gs * a_source_ijlk,  # 125Hz
                             -1.5 + Gs * b_source_ijlk,  # 250Hz
                             -1.5 + Gs * c_source_ijlk,  # 500Hz
                             -1.5 + Gs * d_source_ijlk,  # 1000Hz
                             zeros - 1.5 * (1.0 - Gs),  # 2000Hz
                             zeros - 1.5 * (1.0 - Gs),  # 4000Hz
                             zeros - 1.5 * (1.0 - Gs)  # 8000Hz
                             ])
        Ar_fijlk = np.array([zeros - 1.5,  # 63 Hz
                             -1.5 + Gr * a_rec_ijlk,  # 125 Hz
                             -1.5 + Gr * b_rec_ijlk,  # 250 Hz
                             -1.5 + Gr * c_rec_ijlk,  # 500 Hz
                             -1.5 + Gr * d_rec_ijlk,  # 1000 Hz
                             zeros - 1.5 * (1.0 - Gr),  # 2000 Hz
                             zeros - 1.5 * (1.0 - Gr),  # 4000 Hz
                             zeros - 1.5 * (1.0 - Gr)  # 8000 Hz
                             ])
        # sum up As+Ar+Am
        Agr_fijlk = np.sum([As_fijlk, Ar_fijlk, Am_fijlk], 0)
        # Interpolation to other frequencies are actually not following the standard completely.
        ip = GridInterpolator([freqs_init], Agr_fijlk)
        Agr_ijlkf = np.moveaxis(ip(np.array([freqs]).T), 0, -1)
        # # TODO: move geometric spreading out in to transmission loss! Should be -ve
        # # Area of sphere: 4pi R^2
        # # 10*log_10(4pi) ~ 11
        # # 10 * log_10(R^2) = 20 * log_10(R) =
        # # reference area = 1.0 m^2
        # Adiv_ijlk = 20.0 * np.log10(distance_ijlk / 1.0) + 11.0  # The geometrical spreading (divergence)
        # ISO_ground_ijlkf = - Adiv_ijlk[:, :, :, :, na] - Agr_ijlkf
        # return ISO_ground_ijlkf
        return Agr_ijlkf

    def transmission_loss(self, freqs_init, freqs, dx_ijlk, dy_ijlk, dh_ijlk,
                          src_h, rec_h, ground_type, patm, Temp, RHum):
        """
        Docstring for transmission_loss
        
        :param self: Description
        :param freqs_init: Description
        :param freqs: Description
        :param dx_ijlk: Description
        :param dy_ijlk: Description
        :param dh_ijlk: Description
        :param src_h: Description
        :param rec_h: Description
        :param ground_type: Description
        :param patm: Description
        :param Temp: Description
        :param RHum: Description
        """
        # convert atmospheric pressure from Pa to atm
        patm /= 101325.0 
        # transmission loss = ground effects + atmospheric absorption
        # rec_h = np.zeros_like(rec_x) + rec_h
        # dx_ijlk, dy_ijlk, dh_ijlk = self.distance(self.src_x, self.src_y, self.src_h, wd_l=np.array([270]),
        #                                           dst_xyh_jlk=[rec_x, rec_y, rec_h])
        ground_distance_ijlk = np.sqrt(dx_ijlk**2 + dy_ijlk**2)
        distance_ijlk = np.sqrt(dx_ijlk**2 + dy_ijlk**2 + dh_ijlk**2)
        
        # TODO: multiply the distance here and exmpand the dimentsions
        # The atmospheric absorption term
        atm_abs_ijlkf = -1 * atmab_alpha(freqs, T0=Temp, RH0=RHum, p_s=patm) * distance_ijlk[:, :, na]  
        # ground effect term
        ground_eff_ijlkf = -1 * self.iso_ground_eff(freqs, freqs_init, ground_distance_ijlk, src_h, rec_h, ground_type)
        # # The geometrical spreading (divergence)
        # Area of sphere: 4pi R^2
        # 10*log_10(4pi) ~ 11
        # 10 * log_10(R^2) = 20 * log_10(R) =
        # reference area = 1.0 m^2
        geom_div_ijlk = -1 * (20.0 * np.log10(distance_ijlk / 1.0) + 11.0)  
        #  total transmission loss in sound pressure level
        delta_spl_ijlkf = ground_eff_ijlkf + atm_abs_ijlkf + geom_div_ijlk[:, :, :, :, na]
        
        return delta_spl_ijlkf  # Delta_SPL

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
        Delta_SPL_ijlkf = self.transmission_loss(freqs_init, freqs, 
                                                 dx_ijlk, dy_ijlk, dh_ijlk, 
                                                 src_h, rec_h, ground_type, 
                                                 patm, Temp, RHum)


        sound_power_ijlkf = np.expand_dims(self.sound_power_level,
                                           tuple(range(1, 6 - len(np.shape(self.sound_power_level)))))
        
        # apply directional correction if available
        if self.delta_swl_correction is not None:
            correction = self.delta_swl_correction(rec_x, rec_y, rec_h, self.freqs, **kwargs)
            sound_power_ijlkf = sound_power_ijlkf + correction

        # Do not integrate over different frequencies but make a map for diff. freq. 
        spl_jlkf = 10.0 * np.log10(np.sum(10.0**((sound_power_ijlkf + Delta_SPL_ijlkf) / 10.0), axis=0))
        total_spl_jlk = 10.0 * np.log10(np.sum(10.0**(spl_jlkf / 10.0), axis=-1))
        
        
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
        # plt.show()
        # plt.savefig("iso_noise_model_receivers.png", dpi=150)
        # plt.close()

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
        # fig.savefig("iso_noise_model_map.png", dpi=150)


main()
