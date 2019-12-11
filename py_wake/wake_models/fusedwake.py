from py_wake.wake_model import WakeModel
from fusedwake.WindFarm import WindFarm, WindTurbineList
from fusedwake.WindTurbine import WindTurbineDICT

class PyWakeWindFarm(WindFarm):
    def __init__(self, site, windTurbines):
        self.wt = windTurbines
        self.wt._names# = names
        self.wt._diameters# = np.array(diameters)
        self.wt._hub_heights# = np.array(hub_heights)
        self.wt.ct_funcs# = ct_funcs
        self.wt.power_scale# = {'w': 1, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}[power_unit.lower()]
        self.wt.power_funcs# = list([lambda ws, f=f: f(ws) * self.power_scale for f in power_funcs])
        self.site = site

        # Creating the WindIO dicts
        [WindTurbineDICT({
            'name': self.wt._names[i],
            'turbine_type': 
            'position': self.pos[:,i],
            'power_curve': 
        }) for i in range(len(self.wt._names))]

        ## Setting up the WindFarm class variables
        self.pos = self.site.initial_position.T
        self.nWT = self.pos.shape[0]
        self.WT = WindTurbineList([WindTurbineDICT(wt, wf[wt['turbine_type']]) for wt in wf.wt_list])
        self.name = wf.name


# self.name = wt['name']
# self.turbine_type = wt['turbine_type']
# self.position = wt['position']
# self.type = wt_type['name']
# self.H = wt_type['hub_height']
# self.R = wt_type['rotor_diameter'] / 2.0

# if 'c_t_idle' in wt:
#     self.CT_idle = wt_type['c_t_idle']
# else:
#     self.CT_idle = 0.056

# self.power_factor = 1000.0 # <- Juan Pablo is using W as a basis to define power

# if 'power_curve' in wt_type:
#     self.pc = np.array(wt_type['power_curve'])
#     self.ctc = np.array(wt_type['c_t_curve'])
# elif 'power_curves' in wt_type: #TODO fix this??
#     wt_type['power_curve'] = wt_type['power_curves']
#     wt_type['c_t_curve'] = wt_type['c_t_curves']            
#     self.pc = np.array(wt_type['power_curves'])
#     self.ctc = np.array(wt_type['c_t_curves'])
# else:
#     raise Exception('No power curve found')

# self.u_cutin = wt_type['cut_in_wind_speed']
# self.u_cutout = wt_type['cut_out_wind_speed']
# self.P_rated = wt_type['rated_power'] * self.power_factor

# self.PCI = interpolator(self.pc[:,0], self.pc[:,1]*self.power_factor)
# self.CTCI = interpolator(self.ctc[:,0], self.ctc[:,1])

# index = np.nonzero(self.pc[:,1]*self.power_factor==self.P_rated)[0][0]
# self.PCI_u = interpolator(self.pc[:index+1,1] * self.power_factor, self.pc[:index+1,0])
# self.u_rated = wt_type['rated_wind_speed']
# self.refCurvesArray = np.vstack([self.pc[:,0].T,
#                                     self.pc[:,1].T*self.power_factor,
#                                     self.CTCI(self.pc[:,0].T)]).T

class FusedWakeModel(WakeModel):
    """

    Suffixes:

    - d: turbines down wind order
    - i: turbines ordered by id
    - j: downstream points/turbines
    - k: wind speeds
    - l: wind directions
    - m: turbines and wind directions (il.flatten())
    - n: from_turbines, to_turbines and wind directions (iil.flatten())

    Arguments available for calc_deficit (specifiy in args4deficit):

    - WS_lk: Local wind speed without wake effects
    - TI_lk: local turbulence intensity without wake effects
    - WS_eff_lk: Local wind speed with wake effects
    - TI_eff_lk: local turbulence intensity with wake effects
    - D_src_l: Diameter of source turbine
    - D_dst_jl: Diameter of destination turbine
    - dw_jl: Downwind distance from turbine i to point/turbine j
    - hcw_jl: Horizontal cross wind distance from turbine i to point/turbine j
    - cw_jl: Cross wind(horizontal and vertical) distance from turbine i to point/turbine j
    - ct_lk: Thrust coefficient

    """

    def __init__(self, site, windTurbines, wec=1):
        WakeModel.__init__(self, site, windTurbines, wec)

        self.set_WF(site, windTurbines)
        self.set_WT(windTurbines)

    def set_WF(self, site, windTurbines):
        self.WF = WindFarm()


        

    def set_WT(self, windTurbines):
        self.WT = WindTurbineDICT(wt_dict)

    def calc_wake(self, x_i, y_i, h_i=None, type_i=None, wd=None, ws=None):
        """Calculate wake effects

        Calculate effective wind speed, turbulence intensity (not
        implemented yet), power and thrust coefficient, and local
        site parameters

        Parameters
        ----------
        x_i : array_like
            X position of wind turbines
        y_i : array_like
            Y position of wind turbines
        h_i : array_like or None, optional
            Hub height of wind turbines\n
            If None, default, the standard hub height is used
        type_i : array_like or None, optional
            Wind turbine types\n
            If None, default, the first type is used (type=0)
        wd : int, float, array_like or None
            Wind directions(s)\n
            If None, default, the wake is calculated for site.default_wd
        ws : int, float, array_like or None
            Wind speed(s)\n
            If None, default, the wake is calculated for site.default_ws


        Returns
        -------
        WS_eff_ilk : array_like
            Effective wind speeds [m/s]
        TI_eff_ilk : array_like
            Turbulence intensities. Should be effective, but not implemented yet
        power_ilk : array_like
            Power productions [w]
        ct_ilk : array_like
            Thrust coefficients
        WD_ilk : array_like
            Wind direction(s)
        WS_ilk : array_like
            Wind speed(s)
        TI_ilk : array_like
            Ambient turbulence intensitie(s)
        P_ilk : array_like
            Probability
        """


        return WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, WD_ilk, WS_ilk, TI_ilk, P_ilk
