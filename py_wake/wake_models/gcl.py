import jax.numpy as np
from py_wake.wake_model import WakeModel, LinearSum
from jax import grad, jit, vmap
import dask as da

na = np.newaxis

PARS = [0.435449861,0.797853685,-0.124807893,0.136821858,15.6298,1.0]
# Improved quadrature rule for wake deficit rotor averaging
node_R, node_th, weight = np.array([[ 0.26349922998554242692 ,  4.79436403870179805864 ,  0.00579798753740115753 ],
    [ 0.26349922998554242692 ,  5.13630491629471475079 ,  0.01299684397858970851 ],
    [ 0.26349922998554242692 ,  5.71955352542765460555 ,  0.01905256317618122044 ],
    [ 0.26349922998554242692 ,  0.20924454049880022999 ,  0.02341643323656225281 ],
    [ 0.26349922998554242692 ,  1.10309379714216659885 ,  0.02569988335562909190 ],
    [ 0.26349922998554242692 ,  2.03849885644762496284 ,  0.02569988335562912660 ],
    [ 0.26349922998554242692 ,  2.93234811309099407950 ,  0.02341643323656214179 ],
    [ 0.26349922998554242692 ,  3.70522443534172518653 ,  0.01905256317618119616 ],
    [ 0.26349922998554242692 ,  4.28847304447466459720 ,  0.01299684397858971198 ],
    [ 0.26349922998554242692 ,  4.63041392206758217753 ,  0.00579798753740114539 ],
    [ 0.57446451431535072718 ,  4.79436403870179805864 ,  0.01086984853977092380 ],
    [ 0.57446451431535072718 ,  5.13630491629471475079 ,  0.02436599330905551281 ],
    [ 0.57446451431535072718 ,  5.71955352542765460555 ,  0.03571902745281423097 ],
    [ 0.57446451431535072718 ,  0.20924454049880022999 ,  0.04390024659093685194 ],
    [ 0.57446451431535072718 ,  1.10309379714216659885 ,  0.04818117282305908744 ],
    [ 0.57446451431535072718 ,  2.03849885644762496284 ,  0.04818117282305915683 ],
    [ 0.57446451431535072718 ,  2.93234811309099407950 ,  0.04390024659093664378 ],
    [ 0.57446451431535072718 ,  3.70522443534172518653 ,  0.03571902745281418240 ],
    [ 0.57446451431535072718 ,  4.28847304447466459720 ,  0.02436599330905552321 ],
    [ 0.57446451431535072718 ,  4.63041392206758217753 ,  0.01086984853977089951 ],
    [ 0.81852948743000586429 ,  4.79436403870179805864 ,  0.01086984853977090992 ],
    [ 0.81852948743000586429 ,  5.13630491629471475079 ,  0.02436599330905548505 ],
    [ 0.81852948743000586429 ,  5.71955352542765460555 ,  0.03571902745281418934 ],
    [ 0.81852948743000586429 ,  0.20924454049880022999 ,  0.04390024659093679643 ],
    [ 0.81852948743000586429 ,  1.10309379714216659885 ,  0.04818117282305903193 ],
    [ 0.81852948743000586429 ,  2.03849885644762496284 ,  0.04818117282305909438 ],
    [ 0.81852948743000586429 ,  2.93234811309099407950 ,  0.04390024659093658826 ],
    [ 0.81852948743000586429 ,  3.70522443534172518653 ,  0.03571902745281413383 ],
    [ 0.81852948743000586429 ,  4.28847304447466459720 ,  0.02436599330905549199 ],
    [ 0.81852948743000586429 ,  4.63041392206758217753 ,  0.01086984853977088737 ],
    [ 0.96465960618086743494 ,  4.79436403870179805864 ,  0.00579798753740116100 ],
    [ 0.96465960618086743494 ,  5.13630491629471475079 ,  0.01299684397858971545 ],
    [ 0.96465960618086743494 ,  5.71955352542765460555 ,  0.01905256317618123432 ],
    [ 0.96465960618086743494 ,  0.20924454049880022999 ,  0.02341643323656226669 ],
    [ 0.96465960618086743494 ,  1.10309379714216659885 ,  0.02569988335562910925 ],
    [ 0.96465960618086743494 ,  2.03849885644762496284 ,  0.02569988335562914394 ],
    [ 0.96465960618086743494 ,  2.93234811309099407950 ,  0.02341643323656215567 ],
    [ 0.96465960618086743494 ,  3.70522443534172518653 ,  0.01905256317618120657 ],
    [ 0.96465960618086743494 ,  4.28847304447466459720 ,  0.01299684397858972065 ],
    [ 0.96465960618086743494 ,  4.63041392206758217753 ,  0.00579798753740114886 ]]).T

@jit#(nopython=True)
def my_power(term, factor):
    return np.exp(factor * np.log(term))

@jit#(nopython=True)
def get_r96(D, CT, TI, pars=PARS):
    """Computes the wake radius at 9.6D downstream location of a turbine

    .. math::
        R_{9.6D} = a_1 \\exp (a_2 C_T^2 + a_3 C_T + a_4)  (b_1  TI + b_2)  D

    Inputs
    ----------
    D: float
        Wind turbine diameter
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity
    pars: list
        GCL Model parameters [a1, a2, a3, a4, b1, b2]

    Returns
    -------
    R96: float
        Wake radius at 9.6D downstream location
    """
    a1, a2, a3, a4, b1, b2 = pars
    R96 = a1 * (np.exp(a2 * CT * CT + a3 * CT + a4)) * (b1 * TI + b2) * D

    return R96

@jit #(nopython=True)
def get_Rw(x, R, TI, CT, pars=PARS):
    """Computes the wake radius at a location.
    [1]-eq.3

    .. math::
        R_w = \\left(\\frac{105  c_1^2 }{2 \\pi}\\right)^{0.2} (C_T A (x + x_0))^{1/3}

    with A, the area, and x_0 and c_1 defined as

    .. math::
        x_0 = \\frac{9.6 D}{\\left(\\frac{2 R_96}{k D} \\right)^3 - 1}

        c_1 = \\left(\\frac{k D}{2}\\right)^{5/2}
              \\left(\\frac{105}{2 \\pi} \\right)^{-1/2}
              (C_T A x_0)^{-5/6}

    with k and m defined as

    .. math::
        k = \\sqrt{\\frac{m + 1}{2}}

        m = \\frac{1}{\\sqrt{1 - C_T}}

    Inputs
    ----------
    x: float or ndarray
        Distance between turbines and wake location in the wind direction
    R: float
        Wind turbine radius
    TI: float
        Ambient turbulence intensity
    CT: float
        Outputs WindTurbine object's thrust coefficient

    Returns
    -------
    Rw: float or ndarray
        Wake radius at a location
    """
    D = 2.0 * R
    Area = np.pi * D * D / 4.0

    m = 1.0 / (np.sqrt(1.0 - CT))
    k = np.sqrt((m + 1.0) / 2.0)

    R96 = get_r96(D, CT, TI, pars)

    x0 = (9.6 * D) / (my_power(2.0 * R96 / (k * D), 3.0) - 1.0)
    term1 = my_power(k * D / 2.0, 2.5)
    term2 = my_power(105.0/(2.0*np.pi), -0.5)
    term3 = my_power(CT * Area * x0, -5.0 / 6.0)
    c1 = term1 * term2 * term3

    Rw = my_power(105.0 * c1 * c1 / (2.0 * np.pi), 0.2) * my_power(CT * Area * (x + x0), 1.0 / 3.0)

    #if type(x) == float and x+x0 <= 0.: 
    #    Rw = 0
    #elif type(x) == np.ndarray: 
    Rw = np.where(x + x0 <= 0., 0., Rw)
    return Rw, x0, c1

@jit #(nopython=True)
def get_dU(x,r,R,CT,TI, order=1, pars=PARS):
    """Computes the wake velocity deficit at a location

    Inputs
    ----------
    x: float
        Distance between turbines and wake location in the wind direction
    r: float
        Radial distance between the turbine and the location
    R: float
        Wake producing turbine's radius [m]
    CT: float
        Outputs WindTurbine object's thrust coefficient
    TI: float
        Ambient turbulence intensity [-]
    order: int, optional

    Returns
    -------
    dU: float
        Wake velocity deficit at a location
    """
    #_ones = np.ones(np.shape(x))

    D = 2. * R
    Area = np.pi * D * D / 4.
    Rw, x0, c1 = get_Rw(x, R, TI, CT, pars)
    c1s = c1 * c1

    term10 = (1./9.) #*_ones
    xx0 = x+x0
    term20 = my_power(CT * Area / (xx0 * xx0), 1./3.)
    term310 = my_power(r, 1.5)
    term320 = 1.0 / np.sqrt(3. * c1s * CT * Area * (x+x0))
    term30 = term310 * term320
    term41 = my_power(35./(2. * np.pi), 3./10.)
    term42 = my_power(3. * c1s, -0.2)
    term40 = term41 * term42
    t4 = term30 - term40
    dU1 = -term10 * term20 * t4 * t4

    dU = dU1

    # if order == 2:

    #     z_term1 = r**1.5
    #     z_term2 = (CT*Area*(x+x0))**(-0.5)
    #     z_term3 = ((35./(2.*np.pi))**(-3./10.))*((3.*c1*c1)**(-3./10.))
    #     z = z_term1*z_term2*z_term3

    #     d_term = (4./81.)*(((35./(2.*np.pi))**(6./5.))*((3.*c1*c1)**(-12./15.)))

    #     d_4_const = (1./40.)
    #     d_3_const = (-4.+48./40.)*1./19.
    #     d_2_const = (6.+27.*d_3_const)*1./4.
    #     d_1_const = (4.-12.*d_2_const)*1./5.
    #     d_0_const = (-1.-3.*d_2_const)*1./8.

    #     d_0 = d_term*d_0_const
    #     d_1 = d_term*d_1_const
    #     d_2 = d_term*d_2_const
    #     d_3 = d_term*d_3_const
    #     d_4 = d_term*d_4_const

    #     dU2_const = ((CT*Area*((x+x0)**(-2.)))**(2./3.))
    #     dU2_term0 = d_0*(z**0.)
    #     dU2_term1 = d_1*(z**1.)
    #     dU2_term2 = d_2*(z**2.)
    #     dU2_term3 = d_3*(z**3.)
    #     dU2_term4 = d_4*(z**4.)

    #     dU2 = dU2_const*(dU2_term0+dU2_term1+dU2_term2+dU2_term3+dU2_term4)

    #     dU=dU1 + dU2

    #if type(r)==np.ndarray: 
    dU = np.where(np.logical_or(Rw<r, x<=0.), 0., dU)
    #dU[Rw<r]=0. # Outside the wake
    #dU[x<=0.]=0. # upstream the wake gen. WT
    # elif type(r)==float:
    #     if x<=0.: 
    #         dU = 0.
    #     if Rw<r: 
    #         dU = 0.
    #     if CT==0: 
    #         dU = 0.0*dU

    return dU

@jit #(nopython=True)
def get_dUeq(x,y,z,RT,R,CT,TI, pars=PARS):
    """Computes the wake velocity deficit at a location

    Inputs
    ----------
    x: float or array
        Distance between wake generating turbines and wake operating
        turbines in the streamwise direction
    y: float or array
        Distance between wake generating turbines and wake operating
        turbine in the crossflow horizontal direction
    z: float or array
        Distance between wake generating turbines and wake operating
        turbine in the crossflow vertical direction
    RT: float
        Wake operating turbine's radius [m]
    R: float or array
        Wake generating turbine's radius [m]
    TI: float
        Ambient turbulence intensity for the wake generating turbine [-]
    CT: float
        Thrust coefficient for the wake generating turbine [-]
    order: int, optional

    Returns
    -------
    dUeq: float
        Rotor averaged wake velocity deficit for each wake operating WT
    """
    x_msh, node_R_msh = np.meshgrid(x,node_R)
    y_msh, node_th_msh = np.meshgrid(y,node_th)
    z_msh, weight_msh = np.meshgrid(z,weight)

    xe = x_msh
    ye = y_msh + RT*node_R_msh*np.cos(node_th_msh)
    ze = z_msh + RT*node_R_msh*np.sin(node_th_msh)
    re = np.sqrt( ye**2. + ze**2. )

    dU_msh = get_dU(xe,re,R,CT,TI,order=1,pars=pars)
    dUeq = np.sum(weight_msh*dU_msh,axis=0)

    return dUeq

@jit #(nopython=True)
def gcl_calc_deficit_jax(WS_lk, D_src_l, D_dst_jl, dw_jl, hcw_jl, dh_jl, ct_lk, TI_lk, pars):
    R_dst_jl = D_dst_jl / 2.
    R_src_l =  D_src_l / 2.

    # Define the dimensions of the arrays
    J, L = dw_jl.shape
    R = node_R.shape[0]
    K = ct_lk.shape[1]

    # Broadcasting the arrays
    WS_jlk = np.broadcast_to(WS_lk[na,:,:], (J,L,K))
    dw_jlkr = np.broadcast_to(dw_jl[:,:,na,na], (J,L,K,R))
    hcw_jlkr = np.broadcast_to(hcw_jl[:,:,na,na], (J,L,K,R))
    dh_jlkr = np.broadcast_to(dh_jl[:,:,na,na], (J,L,K,R))
    node_R_jlkr = np.broadcast_to(node_R[na,na,na,:], (J,L,K,R))
    node_th_jlkr = np.broadcast_to(node_th[na,na,na,:], (J,L,K,R))
    weight_jlkr = np.broadcast_to(weight[na,na,na,:], (J,L,K,R))
    R_dst_jlkr = np.broadcast_to(R_dst_jl[:,:,na,na], (J,L,K,R))
    R_src_jlkr = np.broadcast_to(R_src_l[na,:,na,na], (J,L,K,R))
    ct_jlkr = np.broadcast_to(ct_lk[na,:,:,na], (J,L,K,R))
    TI_jlkr = np.broadcast_to(TI_lk[na,:,:,na], (J,L,K,R))

    # Creating the polar parametrization
    ye_jlkr = hcw_jlkr + R_dst_jlkr * node_R_jlkr * np.cos(node_th_jlkr)
    ze_jlkr = dh_jlkr + R_dst_jlkr * node_R_jlkr * np.sin(node_th_jlkr)
    re_jlkr = np.sqrt( ye_jlkr**2. + ze_jlkr**2. )

    dU_jlkr = get_dU(dw_jlkr, re_jlkr, R_src_jlkr, ct_jlkr, TI_jlkr, order=1, pars=pars)
    dUeq_jlk = np.sum(weight_jlkr * dU_jlkr, axis=-1)

    deficit_jlk = WS_jlk * dUeq_jlk
    return deficit_jlk


def gcl_calc_deficit_dask(WS_lk, D_src_l, D_dst_jl, dw_jl, hcw_jl, dh_jl, ct_lk, TI_lk, pars, chunks=None):
    R_dst_jl = D_dst_jl / 2.
    R_src_l =  D_src_l / 2.

    # Define the dimensions of the arrays
    J, L = dw_jl.shape
    R = node_R.shape[0]
    K = ct_lk.shape[1]

    # Broadcasting the arrays
    WS_jlk = da.array.broadcast_to(WS_lk[na,:,:], (J,L,K), chunks)
    dw_jlkr = da.array.broadcast_to(dw_jl[:,:,na,na], (J,L,K,R), chunks)
    hcw_jlkr = da.array.broadcast_to(hcw_jl[:,:,na,na], (J,L,K,R), chunks)
    dh_jlkr = da.array.broadcast_to(dh_jl[:,:,na,na], (J,L,K,R), chunks)
    node_R_jlkr = da.array.broadcast_to(node_R[na,na,na,:], (J,L,K,R), chunks)
    node_th_jlkr = da.array.broadcast_to(node_th[na,na,na,:], (J,L,K,R), chunks)
    weight_jlkr = da.array.broadcast_to(weight[na,na,na,:], (J,L,K,R), chunks)
    R_dst_jlkr = da.array.broadcast_to(R_dst_jl[:,:,na,na], (J,L,K,R), chunks)
    R_src_jlkr = da.array.broadcast_to(R_src_l[na,:,na,na], (J,L,K,R), chunks)
    ct_jlkr = da.array.broadcast_to(ct_lk[na,:,:,na], (J,L,K,R), chunks)
    TI_jlkr = da.array.broadcast_to(TI_lk[na,:,:,na], (J,L,K,R), chunks)

    # Creating the polar parametrization
    ye_jlkr = hcw_jlkr + R_dst_jlkr * node_R_jlkr * np.cos(node_th_jlkr)
    ze_jlkr = dh_jlkr + R_dst_jlkr * node_R_jlkr * np.sin(node_th_jlkr)
    re_jlkr = np.sqrt( ye_jlkr**2. + ze_jlkr**2. )

    dU_jlkr = get_dU(dw_jlkr, re_jlkr, R_src_jlkr, ct_jlkr, TI_jlkr, order=1, pars=pars)
    dUeq_jlk = np.sum(weight_jlkr * dU_jlkr, axis=-1)

    deficit_jlk_dask = WS_jlk * dUeq_jlk
    deficit_jlk = deficit_jlk_dask.compute()
    return deficit_jlk


class GCL(LinearSum, WakeModel):
    args4deficit = ['WS_lk', 'D_src_l', 'D_dst_jl', 'dw_jl', 'hcw_jl', 'dh_jl', 'ct_lk', 'TI_lk']

    def __init__(self, site, windTurbines, pars=PARS, **kwargs):
        WakeModel.__init__(self, site, windTurbines, **kwargs)
        self.pars = pars

    def calc_deficit(self, WS_lk, D_src_l, D_dst_jl, dw_jl, hcw_jl, dh_jl, ct_lk, TI_lk):
        return gcl_calc_deficit_dask(WS_lk, D_src_l, D_dst_jl, dw_jl, hcw_jl, dh_jl, ct_lk, TI_lk, self.pars)


