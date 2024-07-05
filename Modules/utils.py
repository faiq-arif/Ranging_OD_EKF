import numpy as np
from astropy.constants import GM_earth
import astropy.units as u
from astropy.time import Time

mu = GM_earth.value

def rk4_step(func, y, dt):
    k1 = func(y)
    k2 = func(y + 0.5 * dt * k1)
    k3 = func(y + 0.5 * dt * k2)
    k4 = func(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_prop(X, dt, dynamics):
    return rk4_step(dynamics, X, dt)

def euler_prop(X, dt, dynamics):
    f = dynamics(X)
    Xp = np.zeros((6,))
    Xp[3:6] = X[3:6] + dt * f[3:6]  # Using the acceleration part of dynamics
    Xp[0:3] = X[0:3] + (1/2) * dt * (X[3:6] + Xp[3:6])
    return Xp

def linear_int(x, y, xi):
    x = x.astype(float)
    y = y.astype(float)
    xi = xi.astype(float)
    
    yi = np.zeros_like(xi)
    indices = np.searchsorted(x, xi) - 1  # Find the interval for each xi
    for i in range(len(xi)):
        if xi[i] <= x[0]:
            yi[i] = y[0]
        elif xi[i] >= x[-1]:
            yi[i] = y[-1]
        else:
            j = indices[i]
            if j >= 0 and j < len(x) - 1:
                yi[i] = y[j] + (xi[i] - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j])
    return yi

def interpolate_df(df, elapsed_time, df_name):
    """
    Perform linear interpolation for X_TOD, Y_TOD, Z_TOD, Vx_TOD, Vy_TOD, Vz_TOD
    for a given dataframe and return the arrays as pos_eci_'df_name' and vel_eci_'df_name'.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    elapsed_time (array-like): The time values for interpolation.
    df_name (str): The name of the dataframe for naming the output variables.
    
    Returns:
    dict: A dictionary with interpolated position and velocity arrays.
    """
    pos_columns = ['X_TOD', 'Y_TOD', 'Z_TOD']
    vel_columns = ['Vx_TOD', 'Vy_TOD', 'Vz_TOD']
    posgs_columns = ['X_TOD_Site', 'Y_TOD_Site', 'Z_TOD_Site']
    
    if 'TA_Range' in df.columns:
        rho_columns = ['TA_Range']
    else:
        rho_columns = ['Value']
    
    pos_eci = np.zeros((len(elapsed_time), len(pos_columns)))
    vel_eci = np.zeros((len(elapsed_time), len(vel_columns)))
    pos_gs = np.zeros((len(elapsed_time), len(posgs_columns)))
    rho = np.zeros((len(elapsed_time), len(rho_columns)))
    
    for i, col in enumerate(pos_columns):
        pos_eci[:, i] = linear_int(df['Time Elapsed'].values, df[col].values, elapsed_time)
    
    for i, col in enumerate(vel_columns):
        vel_eci[:, i] = linear_int(df['Time Elapsed'].values, df[col].values, elapsed_time)
        
    for i, col in enumerate(posgs_columns):
        pos_gs[:, i] = linear_int(df['Time Elapsed'].values, df[col].values, elapsed_time)
        
    for i, col in enumerate(rho_columns):
        rho[:, i] = linear_int(df['Time Elapsed'].values, df[col].values, elapsed_time)
    
    result = {
        f'pos_eci': pos_eci,
        f'vel_eci': vel_eci,
        f'pos_gs' : pos_gs,
        f'rho': rho
    }
    
    return result