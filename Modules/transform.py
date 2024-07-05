import pandas as pd
import numpy as np
from datetime import datetime
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, ICRS, TETE, get_sun, AltAz, ITRS
from astropy.coordinates import TEME, CIRS, CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.utils.iers import conf as iers_conf
from astropy.utils import iers
from astropy.coordinates import get_sun
from astropy.coordinates.erfa_astrom import get_jd12
from astropy.constants import G, M_earth, GM_earth, R_earth

# Ensure IERS data is loaded
iers_conf.auto_download = True

def julian_centuries(date):
    """
    Convert a given date to Julian centuries from J2000.0 using Astropy.
    
    Parameters:
    date (str or astropy.time.Time): The date for which to calculate Julian centuries.
    
    Returns:
    float: Julian centuries from J2000.0
    """
    # Convert the input date to an Astropy Time object if it isn't one already
    if not isinstance(date, Time):
        date = Time(date, format='iso', scale='utc')
    
    # Julian date of J2000.0
    JD_J2000 = 2451545.0
    
    # Calculate Julian Date (JD) for the given date using Astropy
    JD = date.jd
    
    # Calculate Julian centuries from J2000.0
    T = (JD - JD_J2000) / 36525.0
    
    return T    

def get_tod_values(time):
    """
    Obtain the true of date values of xp, yp, psi, delta_psi, epsilon, and delta_epsilon.
    
    Parameters:
    time (Time): Astropy Time object.
    
    Returns:
    dict: Dictionary containing true of date values.
    """
    # IERS data for polar motion
    iers_b = iers.IERS_B.open()
    xp, yp = iers_b.pm_xy(time)
    
    xp = xp.to(u.rad).value
    yp = yp.to(u.rad).value
    
    # Nutation components (delta_psi, delta_epsilon)
    delta_psi, delta_epsilon = iers_b.dcip_xy(time)
    delta_psi = delta_psi.to(u.rad).value
    delta_epsilon = delta_epsilon.to(u.rad).value

    # Greenwich Apparent Sidereal Time (GAST)
    gast = time.sidereal_time('apparent', 'greenwich').radian

    return {
        'xp': xp,
        'yp': yp,
        'psi': delta_psi,
        'delta_psi': delta_psi,
        'delta_epsilon': delta_epsilon,
        'gast': gast
    }

def eci2ecef(X_eci, time):
    """
    Transform coordinates from Earth-Centred Earth-Fixed (ECEF) to Earth-Centred Inertial (ECI).
    
    Parameters:
    X_ecef (np.array): 3D coordinates in ECEF frame [x, y, z]
    time (Time): Astropy Time object
    
    Returns:
    np.array: 3D coordinates in ECI frame
    """
    # Julian centuries from J2000.0
    T = (time.jd - 2451545.0) / 36525.0
    
    # Precession matrix components
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * u.arcsec.to(u.rad)
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * u.arcsec.to(u.rad)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * u.arcsec.to(u.rad)
    
    # Obtain true of date values
    values = get_tod_values(time)
    xp = values['xp']
    yp = values['yp']
    delta_psi = values['delta_psi']
    delta_epsilon = values['delta_epsilon']
    GAST = values['gast']
    
    RP = np.array([
        [np.cos(z)*np.cos(theta)*np.cos(zeta)-np.sin(z)*np.sin(zeta), -np.cos(z)*np.cos(theta)*np.sin(zeta)-np.sin(z)*np.cos(zeta), -np.cos(z)*np.sin(theta)],
        [np.sin(z)*np.cos(theta)*np.cos(zeta)+np.cos(z)*np.sin(zeta), -np.sin(z)*np.cos(theta)*np.sin(zeta)+np.cos(z)*np.cos(zeta), -np.sin(z)*np.sin(theta)],
        [np.sin(theta)*np.cos(zeta), -np.sin(theta)*np.sin(zeta), np.cos(theta)]
    ])
    
    #RP = np.matmul(R1,np.matmul(R2,R3))
    
    # Nutation matrix components
    epsilon = (84381.448 - 46.8150 * T - 0.00059 * T**2 + 0.001813 * T**3) * u.arcsec.to(u.rad)

    RN = np.array([
        [np.cos(delta_psi), -np.sin(delta_psi)*np.cos(epsilon), -np.sin(delta_psi)*np.sin(epsilon)],
        [np.sin(delta_psi)*np.cos(epsilon+delta_epsilon), np.cos(delta_psi)*np.cos(epsilon+delta_epsilon)*np.cos(epsilon)+np.sin(epsilon+delta_epsilon)*np.sin(epsilon), np.cos(delta_psi)*np.cos(epsilon+delta_epsilon)*np.sin(epsilon)-np.sin(epsilon+delta_epsilon)*np.cos(epsilon)],
        [np.sin(delta_psi)*np.sin(epsilon+delta_epsilon), np.cos(delta_psi)*np.sin(epsilon+delta_epsilon)*np.cos(epsilon)-np.cos(epsilon+delta_epsilon)*np.sin(epsilon), np.cos(delta_psi)*np.sin(epsilon+delta_epsilon)*np.sin(epsilon)+np.cos(epsilon+delta_epsilon)*np.cos(epsilon)]
    ])
    
    #RN = np.matmul((RN1,np.matmul(RN2,RN3)))
    
    # Earth rotation matrix
    RS = np.array([
        [np.cos(GAST), np.sin(GAST), 0],
        [-np.sin(GAST), np.cos(GAST), 0],
        [0, 0, 1]
    ])
    
    # Polar motion matrix
    RM = np.array([
        [np.cos(xp), np.sin(xp)*np.sin(yp), np.sin(xp)*np.cos(yp)],
        [0, np.cos(yp), -np.sin(yp)],
        [-np.sin(xp), np.cos(xp)*np.sin(yp), np.cos(xp)*np.cos(yp)]
    ])

    X_RP = np.matmul(RP,X_eci)
    X_RN = np.matmul(RN,X_RP)
    X_RS = np.matmul(RS,X_RN)
    X_RM = np.matmul(RM,X_RS)
    X_t2ecef = np.matmul(RS,np.matmul(RN,np.matmul(RP,X_eci)))
    X_ecef = np.matmul(RM,np.matmul(RS,np.matmul(RN,np.matmul(RP,X_eci))))
    
    return X_RP, X_RN, X_RS, X_RM, X_ecef, X_t2ecef

def ecef2eci(X_ecef, time):
    """
    Transform coordinates from Earth-Centred Earth-Fixed (ECEF) to Earth-Centred Inertial (ECI).
    
    Parameters:
    X_ecef (np.array): 3D coordinates in ECEF frame [x, y, z]
    time (Time): Astropy Time object
    
    Returns:
    np.array: 3D coordinates in ECI frame
    """
    # Julian centuries from J2000.0
    T = (time.jd - 2451545.0) / 36525.0
    
    # Precession matrix components
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * u.arcsec.to(u.rad)
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * u.arcsec.to(u.rad)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * u.arcsec.to(u.rad)
    
    # Obtain true of date values
    values = get_tod_values(time)
    xp = values['xp']
    yp = values['yp']
    delta_psi = values['delta_psi']
    delta_epsilon = values['delta_epsilon']
    GAST = values['gast']
    
    # Precession Matrix
    RP = np.array([
        [np.cos(z)*np.cos(theta)*np.cos(zeta)-np.sin(z)*np.sin(zeta), -np.cos(z)*np.cos(theta)*np.sin(zeta)-np.sin(z)*np.cos(zeta), -np.cos(z)*np.sin(theta)],
        [np.sin(z)*np.cos(theta)*np.cos(zeta)+np.cos(z)*np.sin(zeta), -np.sin(z)*np.cos(theta)*np.sin(zeta)+np.cos(z)*np.cos(zeta), -np.sin(z)*np.sin(theta)],
        [np.sin(theta)*np.cos(zeta), -np.sin(theta)*np.sin(zeta), np.cos(theta)]
    ])
    
    RP_inv = np.linalg.inv(RP)
    
    # Nutation matrix components
    epsilon = (84381.448 - 46.8150 * T - 0.00059 * T**2 + 0.001813 * T**3) * u.arcsec.to(u.rad)
    
    RN = np.array([
        [np.cos(delta_psi), -np.sin(delta_psi)*np.cos(epsilon), -np.sin(delta_psi)*np.sin(epsilon)],
        [np.sin(delta_psi)*np.cos(epsilon+delta_epsilon), np.cos(delta_psi)*np.cos(epsilon+delta_epsilon)*np.cos(epsilon)+np.sin(epsilon+delta_epsilon)*np.sin(epsilon), np.cos(delta_psi)*np.cos(epsilon+delta_epsilon)*np.sin(epsilon)-np.sin(epsilon+delta_epsilon)*np.cos(epsilon)],
        [np.sin(delta_psi)*np.sin(epsilon+delta_epsilon), np.cos(delta_psi)*np.sin(epsilon+delta_epsilon)*np.cos(epsilon)-np.cos(epsilon+delta_epsilon)*np.sin(epsilon), np.cos(delta_psi)*np.sin(epsilon+delta_epsilon)*np.sin(epsilon)+np.cos(epsilon+delta_epsilon)*np.cos(epsilon)]
    ])
    
    RN_inv = np.linalg.inv(RN)
    
    # Earth rotation matrix
    RS = np.array([
        [np.cos(GAST), np.sin(GAST), 0],
        [-np.sin(GAST), np.cos(GAST), 0],
        [0, 0, 1]
    ])
    
    RS_inv = np.linalg.inv(RS)
    
    # Polar motion matrix
    RM = np.array([
        [np.cos(xp), np.sin(xp)*np.sin(yp), np.sin(xp)*np.cos(yp)],
        [0, np.cos(yp), -np.sin(yp)],
        [-np.sin(xp), np.cos(xp)*np.sin(yp), np.cos(xp)*np.cos(yp)]
    ])
    
    RM_inv = np.linalg.inv(RM)
    
    X_eci = np.matmul(RP_inv,np.matmul(RN_inv,np.matmul(RS_inv,np.matmul(RM_inv,X_ecef))))
                         
    X_RS = np.matmul(RS_inv,X_ecef)
    X_RN = np.matmul(RN_inv,X_RS)
    X_RP = np.matmul(RP_inv,X_RN)
    X_TOD = np.matmul(RS_inv,np.matmul(RM_inv,X_ecef))
    
    return X_RS, X_RN, X_RP, X_eci, X_TOD

def ecef2tod(X_ecef, time):
    """
    Transform coordinates from Earth-Centred Earth-Fixed (ECEF) to Earth-Centred Inertial (ECI TOD).
    
    Parameters:
    X_ecef (np.array): 3D coordinates in ECEF frame [x, y, z]
    time (Time): Astropy Time object
    
    Returns:
    np.array: 3D coordinates in ECI frame
    """
    # Julian centuries from J2000.0
    T = (time.jd - 2451545.0) / 36525.0
    
    # Obtain true of date values
    values = get_tod_values(time)
    xp = values['xp']
    yp = values['yp']
    delta_psi = values['delta_psi']
    delta_epsilon = values['delta_epsilon']
    GAST = values['gast']
    
    # Earth rotation matrix
    RS = np.array([
        [np.cos(GAST), np.sin(GAST), 0],
        [-np.sin(GAST), np.cos(GAST), 0],
        [0, 0, 1]
    ])
    
    RS_inv = np.linalg.inv(RS)
    
    # Polar motion matrix
    RM = np.array([
        [np.cos(xp), np.sin(xp)*np.sin(yp), np.sin(xp)*np.cos(yp)],
        [0, np.cos(yp), -np.sin(yp)],
        [-np.sin(xp), np.cos(xp)*np.sin(yp), np.cos(xp)*np.cos(yp)]
    ])
    
    RM_inv = np.linalg.inv(RM)

    X_RM = np.matmul(RM_inv,X_ecef)
    X_RS = np.matmul(RS_inv,X_RM)
    X_TOD = np.matmul(RS_inv,np.matmul(RM_inv,X_ecef))
    
    return X_RM, X_RS, X_TOD

def tod2ecef(X_tod, time):
    """
    Transform coordinates from Earth-Centred Earth-Fixed (ECEF) to Earth-Centred Inertial (ECI).
    
    Parameters:
    X_ecef (np.array): 3D coordinates in ECEF frame [x, y, z]
    time (Time): Astropy Time object
    
    Returns:
    np.array: 3D coordinates in ECI frame
    """
    # Julian centuries from J2000.0
    T = (time.jd - 2451545.0) / 36525.0
    
    # Precession matrix components
    z = (2306.2181 * T + 1.09468 * T**2 + 0.018203 * T**3) * u.arcsec.to(u.rad)
    theta = (2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3) * u.arcsec.to(u.rad)
    zeta = (2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3) * u.arcsec.to(u.rad)
    
    # Obtain true of date values
    values = get_tod_values(time)
    xp = values['xp']
    yp = values['yp']
    delta_psi = values['delta_psi']
    delta_epsilon = values['delta_epsilon']
    GAST = values['gast']
    
    # Earth rotation matrix
    RS = np.array([
        [np.cos(GAST), np.sin(GAST), 0],
        [-np.sin(GAST), np.cos(GAST), 0],
        [0, 0, 1]
    ])
    
    # Polar motion matrix
    RM = np.array([
        [np.cos(xp), np.sin(xp)*np.sin(yp), np.sin(xp)*np.cos(yp)],
        [0, np.cos(yp), -np.sin(yp)],
        [-np.sin(xp), np.cos(xp)*np.sin(yp), np.cos(xp)*np.cos(yp)]
    ])

    X_ecef = np.matmul(RM,np.matmul(RS,X_tod))
    
    return X_ecef

def rv2coe(state_vec, mu = GM_earth.value):
    """
    Convert state vector to classical orbital elements.
    
    Parameters:
    state_vec (np.array): State vector in ECI frame [rx, ry, rz, vx, vy, vz] in km and km/s
    mu (float): Standard gravitational parameter [km^3/s^2]. Default is Earth's.
    
    Returns:
    dict: Classical orbital elements {p, a, e, i, Omega, omega, nu, u, lambda_true, omega_true}
    """
    r_vec = state_vec[:3]
    v_vec = state_vec[3:]
    
    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    K_vec = np.array([0, 0, 1])
    n_vec = np.cross(K_vec, h_vec)
    n = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    e_vec = (1/mu) * ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)
    
    # Energy
    xi = v**2 / 2 - mu / r
    
    # Semi-major axis (a) and semi-latus rectum (p)
    if e != 1.0:
        a = (-mu / (2 * xi))/1e3
        p = (a * (1 - e**2))/1e3
    else:
        p = h**2 / mu
        a = np.inf
    
    # Inclination (i)
    i = np.arccos(h_vec[2] / h) * u.rad.to(u.deg)
    
    # Right ascension of the ascending node (Omega)
    if n != 0:
        Omega = np.arccos(n_vec[0] / n) * u.rad.to(u.deg)
        if n_vec[1] < 0:
            Omega = 360 - Omega
    else:
        Omega = 0
    
    # Argument of periapsis (omega)
    if e != 0:
        omega = np.arccos(np.dot(n_vec, e_vec) / (n * e)) * u.rad.to(u.deg)
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0
    
    # True anomaly (nu)
    if e != 0:
        nu = np.arccos(np.dot(e_vec, r_vec) / (e * r)) * u.rad.to(u.deg)
        if np.dot(r_vec, v_vec) < 0:
            nu = 360 - nu
    else:
        nu = 0
    
    # Argument of latitude (aol) for circular inclined
    if e == 0 and i != 0:
        aol = np.arccos(np.dot(n_vec, r_vec) / (n * r)) * u.rad.to(u.deg)
        if r_vec[2] < 0:
            aol = 360 - u
    else:
        aol = None
    
    # True longitude (lambda_true) for circular equatorial
    if e == 0 and i == 0:
        lambda_true = np.arccos(r_vec[0] / r) * u.rad.to(u.deg)
        if r_vec[1] < 0:
            lambda_true = 360 - lambda_true
    else:
        lambda_true = None
    
    # Argument of periapsis (omega_true) for elliptical equatorial
    if e != 0 and i == 0:
        omega_true = np.arccos(e_vec[0] / e) * u.rad.to(u.deg)
        if e_vec[1] < 0:
            omega_true = 360 - omega_true
    else:
        omega_true = None

    return {
        'p': p,
        'a': a,
        'e': e,
        'i': i,
        'Omega': Omega,
        'omega': omega,
        'nu': nu,
        'aol': aol,
        'lambda_true': lambda_true,
        'omega_true': omega_true
    }

def coe2rv(a, ecc, inc, raan, argp, nu):
    # Vallado 143-147
    # Gravitational parameter for Earth (mu = 398600 km^3/s^2)
    mu = GM_earth.value/(10**9)

    # Convert angles from degrees to radians
    inc = inc.to(u.rad).value
    raan = raan.to(u.rad).value
    argp = argp.to(u.rad).value
    nu = nu.to(u.rad).value

    # Compute the semi-latus rectum
    p = a * (1 - ecc**2)

    # Helper function to compute rotation matrix
    def rot1(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])
    
    def rot3(theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    # Position and velocity in the PQW frame
    r_PQW = np.array([
        p.to(u.km).value * np.cos(nu) / (1 + ecc * np.cos(nu)),
        p.to(u.km).value * np.sin(nu) / (1 + ecc * np.cos(nu)),
        0
    ]) * u.km
    
    v_PQW = np.array([
        -np.sqrt(mu / p.to(u.km)).value * np.sin(nu),
        np.sqrt(mu / p.to(u.km)).value * (ecc + np.cos(nu)),
        0
    ]) * (u.km / u.s)
    
    # Transformation matrix from PQW to IJK frame
    PQW_to_IJK = np.dot(rot3(-raan), np.dot(rot1(-inc), rot3(-argp)))
    
    # Position and velocity in the IJK frame
    r_IJK_km = np.dot(PQW_to_IJK, r_PQW.value) * u.km
    v_IJK_km = np.dot(PQW_to_IJK, v_PQW.value) * (u.km / u.s)
    
    # Convert position to meters and velocity to meters per second
    r_IJK = r_IJK_km.to(u.m)
    v_IJK = v_IJK_km.to(u.m / u.s)
    
    return r_IJK, v_IJK

def site_track(phi_gd, lam, h_ellp, rho, el, beta, rho_dot, el_dot, beta_dot):
    """
    Perform site tracking calculation to transform site coordinates to ECEF coordinates.

    Parameters:
    phi_gd (float): Geodetic latitude [deg]
    lam (float): Longitude [deg]
    h_ellp (float): Ellipsoidal height [m]
    rho (float): Range [m]
    el (float): Elevation [deg]
    beta (float): Azimuth [deg]
    rho_dot (float): Range rate [m/s]
    el_dot (float): Elevation rate [deg/s]
    beta_dot (float): Azimuth rate [deg/s]

    Returns:
    np.array: ECEF position vector [m]
    np.array: ECEF velocity vector [m/s]
    """
    # Constants
    #e_earth = 0.081819190842622  # Earth's eccentricity
    e_earth = np.sqrt(0.00669437999014)
    R_earth_m = R_earth.to(u.m).value  # Earth's radius in m

    # Convert inputs to radians where necessary
    phi_gd = np.radians(phi_gd)
    lam = np.radians(lam)
    el = np.radians(el)
    beta = np.radians(beta)
    el_dot = np.radians(el_dot)
    beta_dot = np.radians(beta_dot)

    # Geodetic to ECEF conversion
    C_phi = R_earth_m / (np.sqrt(1 - (e_earth**2) * (np.sin(phi_gd)**2)))
    S_phi = C_phi * (1 - (e_earth**2))

    r_delta = (C_phi + h_ellp) * np.cos(phi_gd)
    r_k = (S_phi + h_ellp) * np.sin(phi_gd)

    r_site_ecef = np.array([r_delta * np.cos(lam), r_delta * np.sin(lam), r_k])

    # SEZ to ECEF conversion
    rho_sez = np.array([
        -rho * np.cos(el) * np.cos(beta),
         rho * np.cos(el) * np.sin(beta),
         rho * np.sin(el)
    ])

    rho_sez_dot = np.array([
        -rho_dot * np.cos(el) * np.cos(beta) + rho * np.sin(el) * np.cos(beta) * el_dot + rho * np.cos(el) * np.sin(beta) * beta_dot,
         rho_dot * np.cos(el) * np.sin(beta) - rho * np.sin(el) * np.sin(beta) * el_dot + rho * np.cos(el) * np.cos(beta) * beta_dot,
         rho_dot * np.sin(el) + rho * np.cos(el) * el_dot
    ])
    
    # Rotation Matrix SEZ to ECEF
    sez2ecef = np.array([
        [np.sin(phi_gd)*np.cos(lam), -np.sin(lam), np.cos(phi_gd)*np.cos(lam)],
        [np.sin(phi_gd)*np.sin(lam), np.cos(lam), np.cos(phi_gd)*np.sin(lam)],
        [-np.cos(phi_gd), 0, np.sin(phi_gd)]
    ])
    
    # Rotation from SEZ to ECEF
    #rho_ecef = sez2ecef @ rho_sez
    #rho_ecef_dot = sez2ecef @ rho_sez_dot
    rho_ecef = np.matmul(sez2ecef, rho_sez)
    rho_ecef_dot = np.matmul(sez2ecef, rho_sez_dot)

    # ECEF position and velocity
    r_ecef = rho_ecef + r_site_ecef
    v_ecef = rho_ecef_dot  # Note: We should add site's velocity here, simplified for brevity

    return r_ecef, v_ecef

def transform_dataframe_ssr(df, lat, lon, alt, el, az):
    ecef_columns = ['X_ECEF', 'Y_ECEF', 'Z_ECEF']
    tod_columns = ['X_RM', 'Y_RM', 'Z_RM', 'X_RS', 'Y_RS', 'Z_RS', 'X_TOD', 'Y_TOD', 'Z_TOD']
    
    
    # Initialize new columns
    for col in ecef_columns + tod_columns:
        df[col] = np.nan
    
    for index, row in df.iterrows():
        value = row['Value']
        date_time = row['DateTime']
        
        r_ecef, v_ecef = site_track(lat.value, lon.value, alt.value, value, el.value, az.value, 0, 0, 0)
        
        df.loc[index, 'X_ECEF'] = r_ecef[0]
        df.loc[index, 'Y_ECEF'] = r_ecef[1]
        df.loc[index, 'Z_ECEF'] = r_ecef[2]
        
        time = Time(date_time)
        X_RM, X_RS, X_TOD = ecef2tod(r_ecef, time)
        
        df.loc[index, 'X_RM'] = X_RM[0]
        df.loc[index, 'Y_RM'] = X_RM[1]
        df.loc[index, 'Z_RM'] = X_RM[2]
        df.loc[index, 'X_RS'] = X_RS[0]
        df.loc[index, 'Y_RS'] = X_RS[1]
        df.loc[index, 'Z_RS'] = X_RS[2]
        df.loc[index, 'X_TOD'] = X_TOD[0]
        df.loc[index, 'Y_TOD'] = X_TOD[1]
        df.loc[index, 'Z_TOD'] = X_TOD[2]
    
    return df

def transform_dataframe_tar(df, lat, lon, alt, el, az):
    ecef_columns = ['X_ECEF', 'Y_ECEF', 'Z_ECEF']
    tod_columns = ['X_RM', 'Y_RM', 'Z_RM', 'X_RS', 'Y_RS', 'Z_RS', 'X_TOD', 'Y_TOD', 'Z_TOD']
    
    # Initialize new columns
    for col in ecef_columns + tod_columns:
        df[col] = np.nan
    
    for index, row in df.iterrows():
        value = row['TA_Range']
        date_time = row['DateTime']
        
        r_ecef, v_ecef = site_track(lat.value, lon.value, alt.value, value, el.value, az.value, 0, 0, 0)
        
        df.loc[index, 'X_ECEF'] = r_ecef[0]
        df.loc[index, 'Y_ECEF'] = r_ecef[1]
        df.loc[index, 'Z_ECEF'] = r_ecef[2]
        
        time = Time(date_time)
        X_RM, X_RS, X_TOD = ecef2tod(r_ecef, time)
        
        df.loc[index, 'X_RM'] = X_RM[0]
        df.loc[index, 'Y_RM'] = X_RM[1]
        df.loc[index, 'Z_RM'] = X_RM[2]
        df.loc[index, 'X_RS'] = X_RS[0]
        df.loc[index, 'Y_RS'] = X_RS[1]
        df.loc[index, 'Z_RS'] = X_RS[2]
        df.loc[index, 'X_TOD'] = X_TOD[0]
        df.loc[index, 'Y_TOD'] = X_TOD[1]
        df.loc[index, 'Z_TOD'] = X_TOD[2]
    
    return df

def implicit(df):
    lla_columns = ['Sat_Lat', 'Sat_Lon', 'Sat_Alt']
    coe_columns = ['a', 'e', 'i', 'RAAN', 'AOP', 'nu']
    
    # Initialize new columns
    for col in lla_columns + coe_columns:
        df[col] = np.nan
    
    for index, row in df.iterrows():
        date_time = row['DateTime']
        time = Time(date_time)
        
        #r_ecef = np.array((row['X_ECEF'], row['Y_ECEF'], row['Z_ECEF']))
        r_tod = np.array((row['X_TOD'], row['Y_TOD'], row['Z_TOD']))
        r_ecef = tod2ecef(r_tod, time)
        
        lla = ecef2lla(r_ecef)
        
        df.loc[index, 'Sat_Lat'] = lla['latitude']
        df.loc[index, 'Sat_Lon'] = lla['longitude']
        df.loc[index, 'Sat_Alt'] = lla['altitude']
        
        r_tod = np.array((row['X_TOD'], row['Y_TOD'], row['Z_TOD'], row['Vx_TOD'], row['Vy_TOD'], row['Vz_TOD'])) 
        coe = rv2coe(r_tod)
        
        df.loc[index, 'a'] = coe['a']
        df.loc[index, 'e'] = coe['e']
        df.loc[index, 'i'] = coe['i']
        df.loc[index, 'RAAN'] = coe['Omega']
        df.loc[index, 'AOP'] = coe['omega']
        df.loc[index, 'nu'] = coe['nu']
    
    return df

def site_coord(df, lat, lon, alt):
    ecef_columns = ['X_ECEF', 'Y_ECEF', 'Z_ECEF']
    tod_columns = ['X_RM', 'Y_RM', 'Z_RM', 'X_RS', 'Y_RS', 'Z_RS', 'X_TOD', 'Y_TOD', 'Z_TOD']
    
    # Initialize new columns
    for col in ecef_columns + tod_columns:
        df[col] = np.nan
    
    for index, row in df.iterrows():
        value = row['Value']
        date_time = row['DateTime']
        
        r_ecef, v_ecef = site_track(lat.value, lon.value, alt.value, 0, 0, 0, 0, 0, 0)
        
        df.loc[index, 'X_ECEF_Site'] = r_ecef[0]
        df.loc[index, 'Y_ECEF_Site'] = r_ecef[1]
        df.loc[index, 'Z_ECEF_Site'] = r_ecef[2]
        
        time = Time(date_time)
        X_RM, X_RS, X_TOD = ecef2tod(r_ecef, time)
        
        df.loc[index, 'X_RM_Site'] = X_RM[0]
        df.loc[index, 'Y_RM_Site'] = X_RM[1]
        df.loc[index, 'Z_RM_Site'] = X_RM[2]
        df.loc[index, 'X_RS_Site'] = X_RS[0]
        df.loc[index, 'Y_RS_Site'] = X_RS[1]
        df.loc[index, 'Z_RS_Site'] = X_RS[2]
        df.loc[index, 'X_TOD_Site'] = X_TOD[0]
        df.loc[index, 'Y_TOD_Site'] = X_TOD[1]
        df.loc[index, 'Z_TOD_Site'] = X_TOD[2]
    
    return df

def ecef2razel(sat_ecef, gs_lat, gs_lon, gs_alt):
    """
    Transform ECEF coordinates to RAZEL.

    Parameters:
    sat_ecef (array): Satellite position in ECEF coordinates [x, y, z].
    gs_ecef (array): Ground station position in ECEF coordinates [x, y, z].
    gs_lat (float): Ground station latitude in radians.
    gs_lon (float): Ground station longitude in radians.

    Returns:
    dict: Dictionary containing range (rho), azimuth (az), and elevation (el).
    """
    # Constants
    rad2deg = 180.0 / np.pi
    
    # Convert inputs to radians where necessary
    phi_gd = np.radians(gs_lat)
    lam = np.radians(gs_lon)
    
    gs_ecef, v_ecef = site_track(gs_lat.value, gs_lon.value, gs_alt.value, 0, 0, 0, 0, 0, 0)

    # Calculate the relative position vector (rho_ECEF)
    rho_ecef = np.array(sat_ecef) - np.array(gs_ecef)

    # Define rotation matrices
    sez2ecef = np.array([
        [np.sin(phi_gd)*np.cos(lam), -np.sin(lam), np.cos(phi_gd)*np.cos(lam)],
        [np.sin(phi_gd)*np.sin(lam), np.cos(lam), np.cos(phi_gd)*np.sin(lam)],
        [-np.cos(phi_gd), 0, np.sin(phi_gd)]
    ])
    ecef2sez = np.linalg.inv(sez2ecef)
    #sez_ecef = np.array([[np.sin(gs_lat)*np.cos(gs_lon), np.sin(gs_lat)*np.sin(gs_lon), -np.cos(gs_lat)],
    #                     [-np.sin(gs_lon),               np.cos(gs_lon),                 0],
    #                     [np.cos(gs_lat)*np.cos(gs_lon), np.cos(gs_lat)*np.sin(gs_lon),  np.sin(gs_lat)]])
    

    # Transform the ECEF vector to SEZ vector
    rho_sez = np.matmul(ecef2sez, rho_ecef)

    # Calculate the range (rho)
    rho = np.linalg.norm(rho_sez)

    # Calculate elevation (el)
    el = np.degrees(np.arcsin(rho_sez[2] / rho))
    
    # Calculate the azimuth (az)
    az = np.degrees(np.arctan2(rho_sez[1], -rho_sez[0]))

    if az < 0:
        az += 360

    # Return the results
    return {
        'range': rho,
        'azimuth': az,
        'elevation': el
    }

def ecef2lla(ecef_coords):
    """
    Convert ECEF coordinates to latitude, longitude, and altitude.

    Parameters:
    ecef_coords (array): ECEF coordinates [x, y, z] in meters

    Returns:
    dict: Dictionary containing latitude (lat) in degrees, longitude (lon) in degrees, and altitude (alt) in meters
    """
    x, y, z = ecef_coords
    
    a = 6378137.0  # Semi-major axis (meters)
    e2 = 6.69437999014e-3  # Square of eccentricity
    
    # Calculate longitude (λ)
    lon = np.arctan2(y, x)

    # Intermediate parameter (p)
    p = np.sqrt(x**2 + y**2)
    
    # Initial guess for latitude (φ')
    lat = np.arctan2(z, p * (1 - e2))  # φ0
    lat_prev = 0
    
    # Iterative calculation of geodetic latitude (φ)
    while np.abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    # Calculate altitude (h)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = (p / np.cos(lat)) - N

    # Convert latitude and longitude to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    # Adjust longitude to be within the range [-180, 180] degrees
    if lon_deg > 180:
        lon_deg -= 360
    
    # Return the results
    return {
        'latitude': lat_deg,
        'longitude': lon_deg,
        'altitude': alt
    }
