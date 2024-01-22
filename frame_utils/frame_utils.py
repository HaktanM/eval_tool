from scipy.spatial.transform import Rotation as orientation
import numpy as np


def Radii_of_curvature(L):
        # Parameters
        R_0 = 6378137  # WGS84 Equatorial radius in meters
        e = 0.0818191908425  # WGS84 eccentricity

        # Calculate meridian radius of curvature using (2.105)
        temp = 1 - (e * np.sin(L))**2
        R_N = R_0 * (1 - e**2) / temp**1.5

        # Calculate transverse radius of curvature using (2.105)
        R_E = R_0 / np.sqrt(temp)

        return R_N, R_E

def get_C_ecef2ned(r_eb_e):
        # Parameters
        R_0 = 6378137  # WGS84 Equatorial radius in meters
        e = 0.0818191908425  # WGS84 eccentricity

        # Convert position using Borkowski closed-form exact solution
        # From (2.113)
        lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])

        # From (C.29) and (C.30)
        k1 = np.sqrt(1 - e**2) * np.abs(r_eb_e[2])
        k2 = e**2 * R_0
        beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
        E = (k1 - k2) / beta
        F = (k1 + k2) / beta

        # From (C.31)
        P = 4/3 * (E*F + 1)

        # From (C.32)
        Q = 2 * (E**2 - F**2)

        # From (C.33)
        D = P**3 + Q**2

        # From (C.34)
        V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)

        # From (C.35)
        G = 0.5 * (np.sqrt(E**2 + V) + E)

        # From (C.36)
        T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G

        # From (C.37)
        L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))

        # From (C.38)
        h_b = (beta - R_0 * T) * np.cos(L_b) + (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

        # Calculate ECEF to NED coordinate transformation matrix using (2.150)
        cos_lat = np.cos(L_b)
        sin_lat = np.sin(L_b)
        cos_long = np.cos(lambda_b)
        sin_long = np.sin(lambda_b)
        C_e_n = np.array([
            [-sin_lat * cos_long, -sin_lat * sin_long,  cos_lat],
            [-sin_long,            cos_long,           0],
            [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
        ])

        return C_e_n

def ecef2ned_pose(C_b_e,r_eb_e):
        # Parameters
        R_0 = 6378137  # WGS84 Equatorial radius in meters
        e = 0.0818191908425  # WGS84 eccentricity

        # Convert position using Borkowski closed-form exact solution
        # From (2.113)
        lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])

        # From (C.29) and (C.30)
        k1 = np.sqrt(1 - e**2) * np.abs(r_eb_e[2])
        k2 = e**2 * R_0
        beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
        E = (k1 - k2) / beta
        F = (k1 + k2) / beta

        # From (C.31)
        P = 4/3 * (E*F + 1)

        # From (C.32)
        Q = 2 * (E**2 - F**2)

        # From (C.33)
        D = P**3 + Q**2

        # From (C.34)
        V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)

        # From (C.35)
        G = 0.5 * (np.sqrt(E**2 + V) + E)

        # From (C.36)
        T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G

        # From (C.37)
        L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))

        # From (C.38)
        h_b = (beta - R_0 * T) * np.cos(L_b) + (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

        # Calculate ECEF to NED coordinate transformation matrix using (2.150)
        cos_lat = np.cos(L_b)
        sin_lat = np.sin(L_b)
        cos_long = np.cos(lambda_b)
        sin_long = np.sin(lambda_b)
        C_e_n = np.array([
            [-sin_lat * cos_long, -sin_lat * sin_long,  cos_lat],
            [-sin_long,            cos_long,           0],
            [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
        ])

        # Transform attitude using (2.15)
        C_b_n = np.dot(C_e_n, C_b_e)

        lat_lon_alt = np.array([L_b,lambda_b, h_b])

        return C_b_n,lat_lon_alt
    

    
def calculate_pose_errors_in_NED(est_C_b_n,est_lat_lon_alt,
                                true_C_b_n, true_lat_lon_alt):
        est_L_b = est_lat_lon_alt[0]
        est_lambda_b = est_lat_lon_alt[1]
        est_h_b = est_lat_lon_alt[2]

        true_L_b = true_lat_lon_alt[0]
        true_lambda_b = true_lat_lon_alt[1]
        true_h_b = true_lat_lon_alt[2]

        # Position error calculation, using (2.119)
        R_N, R_E = Radii_of_curvature(true_L_b)
        delta_r_eb_n = np.zeros(3)
        delta_r_eb_n[0] = (est_L_b - true_L_b) * (R_N + true_h_b)
        delta_r_eb_n[1] = (est_lambda_b - true_lambda_b) * (R_E + true_h_b) * np.cos(true_L_b)
        delta_r_eb_n[2] = -(est_h_b - true_h_b)

        # Attitude error calculation, using (5.109) and (5.111)
        delta_C_b_n = est_C_b_n @ true_C_b_n.T
        delta_eul_nb_n = orientation.from_matrix(delta_C_b_n).as_euler("xyz", degrees=True)

        return delta_eul_nb_n,delta_r_eb_n