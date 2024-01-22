import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


class _eval:
    def __init__(self,dir_gt):
        self.estimations = {}
        self.__count = 0
        
    def addEstimation(self, dir_es):
        data = np.loadtxt(dir_es, dtype=np.float64, comments='#', delimiter=",")
        self.estimations.update({self.__count: data})
        self.__count += 1
        
    def get_time_of_from_estimation(self,est_idx):
        return self.estimations[est_idx][:,0]
    
    def get_gyr_bias_from_estimation(self,est_idx):
        return self.estimations[est_idx][:,1:4]
    
    def get_acc_bias_from_estimation(self,est_idx):
        return self.estimations[est_idx][:,4:7]
        
    def plot_biases(self,ax):       
        rad_to_deg = 180 / np.pi
        sec_to_hour =  1 / 3600
        mg = 9.81*(1e-3)
        
        init_time = self.estimations[0][0,0]  # Choose a point to scale time
        
        font_size = 15
        for est_idx in range(self.__count):
            time = self.get_time_of_from_estimation(est_idx) - init_time
            gyr_bias = self.get_gyr_bias_from_estimation(est_idx) * rad_to_deg / sec_to_hour 
            acc_bias = self.get_acc_bias_from_estimation(est_idx) / mg
            
            ax[0,0].plot(time, gyr_bias[:,0],"--",linewidth=1.5)
            ax[0,1].plot(time, acc_bias[:,0],"--",linewidth=1.5)
            ax[0,0].set_ylabel(r'x',fontsize = font_size)
            ax[0,0].set_title(r'Gyr Bias (deg/hour)',fontsize = font_size)
            ax[0,1].set_title(r'Acc Bias (mg)',fontsize = font_size)
            
            ax[1,0].plot(time, gyr_bias[:,1],"--",linewidth=1.5)
            ax[1,1].plot(time, acc_bias[:,1],"--",linewidth=1.5)
            ax[1,0].set_ylabel(r'y',fontsize = font_size)

            ax[2,0].plot(time, gyr_bias[:,2],"--",linewidth=1.5)
            ax[2,1].plot(time, acc_bias[:,2],"--",linewidth=1.5)
            ax[2,0].set_ylabel(r'z',fontsize = font_size)


