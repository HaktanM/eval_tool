import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from scipy.spatial.transform import Rotation as orientation

from frame_utils import frame_utils

# Accessing the default color cycle
import matplotlib
color_cycle = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


    

def perform_assocation(ref_time_arr, target_time_arr, target_data):
    ref_size = ref_time_arr.shape[0]
    target_alignment_statu = np.zeros(ref_size,dtype=bool)
    target_aligned_data = np.zeros((ref_size,target_data.shape[1]),dtype=float)

    max_difference = 0.02
    for ref_time_index in range(ref_size):
        curr_ref_time = ref_time_arr[ref_time_index]
        difference_array = np.absolute(target_time_arr-curr_ref_time)
        best_target_idx = difference_array.argmin()
        best_diff = difference_array[best_target_idx]
        if best_diff <= max_difference:
            target_alignment_statu[ref_time_index] = True
            target_aligned_data[ref_time_index,:] = target_data[best_target_idx,:]
        else:
            target_alignment_statu[ref_time_index] = False
    return target_alignment_statu, target_aligned_data


class _eval:
    def __init__(self,dir_gt):

        data = np.loadtxt(dir_gt, dtype=np.float64, comments='#', delimiter=",")


        self.time = data[:,0] * 1e-9
        self.gt = data[:,1:11] 
        self.dataLength = self.time.shape[0]
        
        self.estimations = {}
        self.__assocData = {}
        self.__assocStatus = {}
        self.__assocAll = np.ones(self.dataLength,dtype=bool)    

        
        self.show = False
        self.plotVar = True
        self.save = True
        
        self.num_of_estimated_var = 100

        self.haslabels = False
        self.labels = {}
        
        self.__hasCov = True
        self.__association_performed = False
        self.__estimations_are_cropped = False
        self.__count = 0
        
        
    def addEstimation(self, dir_es, label="non"):
        data = np.loadtxt(dir_es, dtype=np.float64, comments='#', delimiter=",")
        self.estimations.update({self.__count: data})
        self.labels.update({self.__count:label})
 
        self.num_of_estimated_var = np.minimum(self.num_of_estimated_var,data.shape[1]-1)
        
        self.__count += 1
        self.__association_performed = False
        self.__estimations_are_cropped = False

    
    def performAssociation(self):
        for i in range(self.__count):
            gt_time = self.time
            es_time = self.estimations[i][:,0]
            es_data = self.estimations[i][:,1:]
            assocStatu, assocData = perform_assocation(gt_time, es_time, es_data)
            self.__assocData.update({i:assocData})
            self.__assocStatus.update({i:assocStatu})
            self.__assocAll = np.logical_and(self.__assocAll,assocStatu)
        self.__association_performed = True
        
    
    def crop_estimations(self):
        if not self.__association_performed:
            self.performAssociation()
            
        self.__assocData_cropped = {}        
        
        time = self.time
        gt = self.gt
        cropped_time = np.zeros(0)
        cropped_gt = np.zeros(0)
        assoc_counter = 0
        for idx,associtaion_statu in enumerate(self.__assocAll):
            if associtaion_statu:
                cropped_time = np.append(cropped_time,time[idx])
                cropped_gt = np.append(cropped_gt,gt[idx,:])
                assoc_counter +=1
                
        cropped_gt = cropped_gt.reshape(assoc_counter,gt.shape[1])
        self.__cropped_gt = cropped_gt
        self.__cropped_time = cropped_time

        
        croped_estimations = np.zeros((assoc_counter,self.num_of_estimated_var,self.__count))
        """
        First index stands for the time where the association is happened.
        Second index is the variables belonging to a single estimation such as pose, velocity, covariances
        Third term is the estimation index
        """
        
        
        """
        The loop checks where the association is common for all estimations.
        Only the associated data is saved in croped_estimations. The rest is cropped.
        """  
        assoc_idx = 0
        for time_idx,associtaion_statu in enumerate(self.__assocAll):
            if associtaion_statu:
                for est_idx in range(self.__count):
                    estimation = self.__assocData[est_idx][:,0:self.num_of_estimated_var]
                    croped_estimations[assoc_idx,:,est_idx] = estimation[time_idx,:]
                assoc_idx += 1
              
                
        for est_idx in range(self.__count):
            self.__assocData_cropped.update({est_idx:croped_estimations[:,:,est_idx]})
            
        self.__estimations_are_cropped = True
        
    
    def get_linear_velocity_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,0:3]
    
    def get_linear_velocity_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,3:12]
    
    def create_error_plots(self,ax):
        if not self.__association_performed:
            self.performAssociation()
            
        if not self.__estimations_are_cropped:
            self.crop_estimations()
        
        gt_velocity = self.__cropped_gt[:,7:10]
        plot_time = (self.__cropped_time - self.__cropped_time[0])
        
        self.standard_deviations = np.zeros((plot_time.shape[0],3, self.__count), dtype=np.float32)
        self.errors = np.zeros((plot_time.shape[0],3, self.__count), dtype=np.float32)
        """
        First index stands for the time 
        Second index is representing the estimation axis, such as xyz
        Third term is the estimation index
        """

        for est_idx in range(self.__count):
            # First Plot The Error
            error = self.get_linear_velocity_from_estimation(est_idx) - gt_velocity
            self.errors[:,:,est_idx] = error

            # Now handle the standard deviations
            velCov = self.get_linear_velocity_covariance_from_estimation(est_idx)
            for time_idx in range(plot_time.shape[0]):
                cov = velCov[time_idx,:].reshape(3,3)
                sigma_x = np.sqrt(cov[0,0])
                sigma_y = np.sqrt(cov[1,1])
                sigma_z = np.sqrt(cov[2,2])
                
                self.standard_deviations[time_idx,0,est_idx] += sigma_x
                self.standard_deviations[time_idx,1,est_idx] += sigma_y
                self.standard_deviations[time_idx,2,est_idx] += sigma_z    
        
        self.fig, self.ax = plt.subplots(3,figsize=(6,6))
        self.plot_time = plot_time
        self.create_three_axis_plot(ax)
        

        
            
    def plot_errors_in_NED(self,ax):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
            
        plot_time = (self.__cropped_time - self.__cropped_time[0])       

        gt_velocity = self.__cropped_gt[:,7:10]
        gt_position = self.__cropped_gt[:,0:3]
              
        vel_errors_ned = np.zeros((plot_time.shape[0],3,self.__count))
        """
        In the error matrices, first index stands for the time
        Second index stands for the estimation axis, i.e. x, y, z
        Third axis stands for the estimation id
        """  

        vel_sd_ned = np.zeros((plot_time.shape[0],3,self.__count))
        """
        First index stands for the time
        Second index stands for the estimation axis, i.e. x, y, z
        Third axis stands for the estimation id
        """


        for est_idx in range(self.__count):
 
            est_v_eb_e_all = self.get_linear_velocity_from_estimation(est_idx)
            velCov = self.get_linear_velocity_covariance_from_estimation(est_idx)

            for time_idx in range(plot_time.shape[0]):
                # Read the velovity current velocities first
                est_v_eb_e = est_v_eb_e_all[time_idx,:].reshape(3,1)
                gt_v_eb_e = gt_velocity[time_idx,:].reshape(3,1)
                
                ### Express the estimted velocity in NED frame
                # We utilize the ground-truth pose to calculate the ecef2ned transformation matrix
                gt_p_eb_e = gt_position[time_idx,:].reshape(-1)
                C_ecef_ned = frame_utils.get_C_ecef2ned(gt_p_eb_e).reshape(3,3)
                est_v_nb_n = C_ecef_ned @ est_v_eb_e
                
                ### Express the groun-truth velocity in NED frame
                gt_v_nb_n = C_ecef_ned @ gt_v_eb_e
                
                ### Convert the covariance matrices from ecef to ned
                C_vel_ecef = velCov[time_idx,:].reshape(3,3)
                C_vel_ned = (C_ecef_ned @ C_vel_ecef) @ C_ecef_ned.transpose()
                
                
                ### Save the values
                vel_errors_ned[time_idx,:,est_idx] = (est_v_nb_n - gt_v_nb_n).reshape(-1)
                vel_sd_ned[time_idx,:,est_idx] = np.sqrt(np.array([C_vel_ned[0,0],C_vel_ned[1,1],C_vel_ned[2,2]]) ).reshape(-1)

        self.errors = vel_errors_ned
        self.standard_deviations = vel_sd_ned
        self.plot_time = plot_time
        self.create_three_axis_plot(ax)
        
            

    def create_three_axis_plot(self,ax):
        for est_idx in range(self.__count):           
            ax[0].plot(self.plot_time, self.errors[:,0,est_idx],linewidth=1.5,color=color_cycle[est_idx])
            ax[1].plot(self.plot_time, self.errors[:,1,est_idx],linewidth=1.5,color=color_cycle[est_idx])
            ax[2].plot(self.plot_time, self.errors[:,2,est_idx],linewidth=1.5,color=color_cycle[est_idx])
                
            ax[0].plot(self.plot_time, self.standard_deviations[:,0,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])
            ax[0].plot(self.plot_time, -self.standard_deviations[:,0,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])

            ax[1].plot(self.plot_time, self.standard_deviations[:,1,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])
            ax[1].plot(self.plot_time, -self.standard_deviations[:,1,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])

            ax[2].plot(self.plot_time, self.standard_deviations[:,2,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])
            ax[2].plot(self.plot_time, -self.standard_deviations[:,2,est_idx],"--",linewidth=1.5,color=color_cycle[est_idx])  