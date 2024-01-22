import matplotlib.pyplot as plt
from eval_tools import _pose
from eval_tools import _vel
from eval_tools import _bias
import os
            
if __name__ == "__main__":            
    dataset = "Part5"
    gt_dir = "/home/hakito/python_scripts/selcuk2/GimballiSelcuk/" + dataset + "/gt.csv"
    
    # Initialize our pose evaluation tool
    pose_eval = _pose._eval(gt_dir)
    pose_eval.show = False
    pose_eval.save = True
    pose_eval.plotVar = True
    pose_eval.haslabels = True
    
    # Initialize our velocity evaluation tool
    vel_eval = _vel._eval(gt_dir)
    vel_eval.show = False
    vel_eval.save = True
    vel_eval.plotVar = True
    vel_eval.haslabels = True
    
    # Initialize our bias evaluation tool
    bias_eval = _bias._eval(gt_dir)

    # Load datas to corresponding evaluation tools
    estimation_path = os.path.join("deneme","pose_estimate.csv")
    pose_eval.addEstimation(estimation_path,"vins")
    
    estimation_path = os.path.join("deneme","velocity_estimate.csv")
    vel_eval.addEstimation(estimation_path,"vins")
    
    estimation_path = os.path.join("deneme","bias_estimate.csv")
    bias_eval.addEstimation(estimation_path)

    saving_format = "png"

    ############ Position Errors ############
    # ECEF
    fig, ax = plt.subplots(3,figsize=(6,6))
    pose_eval.create_pos_error_ecef(ax)
    
    ax[0].legend()
    ax[2].set_xlabel("time (sec)")
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")
    ax[0].set_title("Position Errors versus Time")

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    file_name = "error_pos." + saving_format
    fig.savefig(file_name)
    
    
    ############ Orientation Errors ############
    fig, ax = plt.subplots(3,figsize=(6,6))
    pose_eval.create_attitude_error_ecef(ax)
    
    ax[0].legend()
    ax[2].set_xlabel("time (sec)")
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")
    ax[0].set_title("Attitude Error - Euler (xyz) - Degree")

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    file_name = "error_ori." + saving_format
    fig.savefig(file_name)
    
    ############ Position and Orientation Errors in NED ############
    fig_pos, ax_pos = plt.subplots(3,figsize=(6,6))
    fig_ori, ax_ori = plt.subplots(3,figsize=(6,6))
    pose_eval.create_errors_NED(ax_pos,ax_ori)
    
    
    ax_pos[0].legend()
    ax_pos[2].set_xlabel("time (sec)")
    ax_pos[0].set_ylabel("x")
    ax_pos[1].set_ylabel("y")
    ax_pos[2].set_ylabel("z")
    ax_pos[0].set_title("Position Errors versus Time (NED)")

    ax_pos[0].grid()
    ax_pos[1].grid()
    ax_pos[2].grid()
    
    file_name_pos = "error_pos_ned." + saving_format
    fig_pos.savefig(file_name_pos)
    
    ax_ori[0].legend()
    ax_ori[2].set_xlabel("time (sec)")
    ax_ori[0].set_ylabel("x")
    ax_ori[1].set_ylabel("y")
    ax_ori[2].set_ylabel("z")
    ax_ori[0].set_title("Attitude Error - Euler (xyz) - Degree (NED)")

    ax_ori[0].grid()
    ax_ori[1].grid()
    ax_ori[2].grid()
    
    file_name_ori = "error_ori_ned." + saving_format
    fig_ori.savefig(file_name_ori)
    
        
    # ############ Errors in NED ############
    # fig, ax = plt.subplots(3,3,figsize=(9,9))
    # file_name = "error_ned." + saving_format
    # eval.plot_errors_in_NED(ax[0,:],ax[1,:])
    # fig.savefig(file_name)
    
    
    
    ############ Velocity Errors ############
    fig, ax = plt.subplots(3,figsize=(6,6))
    vel_eval.create_error_plots(ax)
    ax[0].legend()
    ax[2].set_xlabel("time (sec)")
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")
    ax[0].set_title("Velocity Error versus Time (meter/sec)")

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    file_name = "error_vel." + saving_format
    fig.savefig(file_name)
    
    ############ Velocity Errors in NED ############
    fig, ax = plt.subplots(3,figsize=(6,6))
    vel_eval.plot_errors_in_NED(ax)
    
    ax[0].legend()
    ax[2].set_xlabel("time (sec)")
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_ylabel("z")
    ax[0].set_title("Velocity Error versus Time (meter/sec) (NED)")

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    file_name = "error_vel_ned." + saving_format
    fig.savefig(file_name)
    
    
    ############## Finally Create the Bias Plots ##############
    
    font_size = 15
    fig, ax = plt.subplots(3,2,figsize=(9,6))
    fig.suptitle(r'Bias Estimations',fontsize = font_size)
    fig.supxlabel(r'time $(sec)$',fontsize = font_size)
    bias_eval.plot_biases(ax)
    # Ad grids
    for h_i in range(2):
        for v_i in range(3):
            ax[v_i,h_i].grid()
    
    file_name = "bias." + saving_format
    fig.savefig(file_name)
    