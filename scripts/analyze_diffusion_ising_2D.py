import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from util.ising_2D import Ising2D

if __name__=='__main__':
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    iter_list = []
    energy_list = []
    mag_list = []
    
    for dirs in all_subdirs:
        if 'iter_' in dirs:
            dir_name = dirs
            iter_num_str = dir_name.replace('iter_', '')
            
            iter_num = int(iter_num_str)
            
            iter_list.append(iter_num)
            
            if os.path.isfile(os.path.join(dirs, "sample_x.npy")):
                file_path = os.path.join(dirs, "sample_x.npy")
            elif os.path.isfile(os.path.join(dirs, "sample.npy")):
                file_path = os.path.join(dirs, "sample.npy")
            
            state = np.load(file_path)
            
            state = np.reshape(state, (state.shape[-2], state.shape[-1]))
            
            ising = Ising2D(state=state)
            
            energy_list.append(ising.energy/state.size)
            mag_list.append(ising.mag/state.size)
            
    
    fig, axs = plt.subplots(1, 2)
    
    axs[0].set_ylim(-1,1)
    axs[1].set_ylim(0,1)
    
    axs[0].set_title("Energy")
    axs[1].set_title("|Magnetization|")
    
    axs[0].set_xlabel("iteration")
    axs[1].set_xlabel("iteration")

    axs[0].scatter(iter_list, energy_list, clip_on=False)
    axs[1].scatter(iter_list, np.absolute(np.array(mag_list)), clip_on=False)
    
    plt.savefig("e_and_m.png")
    plt.close()        