import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from util.ising_2D import Ising2D

def get_data():
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    iter_list = []
    energy_list = []
    mag_list = []
    
    if len(all_subdirs) > 0:
        plot_color='#1f77b4'
        for dirs in all_subdirs:
            if 'iter_' in dirs:
                title = f'Diffusion Samples (Training Cumulative)'
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
    else:
        plot_color='#ff7f0e'
        title = f'Training Data'
        
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        
        for file in files:
            if '.npy' in file:
                iter_num_str = file.replace('train_', '').replace('.npy', '')
                iter_num = int(iter_num_str)
                
                iter_list.append(iter_num)
                
                state = np.load(file)
                
                state = np.reshape(state, (state.shape[-2], state.shape[-1]))
                
                ising = Ising2D(state=state)
                
                energy_list.append(ising.energy/state.size)
                mag_list.append(ising.mag/state.size)
    return iter_list, energy_list, mag_list, title, plot_color

def plot_e_and_m(iter_list, energy_list, mag_list, title, plot_color):
    fig, axs = plt.subplots(1, 2, layout='constrained')
    
    if title is not None:
        fig.suptitle(title)
    
    axs[0].set_ylim(-1,1)
    axs[1].set_ylim(0,1)
    
    axs[0].set_title("Energy")
    axs[1].set_title("|Magnetization|")
    
    axs[0].set_xlabel("iteration")
    axs[1].set_xlabel("iteration")

    axs[0].scatter(iter_list, energy_list, clip_on=False, color=plot_color)
    axs[1].scatter(iter_list, np.absolute(np.array(mag_list)), clip_on=False, color=plot_color)
    
    plt.savefig("e_and_m.png")
    plt.close()

def plot_e_and_m_hist(energy_list, mag_list, title, plot_color):
    fig, axs = plt.subplots(1, 2, layout='constrained')
    
    if title is not None:
        fig.suptitle(title)
    
    num_bins = 10
    
    axs[0].set_ylim(0,3.5)
    axs[1].set_ylim(0,3.5)
    
    axs[0].set_title("Energy")
    axs[1].set_title("|Magnetization|")

    axs[0].hist(energy_list, bins=num_bins, range=(-4,0), density=True, color=plot_color)
    axs[1].hist(mag_list, bins=num_bins, range=(-1,1), density=True, color=plot_color)
    
    plt.savefig("hist_e_and_m.png")
    plt.close()

    
def plot_mag_hist(mag_list, title, plot_color):
    fig, ax = plt.subplots(1, layout='constrained')
    
    fig.set_size_inches(3.5, 4.8)
    
    if title is not None:
        fig.suptitle(title)
    
    num_bins = 10

    ax.set_ylim(0,3.5)
    ax.set_xlabel("|Magnetization|")

    ax.hist(mag_list, bins=num_bins, range=(-1,1), density=True, color=plot_color)
    
    plt.savefig("hist_mag.png")
    plt.close()
    
if __name__=='__main__':
    iter_list, energy_list, mag_list, title, plot_color = get_data() 
    
    plot_mag_hist(mag_list, title, plot_color)  