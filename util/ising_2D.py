import numpy as np
from torch.utils.data import Dataset
import torch

class Ising2DDataset(Dataset):
    def __init__(self, beta=None, N=None, num_steps=None, num_samples=None):
        self.beta = beta
        self.N = N
        self.num_steps = num_steps
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def init_state(self):
        return 2*np.random.randint(2, size=(self.N,self.N)) - 1
    
    def calc_num_bonds(self, config=None, i=None, j=None):
        num_bonds = config[(i+1)%self.N, j] + \
                    config[i, (j+1)%self.N] + \
                    config[(i-1)%self.N, j] + \
                    config[i, (j-1)%self.N]
                    
        return num_bonds
    
    def mc_steps(self, state=None, num_steps=1):
        for _ in range(num_steps):
            a = np.random.randint(0, self.N)
            b = np.random.randint(0, self.N)
            
            s = state[a,b]
            
            num_bonds = self.calc_num_bonds(config=state, i=a, j=b)
            
            cost = 2*s*num_bonds
            
            if cost < 0:
                s *= -1
            elif np.random.uniform() < np.exp(-cost*self.beta):
                s *= -1
                
            state[a,b] = s
        
        return state
    
    def __getitem__(self, idx):
        state = self.mc_steps(state=self.init_state(), num_steps=self.num_steps)
        
        return torch.from_numpy(state.astype(np.float64)).reshape(1, self.N, self.N), self.beta

class Ising2D:
    def __init__(self, N=10, num_steps=None, temp=1.0, state=None):
        self.N = N
        self.num_steps = num_steps
        self.temp = temp
        self.beta = 1/self.temp
        
        if state is None:
            self.state = self.init_state()
        else:
            self.state = state
        
        self.state_list = []
        
        self.energy = self.calc_energy()
        self.mag = self.calc_mag()
        
        self.energy_list = [self.energy]
        self.mag_list = [self.mag]
        
        if self.num_steps is not None:
            self.mc_steps(num_steps=self.num_steps)

    def init_state(self):
        return 2*np.random.randint(2, size=(self.N,self.N)) - 1
    
    def calc_num_bonds(self, config=None, i=None, j=None):
        num_bonds = config[(i+1)%self.N, j] + \
                    config[i, (j+1)%self.N] + \
                    config[(i-1)%self.N, j] + \
                    config[i, (j-1)%self.N]
                    
        return num_bonds
    
    def mc_steps(self, num_steps=1):
        for _ in range(num_steps):
            a = np.random.randint(0, self.N)
            b = np.random.randint(0, self.N)
            
            s = self.state[a,b]
            
            num_bonds = self.calc_num_bonds(config=self.state, i=a, j=b)
            
            cost = 2*s*num_bonds
            
            if cost < 0:
                s *= -1
            elif np.random.uniform() < np.exp(-cost*self.beta):
                s *= -1
                
            self.state[a,b] = s
            state = self.state.copy()
            self.state_list.append(state)
            
            self.energy_list.append(self.calc_energy())
            self.mag_list.append(self.calc_mag())
        
    def calc_energy(self):
        energies = [-self.calc_num_bonds(config=self.state, i=i, j=j)*self.state[i,j] for i in range(self.N) for j in range(self.N)]
        self.energy = np.sum(energies)/4

        return self.energy
    
    def calc_mag(self):
        self.mag = np.sum(self.state)
        return self.mag
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    N = 32
    num_steps = 100000
    
    temp_arr = np.linspace(0.01,5,10)
    
    energy_list = []
    mag_list = []

    for idx, temp in enumerate(temp_arr):
        myIsing = Ising2D(N=N, num_steps=num_steps, temp=temp)
        
        energy_list.append(myIsing.energy_list[-1]/N**2)
        mag_list.append(abs(myIsing.mag_list[-1])/N**2)
        
        fig_lattice, axs_lattice = plt.subplots(1,3)
        
        fig_lattice.suptitle(f"T={temp}")
        
        axs_lattice[0].imshow(myIsing.state_list[0], cmap='binary')
        axs_lattice[0].set_title("iter 0")
        
        axs_lattice[1].imshow(myIsing.state_list[len(myIsing.state_list)//2], cmap='binary')
        axs_lattice[1].set_title(f"iter {len(myIsing.state_list)//2}")
        
        axs_lattice[2].imshow(myIsing.state_list[-1], cmap='binary')
        axs_lattice[2].set_title(f"iter {len(myIsing.state_list)}")
        
        figname_lattice = f"lattice_T{temp:.2f}_N{N}_{num_steps}.png"
        plt.savefig(figname_lattice)
        plt.close()
        
        print(f"({idx+1}/{temp_arr.size}) {temp:.2f} done")
    
    fig, axs = plt.subplots(1,2)
    
    axs[0].set_ylim(-1,1)
    axs[1].set_ylim(0,1)
    
    axs[0].scatter(temp_arr, energy_list, clip_on=False)
    axs[1].scatter(temp_arr, mag_list, clip_on=False)
    
    axs[0].set_ylabel('energy')
    axs[1].set_ylabel('magnetization')
    
    axs[0].set_xlabel('temperature')
    axs[1].set_xlabel('temperature')
    
    figname = f"ising_N{N}_{num_steps}.png"
    plt.savefig(figname)
    plt.close()
    
        