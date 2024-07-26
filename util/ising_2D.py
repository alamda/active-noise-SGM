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
    def __init__(self, N=10, num_steps=None, temp=1.0):
        self.N = N
        self.num_steps = num_steps
        self.temp = temp
        self.beta = 1/self.temp
        
        self.state = self.init_state()
        
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
            self.state_list.append(self.state)
            
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
    
    N = 10
    num_steps = 10000
    temp = 2
    
    
    
    fig, axs = plt.subplots(1,2)
    
    temp_arr = np.linspace(0.01,5,100)
    
    for temp in temp_arr:
        myIsing = Ising2D(N=N, num_steps=num_steps, temp=temp)
        
        axs[0].scatter(temp, myIsing.energy_list[-1]/N**2)
        axs[1].scatter(temp, myIsing.mag_list[-1]/N**2)
    
    plt.show()
    
        