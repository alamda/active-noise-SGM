import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt

import sys

sys.path.insert(0, '../')

def inf_data_gen(dataset, batch_size):
    if dataset in ('diamond', 'diamond_close'):
        WIDTH = 3
        if dataset == 'diamond':
            BOUND = 0.5
        elif dataset == 'diamond_close':
            BOUND = 0.2
        NOISE = 0.04
        ROTATION_MATRIX = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)

        means = np.array([(x, y) for x in np.linspace(-BOUND, BOUND, WIDTH)
                          for y in np.linspace(-BOUND, BOUND, WIDTH)])
        means = means @ ROTATION_MATRIX
        
        covariance_factor = NOISE * np.eye(2)

        index = np.random.choice(
            range(WIDTH ** 2), size=batch_size, replace=True)
        noise = np.random.randn(batch_size, 2)
        data = means[index] + noise @ covariance_factor
        return means, NOISE, torch.from_numpy(data.astype('float32'))
    
    elif dataset in ('multigaussian_2D', 'multigaussian_2D_close'):
        if dataset == 'multigaussian_2D':
            mu_x_list = [-1.2, 1.2]
        elif dataset == 'multigaussian_2D_close':
            mu_x_list = [-1.0, 1.0]
            
        NOISE = 0.5 
        mu_y_list = [0., 0.]
        sigma_list = [NOISE, NOISE]
        pi_list = [0.5, 0.5]

        means_x = np.array(mu_x_list)
        means_y = np.array(mu_y_list)
        sigmas = np.array(sigma_list)
        weights = np.array(pi_list)/np.sum(np.array(pi_list))

        index_x = np.random.choice(range(len(mu_x_list)), 
                                   size=batch_size, 
                                   replace=True,
                                   p=weights)
        
        index_y = np.random.choice(range(len(mu_y_list)), 
                            size=batch_size, 
                            replace=True,
                            p=weights)
        
        noise_x = np.random.randn(batch_size, 1)
        noise_y = np.random.randn(batch_size, 1)

        data_x = means_x[index_x] + noise_x.flatten() * sigmas[index_x]
        data_y = means_y[index_y] + noise_y.flatten() * sigmas[index_y]
        
        data_x = torch.from_numpy(data_x.reshape((-1, 1)))
        data_y = torch.from_numpy(data_y.reshape((-1, 1)))
        
        data = torch.cat((data_x, data_y), dim=1)
        
        means = np.column_stack((np.array(mu_x_list), np.array(mu_y_list)))
        
        return means, NOISE, data

def passive_analytic_score(x=None, Tp=None, mu_list=None, sigma_list=None, pi_list=None, t=None):
    x = x.numpy()
    a = np.exp(-t)
    Delta = Tp*(1-np.exp(-2*t))
    Fx_num = np.zeros_like(x)
    Fx_den = np.zeros_like(x)

    for idx, mu in enumerate(mu_list):
        mu = mu.flatten()
        sigma = sigma_list[idx].flatten()
        pi = pi_list[idx].flatten()
        
        h = sigma**2
        
        Delta_eff = a*a*h + Delta
        
        q = np.power(np.prod(Delta_eff), -0.5)

        z = np.exp((-np.sum((x-a*mu)**2/(2*Delta_eff), axis=0)))
        
        Fx_num = Fx_num - ((pi/q)/Delta_eff)*(x-a*mu)*z
        Fx_den = Fx_den + (pi/q)*z
    
    Fx = Fx_num/Fx_den
    return Fx

def passive_prior(shape=None, Tp=1, k=1):
    return np.sqrt(Tp / k) * torch.randn(*shape)

def reverse_passive(x=None, Tp=1, dt=None, noise_level=100, mu_list=None, sigma_list=None, pi_list=None, t_idx=None):
    t=t_idx*dt
    
    Fx = passive_analytic_score(x=x, Tp=Tp, mu_list=mu_list, sigma_list=sigma_list, pi_list=pi_list, t=t)
    x = x + dt*x + 2*Tp*Fx*dt + np.sqrt(2*Tp*dt)*torch.randn_like(x)
    
    return x, Fx

###

def M_11_12_22(Tp=None, Ta=None, tau=None, k=None, t=None):
    a = np.exp(-k*t)
    b = np.exp(-t/tau)
    Tx = Tp
    Ty = Ta/(tau*tau)
    w = (1/tau)
    M11 = (1/k)*Tx*(1-a*a) + (1/k)*Ty*( 1/(w*(k+w)) + 4*a*b*k/((k+w)*(k-w)**2) - (k*b*b + w*a*a)/(w*(k-w)**2) )
    M12 = (Ty/(w*(k*k - w*w))) * ( k*(1-b*b) - w*(1 + b*b - 2*a*b) )
    M22 = (Ty/w)*(1-b*b)
    return M11, M12, M22

def active_analytic_score(x=None, eta=None, Tp=0, Ta=1, tau=0.25, mu_list=None, sigma_list=None, pi_list=None, k=1, t=None):
    x = x.numpy()
    eta = eta.numpy()
    
    a = np.exp(-k*t)
    b = (np.exp(-t/tau) - np.exp(-k*t))/(k-(1/tau))
    c = np.exp(-t/tau)
    g = Ta/tau
    
    M11, M12, M22 = M_11_12_22(Tp=Tp, Ta=Ta, tau=tau, k=k, t=t)
    
    Fx_num = np.zeros_like(x)
    Fx_den = np.zeros_like(x)
    Feta_num = np.zeros_like(eta)
    Feta_den = np.zeros_like(eta)
    
    for idx, mu in enumerate(mu_list):
        mu = mu.flatten()
        sigma = sigma_list[idx].flatten()
        pi = pi_list[idx].flatten()
        
        h = sigma**2
        
        K1 = c*c*g + M22
        K2 = b*c*g + M12
        K3 = b*b*g + a*a*h + M11
        
        Delta_eff = K1*K3 - K2*K2
        # Delta_eff = Delta_eff.reshape(-1, 1)

        q = np.prod(h/np.sqrt(Delta_eff))

        z = np.exp(np.sum((-K1*(x-a*mu)**2 + 2*K2*(x-a*mu)*eta - K3*eta**2)/(2*Delta_eff), axis=0)) # This leads to numerical instability as it is very close to 0 if t is small and we start far from the true distribution
        eh = np.sum((-K1*(x-a*mu)**2 + 2*K2*(x-a*mu)*eta - K3*eta**2)/(2*Delta_eff), axis=0)
        print(eh, np.exp(eh))
        print(a,b,c,g)
        breakpoint()
        
        Fx_num = Fx_num + pi*q*(K2*eta - K1*(x-a*mu))*z/Delta_eff
        Fx_den = Fx_den + pi*q*z
        Feta_num = Feta_num + pi*q*(K2*(x-a*mu) - K3*eta)*z/Delta_eff
        Feta_den = Feta_den + pi*q*z
    Fx = Fx_num/Fx_den
    Feta = Feta_num/Feta_den
    return Fx, Feta

def active_prior(shape=None, Tp=0, Ta=1, k=1, tau=0.25):
    var_11 = 1/k * (Tp + Ta/(1+ k*tau))
    var_12 = Ta/(1 + tau*k)
    var_22 = Ta / tau
    
    zero_mean = torch.zeros(2)
    
    covar = torch.tensor([var_11, var_12, var_12, var_22])
    covar = torch.reshape(covar, (2,2))
    
    sampler = MultivariateNormal(loc=zero_mean, covariance_matrix=covar)
    
    sample_1 = sampler.sample(sample_shape=torch.Size([shape[0]]))
    sample_2 = sampler.sample(sample_shape=torch.Size([shape[0]]))
    
    sample_1x, sample_1eta = torch.chunk(sample_1, 2, dim=1)
    sample_2x, sample_2eta = torch.chunk(sample_2, 2, dim=1)
    
    if shape[1] == 2:
        sample_x = torch.cat((sample_1x, sample_2x), dim=1)
        sample_eta = torch.cat((sample_1eta, sample_2eta), dim=1)
    if shape[1] == 1:
        sample_x = sample_1x
        sample_eta = sample_1eta
    
    return sample_x, sample_eta

def reverse_active(x=None, y=None, Tp=0, Ta=1, tau=0.25, mu_list=None, sigma_list=None, pi_list=None, t_idx=None, dt=None, noise_level=100):
    t=t_idx*dt
    print(t)
    
    Fx, Fy = active_analytic_score(x=x, eta=y, Tp=Tp, Ta=Ta, tau=tau, mu_list=mu_list, sigma_list=sigma_list, pi_list=pi_list, t=t)
    x = x + dt*(x-y) + 2*Tp*Fx*dt + np.sqrt(2*Tp*dt)*torch.randn_like(x)
    y = y + dt*y/tau + (2*Ta/(tau*tau))*(Fy)*dt + (1/tau)*np.sqrt(2*Ta*dt)*torch.randn_like(y)
    
    # print(Fy)
    
    return x, y, Fx, Fy

def plot_score_hist(dataset=None):
    batch_size = 100
    # dataset = "multigaussian_2D"
    dt = 0.01
    num_steps = 100
    
    mu_list, sigma, data = inf_data_gen(dataset, batch_size)
    
    sigma_list = sigma*np.ones_like(mu_list)
    pi_list = (1/mu_list.size)*np.ones_like(mu_list)
    
    # Fx = passive_analytic_score(x=data, Tp=1, mu_list=mu_list, sigma_list=sigma_list, pi_list=pi_list, t=0)

    # eta = torch.ones_like(data)
    
    # Fx, Feta = active_analytic_score(x=data, eta=eta, mu_list=mu_list, sigma_list=sigma_list, pi_list=pi_list, t=0)
    
    x = passive_prior(shape=data.shape)
    
    # x, eta = active_prior(shape=data.shape)

    for t_idx in range(num_steps, 0, -1):
        x, Fx = reverse_passive(x=x, dt=dt, mu_list=mu_list, sigma_list=sigma_list, pi_list=pi_list, t_idx=t_idx)
        # print(Fx)
        
        score=Fx
        
        fig, ax = plt.subplots()
        
        time = "{:0.2f}".format(t_idx*dt)

        ax.set_title(f"ANALYTIC passive, {dataset}, t={time}")
        ax.set_ylim(0,1)
        ax.set_xlim(-5, 5)
        fake_x_arr = np.linspace(-5, 5, 10000)
        fake_y_arr = np.exp(-(fake_x_arr)**2)/np.sqrt(2*np.pi)
    
        ax.plot(fake_x_arr, fake_y_arr, label="N(0,I)", color='black')
        ax.hist(np.array(score[:,0]), bins = 50, density=True, range=(-5, 5), label="x", alpha=0.5)
        ax.hist(np.array(score[:,1]), bins = 50, density=True, range=(-5, 5), label="y", alpha=0.5)
        ax.legend()
        
        fname = f"ANALYTIC_passive_{dataset}_t{time}.png"
            
        plt.savefig(fname)
    
        plt.close()
        
    x, eta = active_prior(shape=data.shape)
    
    for t_idx in range(num_steps, 0, -1):
        x, eta, Fx, Feta = reverse_active(x=x, y=eta, mu_list=mu_list, sigma_list=pi_list, pi_list=pi_list, t_idx=t_idx, dt=dt)
    
        score=Feta
        
        fig, ax = plt.subplots()
        
        time = "{:0.2f}".format(t_idx*dt)

        ax.set_title(f"ANALYTIC active, {dataset}, t={time}")
        ax.set_ylim(0,1)
        ax.set_xlim(-5, 5)
        fake_x_arr = np.linspace(-5, 5, 10000)
        fake_y_arr = np.exp(-(fake_x_arr)**2)/np.sqrt(2*np.pi)
    
        ax.plot(fake_x_arr, fake_y_arr, label="N(0,I)", color='black')
        ax.hist(np.array(score[:,0]), bins = 50, density=True, range=(-5, 5), label="x", alpha=0.5)
        ax.hist(np.array(score[:,1]), bins = 50, density=True, range=(-5, 5), label="y", alpha=0.5)
        ax.legend()
        
        fname = f"ANALYTIC_active_{dataset}_t{time}.png"
            
        plt.savefig(fname)
    
        plt.close()

if __name__=="__main__":
    datasets = [ "multigaussian_2D",
                 "multigaussian_2D_close", 
                 "diamond",
                 "diamond_close"]
    
    for dataset in datasets:
        plot_score_hist(dataset=dataset)
    
