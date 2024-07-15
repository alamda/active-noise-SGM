# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from util.utils import add_dimensions

from torch.distributions.multivariate_normal import MultivariateNormal

class CLD(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn
        self.m_inv = config.m_inv
        self.f = 2. / np.sqrt(config.m_inv)
        self.g = 1. / self.f
        self.gamma = config.gamma
        self.numerical_eps = config.numerical_eps

    @property
    def type(self):
        return 'cld'

    @property
    def is_augmented(self):
        return True

    def sde(self, u, t):
        '''
        Evaluating drift and diffusion of the SDE.
        '''
        x, v = torch.chunk(u, 2, dim=1)
        
        if self.config.data_dim == 1:
            x = x.flatten()
            v = v.flatten()

        beta = add_dimensions(self.beta_fn(t), self.config.is_image, dim=self.config.data_dim)

        drift_x = self.m_inv * beta * v
        drift_v = -beta * x - self.f * self.m_inv * beta * v

        diffusion_x = torch.zeros_like(x)
        diffusion_v = torch.sqrt(2. * self.f * beta) * torch.ones_like(v)
        
        if self.config.data_dim == 1:
            drift_x = drift_x.reshape((-1,1))
            drift_v = drift_v.reshape((-1,1))
            diffusion_x = diffusion_x.reshape((-1,1))
            diffusion_v = diffusion_v.reshape((-1,1))

        return torch.cat((drift_x, drift_v), dim=1), torch.cat((diffusion_x, diffusion_v), dim=1)

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            '''
            Evaluating drift and diffusion of the ReverseSDE.
            '''
            drift, diffusion = sde_fn(u, self.config.max_time - t)
            # print(u.shape, drift.shape, diffusion.shape)
            u = torch.reshape(u, (t.shape[0], -1))
            # print(u.shape)
            score = score if score is not None else score_fn(u, self.config.max_time - t)
            
            drift_x, drift_v = torch.chunk(drift, 2, dim=1)
            _, diffusion_v = torch.chunk(diffusion, 2, dim=1)

            if self.config.data_dim == 1:
                drift_x = drift_x.flatten()
                drift_v = drift_v.flatten()
                diffusion_v = diffusion_v.flatten()
            
            reverse_drift_x = -drift_x
            reverse_drift_v = -drift_v + diffusion_v ** 2. * \
                score * (0.5 if probability_flow else 1.)

            reverse_diffusion_x = torch.zeros_like(diffusion_v)
            reverse_diffusion_v = torch.zeros_like(
                diffusion_v) if probability_flow else diffusion_v
            
            if self.config.data_dim == 1:
                reverse_drift_x = reverse_drift_x.reshape((-1,1))
                reverse_drift_v = reverse_drift_v.reshape((-1,1))
                reverse_diffusion_x = reverse_diffusion_x.reshape((-1,1))
                reverse_diffusion_v = reverse_diffusion_v.reshape((-1,1))
                
            return torch.cat((reverse_drift_x, reverse_drift_v), dim=1), torch.cat((reverse_diffusion_x, reverse_diffusion_v), dim=1)

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), torch.randn(*shape, device=self.config.device) / np.sqrt(self.m_inv)

    def prior_logp(self, u):
        x, v = torch.chunk(u, 2, dim=1)
        N = np.prod(x.shape[1:])

        logx = -N / 2. * np.log(2. * np.pi) - \
            torch.sum(x.view(x.shape[0], -1) ** 2., dim=1) / 2.
        logv = -N / 2. * np.log(2. * np.pi / self.m_inv) - torch.sum(
            v.view(v.shape[0], -1) ** 2., dim=1) * self.m_inv / 2.
        return logx, logv

    def mean(self, u, t):
        '''
        Evaluating the mean of the conditional perturbation kernel.
        '''
        x, v = torch.chunk(u, 2, dim=1)
        
        if self.config.data_dim == 1:
            x = x.flatten()
            v = v.flatten()

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        coeff_mean = torch.exp(-2. * beta_int * self.g)

        mean_x = coeff_mean * (2. * beta_int * self.g *
                               x + 4. * beta_int * self.g ** 2. * v + x)
        mean_v = coeff_mean * (-beta_int * x - 2. * beta_int * self.g * v + v)
        
        if self.config.data_dim == 1:
            mean_x = mean_x.reshape((-1,1))
            mean_v = mean_v.reshape((-1,1))
            
        return torch.cat((mean_x, mean_v), dim=1)

    def var(self, t, var0x=None, var0v=None):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image, dim=self.config.data_dim)
        if var0v is None:
            if self.config.cld_objective == 'dsm':
                var0v = torch.zeros_like(
                    t, dtype=torch.float64, device=t.device)
            elif self.config.cld_objective == 'hsm':
                var0v = (self.gamma / self.m_inv) * torch.ones_like(t,
                                                                    dtype=torch.float64, device=t.device)

            var0v = add_dimensions(var0v, self.config.is_image, dim=self.config.data_dim)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        multiplier = torch.exp(-4. * beta_int * self.g)

        var_xx = var0x + (1. / multiplier) - 1. + 4. * beta_int * self.g * (var0x - 1.) + 4. * \
            beta_int ** 2. * self.g ** 2. * \
            (var0x - 2.) + 16. * self.g ** 4. * beta_int ** 2. * var0v
        var_xv = -var0x * beta_int + 4. * self.g ** 2. * beta_int * var0v - 2. * self.g * \
            beta_int ** 2. * (var0x - 2.) - 8. * \
            self.g ** 3. * beta_int ** 2. * var0v
        var_vv = self.f ** 2. * ((1. / multiplier) - 1.) / 4. + self.f * beta_int - 4. * self.g * beta_int * \
            var0v + 4. * self.g ** 2. * beta_int ** 2. * \
            var0v + var0v + beta_int ** 2. * (var0x - 2.)
        return [var_xx * multiplier + self.numerical_eps, var_xv * multiplier, var_vv * multiplier + self.numerical_eps]

    def mean_and_var(self, u, t, var0x=None, var0v=None):
        return self.mean(u, t), self.var(t, var0x, var0v)

    def noise_multiplier(self, t, var0x=None, var0v=None):
        '''
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        '''
        var = self.var(t, var0x, var0v)
        coeff = torch.sqrt(var[0] / (var[0] * var[2] - var[1]**2))

        if torch.sum(torch.isnan(coeff)) > 0:
            raise ValueError('Numerical precision error.')

        return -coeff

    def loss_multiplier(self, t):
        '''
        Evaluating the "maximum likelihood" multiplier.
        '''
        return self.beta_fn(t) * self.f

    def perturb_data(self, batch, t, var0x=None, var0v=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and 
        \gamma * M for HSM.
        '''
        mean, var = self.mean_and_var(batch, t, var0x, var0v)

        cholesky11 = (torch.sqrt(var[0]))
        cholesky21 = (var[1] / cholesky11)
        cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

        if torch.sum(torch.isnan(cholesky11)) > 0 or torch.sum(torch.isnan(cholesky21)) > 0 or torch.sum(torch.isnan(cholesky22)) > 0:
            raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(batch, device=batch.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)
        
        if self.config.data_dim == 1:
            batch_randn_x = batch_randn_x.flatten()
            batch_randn_v = batch_randn_v.flatten()

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        
        if self.config.data_dim == 1:
            noise_x = noise_x.reshape((-1,1))
            noise_v = noise_v.reshape((-1,1))
        noise = torch.cat((noise_x, noise_v), dim=1)

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn(*u.shape, device=u.device)

            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn


class VPSDE(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn

    @property
    def type(self):
        return 'vpsde'

    @property
    def is_augmented(self):
        return False

    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image, dim=self.config.data_dim)

        if self.config.data_dim == 1:
            u = u.flatten()
            
        drift = -0.5 * beta * u
        diffusion = torch.sqrt(beta) * torch.ones_like(u,
                                                       device=self.config.device)

        return drift, diffusion

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            drift, diffusion = sde_fn(u, self.config.max_time - t)
            score = score if score is not None else score_fn(u, self.config.max_time - t)

            reverse_drift = -drift + diffusion**2 * \
                score * (0.5 if probability_flow else 1.0)
            reverse_diffusion = torch.zeros_like(
                diffusion) if probability_flow else diffusion

            return reverse_drift, reverse_diffusion

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), None

    def prior_logp(self, u):
        shape = u.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2. * np.pi) - torch.sum(u.view(u.shape[0], -1) ** 2., dim=1) / 2., None

    def var(self, t, var0x=None):
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image, dim=self.config.data_dim)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)

        coeff = torch.exp(-beta_int)
        return [1. - (1. - var0x) * coeff]

    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)

        return x * torch.exp(-0.5 * beta_int)

    def mean_and_var(self, x, t, var0x=None):
        if var0x is None:
            var0x = torch.zeros_like(x, device=self.config.device)

        return self.mean(x, t), self.var(t, var0x)

    def noise_multiplier(self, t, var0x=None):
        _var = self.var(t, var0x)[0]
        return -1. / torch.sqrt(_var)

    def loss_multiplier(self, t):
        return 0.5 * self.beta_fn(t)

    def perturb_data(self, batch, t, var0x=None):
        mean, var = self.mean_and_var(batch, t, var0x)
        cholesky = torch.sqrt(var[0])
        
        if self.config.data_dim == 1:
            cholesky = cholesky.flatten()

        batch_randn = torch.randn_like(batch, device=batch.device)
        noise = cholesky * batch_randn

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)
            
            if self.config.data_dim == 1:
                u = u.flatten()
            
            noise = torch.randn_like(u, device=u.device)
            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn

class PassiveDiffusion(VPSDE):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__(config, beta_fn, beta_int_fn)
        
        self.Tp = config.Tp
        self.k = config.k
        
    @property
    def type(self):
        return 'passive'
    
    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        if self.config.data_dim == 1:
            u = u.flatten()
            
        drift = - self.k * beta * u
        
        diffusion = torch.sqrt(2 * beta * self.Tp)

        return drift, diffusion

    def prior_sampling(self, shape):
        return np.sqrt(self.Tp / self.k ) * torch.randn(*shape, device=self.config.device), None
    
    def var(self, t, var0x=None): 
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)

        a = torch.exp(-self.k * beta_int)
        
        var = (1/self.k)*self.Tp*(1-a**2)
        return [var]
    
    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)

        a = torch.exp(-self.k * beta_int)
        
        mean_x = a * x
        return mean_x
    
    def loss_multiplier(self, t):
        beta = self.beta_fn(t)
        
        return self.Tp * beta
    
class ActiveDiffusion(CLD):
    def __init__(self, config, beta_fn, beta_int_fn):
        config.m_inv = 1
        
        super().__init__(config, beta_fn, beta_int_fn)
        
        self.Tp = config.Tp
        self.Ta = config.Ta
        self.tau = config.tau
        self.k = config.k
    
    @property
    def type(self):
        return 'active'
    
    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k
        tau = self.tau
        Tp = self.Tp
        Ta = self.Ta

        x, eta = torch.chunk(u, 2, dim=1)
        
        if self.config.data_dim == 2:
            x = torch.reshape(x, (-1,2))
            eta = torch.reshape(eta, (-1,2))
        elif self.config.data_dim == 1:
            x = x.flatten()
            eta = eta.flatten()
        
        drift_x = - k * beta * x + beta*eta
        drift_eta =  - beta *eta / tau
        
        diffusion_x = torch.sqrt(2 * beta * Tp) * torch.ones_like(x)
        
        diffusion_eta = 1 / tau * torch.sqrt(2 * beta * Ta) * torch.ones_like(eta)
        
        if self.config.data_dim == 1:
            drift_x = drift_x.reshape((-1,1))
            drift_eta = drift_eta.reshape((-1,1))
            
            diffusion_x = diffusion_x.reshape((-1,1))
            diffusion_eta = diffusion_eta.reshape((-1,1))
                                              
        return torch.cat((drift_x, drift_eta), dim=1), \
               torch.cat((diffusion_x, diffusion_eta), dim=1)
    
    def prior_sampling(self, shape):
        var_11 = 1/self.k * (self.Tp + self.Ta/(1+ self.k*self.tau))
        var_12 = self.Ta/(1 + self.tau*self.k)
        var_22 = self.Ta / self.tau
        
        zero_mean = torch.zeros(2, device=self.config.device)
        
        covar = torch.tensor([var_11, var_12, var_12, var_22], device=self.config.device)
        covar = torch.reshape(covar, (2,2))
        
        sampler = MultivariateNormal(loc=zero_mean, covariance_matrix=covar)
        
        sample_1 = sampler.sample(sample_shape=torch.Size([shape[0]]))
        sample_2 = sampler.sample(sample_shape=torch.Size([shape[0]]))
        
        sample_1x, sample_1eta = torch.chunk(sample_1, 2, dim=1)
        sample_2x, sample_2eta = torch.chunk(sample_2, 2, dim=1)
        
        if self.config.data_dim == 2:
            sample_x = torch.cat((sample_1x, sample_2x), dim=1)
            sample_eta = torch.cat((sample_1eta, sample_2eta), dim=1)
        if self.config.data_dim == 1:
            sample_x = sample_1x
            sample_eta = sample_1eta
        
        return sample_x, sample_eta
    
    def var(self, t, var0x=None, var0v=None):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k      
        w = 1 / self.tau
        tau = self.tau
        
        a = torch.exp(-k*beta_int)
        b = torch.exp(-w*beta_int)
        c = k + w
        d = k - w

        Tp = self.Tp
        Ta = self.Ta

        M11 = (Tp/k)*(1-a**2) + \
                (Ta / tau**2) * (tau/(k*c) + 1/d**2*(4*a*b/c - b**2*tau - a**2/k))
  
        M12 = Ta /(tau * c* d) * (k *(1-b**2) - 1/tau * (1 + b**2 - 2*a*b))
   
        M22 = (Ta/tau)*(1-b**2)

        return [ M11 + self.numerical_eps, M12 + self.numerical_eps, M22 + self.numerical_eps]
    
    def mean(self, batch, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k
        w = 1/self.tau

        a = torch.exp(-k * beta_int)
        b = (torch.exp(-w * beta_int)-torch.exp(-k * beta_int))/(k-w)
        c = torch.exp(-w * beta_int)
        
        batch_x, batch_eta = torch.chunk(batch, 2, dim=1)
        
        if self.config.data_dim == 1:
            batch_x = batch_x.flatten()
            batch_eta = batch_eta.flatten()
        
        mean_x = a * batch_x + b * batch_eta
        mean_eta = c * batch_eta
        
        if self.config.data_dim == 1:
            mean_x = mean_x.reshape((-1,1))
            mean_eta = mean_eta.reshape((-1,1))
                
        return torch.cat((mean_x, mean_eta), dim=1)
    
    def loss_multiplier(self, t):
        beta = self.beta_fn(t)
        return self.Ta * beta / self.tau**2
    
    
class ChiralActiveDiffusion(CLD):
    def __init__(self, config, beta_fn, beta_int_fn):
        config.m_inv = 1
        
        super().__init__(config, beta_fn, beta_int_fn)
        
        self.Tp = config.Tp
        self.Ta = config.Ta
        self.tau = config.tau
        self.k = config.k
        self.omega = config.omega
    
    @property
    def type(self):
        return 'chiral_active'
    
    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k
        tau = self.tau
        Tp = self.Tp
        Ta = self.Ta
        omega = self.omega
        

        x, eta = torch.chunk(u, 2, dim=1)
        
        if self.config.data_dim == 2:
            x = torch.reshape(x, (-1,2))
            eta = torch.reshape(eta, (-1,2))
        elif self.config.data_dim == 1:
            x = x.flatten()
            eta = eta.flatten()
            
        M = torch.tensor([[tau, omega],
                          [-omega, 1/tau]],
                         device=eta.device)
        
        drift_x = - k * beta * x + beta*eta
        drift_eta =  - beta  * torch.matmul(M, eta.T[:,None].reshape((2,-1))).T
        
        diffusion_x = torch.sqrt(2 * beta * Tp) * torch.ones_like(x)
        
        diffusion_eta = 1 / tau * torch.sqrt(2 * beta * Ta) * torch.ones_like(eta)
        
        if self.config.data_dim == 1:
            drift_x = drift_x.reshape((-1,1))
            drift_eta = drift_eta.reshape((-1,1))
            
            diffusion_x = diffusion_x.reshape((-1,1))
            diffusion_eta = diffusion_eta.reshape((-1,1))
                   
        return torch.cat((drift_x, drift_eta), dim=1), \
               torch.cat((diffusion_x, diffusion_eta), dim=1)
    
    def prior_sampling(self, shape):
        
        k = self.k
        tau = self.tau
        omega = self.omega
        Tp = self.Tp
        Ta = self.Ta
        
        K = k * tau
        Omega = omega * tau
        kappa_plus = 1 + K
        kappa_minus = 1 - K
        
        m11 = 1/k * (Tp + Ta * kappa_plus / (kappa_plus**2 + Omega**2) )
        m12 = Ta * kappa_plus / (kappa_plus**2 + Omega**2)
        m22 = Ta / tau
        
        u12 = Ta * Omega / (kappa_plus**2 + Omega**2)

        covar = torch.tensor([[m11, 0, m12, u12], \
                              [0, m11, -u12, m12], \
                              [m12, -u12, m22, 0], \
                              [u12, m12, 0, m22]])

        zero_mean = torch.zeros(4)

        sampler = MultivariateNormal(loc=zero_mean, covariance_matrix=covar)
        
        sample = sampler.sample(sample_shape=torch.Size([shape[0]]))
        
        sample_x, sample_eta = torch.chunk(sample, 2, dim=1)
        
        return sample_x.to(device=self.config.device), sample_eta.to(device=self.config.device)
    
    def var(self, t, var0x=None, var0v=None):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k      
        tau = self.tau
        omega = self.omega
        Tp = self.Tp
        Ta = self.Ta
        
        a = torch.exp(-k*beta_int)
        b = torch.exp(-beta_int/tau)
        K = k * tau
        Omega = omega*tau
        kappa_plus = 1 + K
        kappa_minus = 1 - K


        M11 = (Tp/k)*(1-a**2) + \
                ( (Ta / k)/((kappa_plus**2 + Omega**2)*(kappa_minus**2 + Omega**2)) ) * \
                        ( kappa_plus*(kappa_minus**2 + Omega**2) - \
                          (a**2 + b**2 * K)*(kappa_plus**2 + Omega**2) + \
                          4*a*b*K*( kappa_plus*torch.cos(omega*beta_int) - \
                                    Omega*torch.sin(omega*beta_int)
                                  )
                        )
  
        M12 = Ta / ((kappa_plus**2 + Omega**2)*(kappa_minus**2 + Omega**2)) * \
                ( kappa_plus*(kappa_minus**2 + Omega**2) + \
                  b**2*kappa_minus*(kappa_plus**2 + Omega**2) - \
                  2*a*b*( ( kappa_plus*kappa_minus + Omega**2)*torch.cos(omega*beta_int) - \
                            2*K*Omega*torch.sin(omega*beta_int)
                        )
                )
   
        M22 = (Ta/tau)*(1-b**2)
        
        u12 = Ta / ((kappa_plus**2 + Omega**2)*(kappa_minus**2 + Omega**2)) * \
            (Omega*(kappa_minus**2 + Omega**2) - b**2*Omega*(kappa_plus**2 + Omega**2) + \
                2*a*b*(2*K*Omega*torch.cos(omega*beta_int) - (kappa_plus*kappa_minus+Omega**2)*torch.sin(omega*beta_int) ))

        return [ M11 + self.numerical_eps, M12 + self.numerical_eps, M22 + self.numerical_eps, u12 + self.numerical_eps]
    
    def mean(self, batch, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image, dim=self.config.data_dim)
        
        k = self.k
        tau = self.tau
        omega = self.omega
        
        A = ( torch.exp(-k*beta_int)*(1-k*tau) - 
              torch.exp(-beta_int/tau) * \
                  ( (1-k*tau)*torch.cos(omega*beta_int) - \
                    omega*tau*torch.sin(omega*beta_int)) \
            ) /\
            ((1-k*tau)**2 + (omega*tau)**2)
                    
        B = ( torch.exp(-k*beta_int)*omega*tau - \
              torch.exp(-beta_int/tau) * \
                  (omega*tau*torch.cos(omega*beta_int) + \
                   (1-k*tau)*torch.sin(omega*beta_int)) \
            ) /\
            ((1-k*tau)**2 + (omega*tau)**2)
        
        batch_x, batch_eta = torch.chunk(batch, 2, dim=1)
        
        batch_x0, batch_x1 = torch.chunk(batch_x, 2, dim=1)
        batch_e0, batch_e1 = torch.chunk(batch_eta, 2, dim=1)
        
        mean_x0 = batch_x0*torch.exp(-k*beta_int) + tau * A * batch_e0 - tau * B * batch_e1
        mean_x1 = batch_x1*torch.exp(-k*beta_int) + tau * B * batch_e0 - tau * A * batch_e1
        
        mean_e0 = torch.exp(-beta_int/tau)*(torch.cos(omega*beta_int)*batch_e0 - torch.sin(omega*beta_int)*batch_e1)
        mean_e1 = torch.exp(-beta_int/tau)*(torch.sin(omega*beta_int)*batch_e0 + torch.cos(omega*beta_int)*batch_e1)
        
        mean_x = torch.cat((mean_x0, mean_x1), dim=1)
        mean_eta = torch.cat((mean_e0, mean_e1), dim=1)
                
        return torch.cat((mean_x, mean_eta), dim=1)
    
    def loss_multiplier(self, t):
        beta = self.beta_fn(t)
        return self.Ta * beta / self.tau**2
    
    def perturb_data(self, batch, t, var0x=None, var0v=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and 
        \gamma * M for HSM.
        '''
        mean, var = self.mean_and_var(batch, t, var0x, var0v)

        m11 = var[0]
        m12 = var[1]
        m22 = var[2]
        u12 = var[3]

        A11 = m11
        A21 = 0
        A22 = m11
        A31 = m12
        A32 = -u12
        A33 = m22
        A41 = u12
        A42 = m12
        A43 = 0
        A44 = m22

        L11 = torch.sqrt(A11)    
        L21 = A21/L11
        L22 = torch.sqrt(A22 - L21**2)
        L31 = A31/L11
        L32 = (A32 - (L31*L21))/L22
        L33 = torch.sqrt(A33 - (L31**2 + L32**2))
        L41 = A41/L11
        L42 = (A42 - L41*L21)/L22
        L43 = (A43 - (L41*L31 + L42*L32))/L33
        L44 = torch.sqrt(A44 - (L41**2 + L42**2 + L43**2))
        
        cholesky_list = [L11, L21, L22, L31, L32, L33, L41, L42, L43, L44]

        for cholesky in cholesky_list:
            if torch.sum(torch.isnan(cholesky)) > 0:
                raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(batch, device=batch.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)
        batch_randn_x0, batch_randn_x1 = torch.chunk(batch_randn_x, 2, dim=1)
        batch_randn_v0, batch_randn_v1 = torch.chunk(batch_randn_v, 2, dim=1)

        noise_x0 = L11*batch_randn_x0
        noise_x1 = L21*batch_randn_x0 + L22*batch_randn_x1
        noise_v0 = L31*batch_randn_x0 + L32*batch_randn_x1 + L33*batch_randn_v0
        noise_v1 = L41*batch_randn_x0 + L42*batch_randn_x1 + L43*batch_randn_v0 + L44*batch_randn_v1

        noise_x = torch.cat((noise_x0, noise_x1), dim=1)
        noise_v = torch.cat((noise_v0, noise_v1), dim=1)
        
        noise = torch.cat((noise_x, noise_v), dim=1)

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn
    

if __name__=="__main__":
    class Test:
        def __init__(self):
            self.Tp = 0
            self.Ta = 1
            self.tau = 0.25
            self.k = 1
            self.omega = 0.5
        
        def prior_sampling(self):
            
            k = self.k
            tau = self.tau
            omega = self.omega
            Tp = self.Tp
            Ta = self.Ta
            
            K = k*tau
            Omega = omega*tau
            kappa_plus = 1 + K
            kappa_minus = 1 - K
            
            m11 = 1/k * (Tp + Ta * kappa_plus / (kappa_plus**2 + Omega**2) )
            m12 = Ta * kappa_plus / (kappa_plus**2 + Omega**2)
            m22 = Ta / tau
            
            u12 = Ta * Omega / (kappa_plus**2 + Omega**2)
            
            I_mat = torch.eye(2)
            e_mat = torch.tensor([[0,1],[-1,0]])
            
            C11 = (m11*I_mat).tolist()
            C12 = (m12*I_mat + u12*e_mat).tolist()
            C21 = (m12*I_mat - u12*e_mat).tolist()
            C22 = (m22*I_mat).tolist()
          
            covar = torch.tensor([ C11[0], C12[0], 
                                   C11[1], C12[1], 
                                   C21[0], C22[0], 
                                   C21[1], C22[1] ])                         

            covar = covar.reshape((4,4))
        
            zero_mean = torch.zeros(4)

            sampler = MultivariateNormal(loc=zero_mean, covariance_matrix=covar)
            
            sample = sampler.sample()
            
            sample_x, sample_eta = torch.chunk(sample, 2)
            
            return sample_x, sample_eta
        
    myTest = Test()
    myTest.prior_sampling()