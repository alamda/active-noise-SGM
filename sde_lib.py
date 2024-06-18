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

        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift_x = self.m_inv * beta * v
        drift_v = -beta * x - self.f * self.m_inv * beta * v

        diffusion_x = torch.zeros_like(x)
        diffusion_v = torch.sqrt(2. * self.f * beta) * torch.ones_like(v)

        return torch.cat((drift_x, drift_v), dim=1), torch.cat((diffusion_x, diffusion_v), dim=1)

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            '''
            Evaluating drift and diffusion of the ReverseSDE.
            '''
            drift, diffusion = sde_fn(u, self.config.max_time - t)
            score = score if score is not None else score_fn(u, self.config.max_time - t)

            drift_x, drift_v = torch.chunk(drift, 2, dim=1)
            _, diffusion_v = torch.chunk(diffusion, 2, dim=1)

            reverse_drift_x = -drift_x
            reverse_drift_v = -drift_v + diffusion_v ** 2. * \
                score * (0.5 if probability_flow else 1.)

            reverse_diffusion_x = torch.zeros_like(diffusion_v)
            reverse_diffusion_v = torch.zeros_like(
                diffusion_v) if probability_flow else diffusion_v

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

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        coeff_mean = torch.exp(-2. * beta_int * self.g)

        mean_x = coeff_mean * (2. * beta_int * self.g *
                               x + 4. * beta_int * self.g ** 2. * v + x)
        mean_v = coeff_mean * (-beta_int * x - 2. * beta_int * self.g * v + v)
        return torch.cat((mean_x, mean_v), dim=1)

    def var(self, t, var0x=None, var0v=None):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image)
        if var0v is None:
            if self.config.cld_objective == 'dsm':
                var0v = torch.zeros_like(
                    t, dtype=torch.float64, device=t.device)
            elif self.config.cld_objective == 'hsm':
                var0v = (self.gamma / self.m_inv) * torch.ones_like(t,
                                                                    dtype=torch.float64, device=t.device)

            var0v = add_dimensions(var0v, self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
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

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
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
        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

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
                t, dtype=torch.float64, device=t.device), self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        coeff = torch.exp(-beta_int)
        return [1. - (1. - var0x) * coeff]

    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

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
        beta = add_dimensions(self.beta_fn(t), self.config.is_image)
        
        drift = - self.k * beta * u
        
        diffusion = torch.sqrt(2 * beta * self.Tp)

        return drift, diffusion

    def prior_sampling(self, shape):
        return np.sqrt(self.Tp / self.k ) * torch.randn(*shape, device=self.config.device), None
    
    def var(self, t, var0x=None): 
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        a = torch.exp(-self.k * beta_int)
        
        var = (1/self.k)*self.Tp*(1-a**2)
        return [var]
    
    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        
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
        beta = add_dimensions(self.beta_fn(t), self.config.is_image)
        
        k = self.k
        tau = self.tau
        Tp = self.Tp
        Ta = self.Ta

        x, eta = torch.chunk(u, 2, dim=1)
        
        x = torch.reshape(x, (-1,2))
        eta = torch.reshape(eta, (-1,2))
                                
        drift_x = - k * beta * x + beta*eta
        drift_eta =  - beta *eta / tau
        
        diffusion_x = torch.sqrt(2 * beta * Tp) * torch.ones_like(x)
        
        diffusion_eta = 1 / tau * torch.sqrt(2 * beta * Ta) * torch.ones_like(eta)
                                      
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
        
        sample_x = torch.cat((sample_1x, sample_2x), dim=1)
        sample_eta = torch.cat((sample_1eta, sample_2eta), dim=1)
        
        return sample_x, sample_eta
    
    def var(self, t, var0x=None, var0v=None):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        
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
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        
        k = self.k
        w = 1/self.tau

        a = torch.exp(-k * beta_int)
        b = (torch.exp(-w * beta_int)-torch.exp(-k * beta_int))/(k-w)
        c = torch.exp(-w * beta_int)
        
        batch_x, batch_eta = torch.chunk(batch, 2, dim=1)
        
        mean_x = a * batch_x + b * batch_eta
        mean_eta = c * batch_eta
                
        return torch.cat((mean_x, mean_eta), dim=1)
    
    def loss_multiplier(self, t):
        beta = self.beta_fn(t)
        return self.Ta * beta / self.tau**2