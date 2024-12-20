# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from scipy import linalg
from torch.optim import Adamax, AdamW
try:
    from apex.optimizers import FusedAdam as Adam
except ImportError:
    logging.info('Apex is not available. Falling back to PyTorch\'s native Adam. Install Apex for faster training.')
    from torch.optim import Adam as Adam

import random


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')


def optimization_manager(config):

    def optimize_fn(optimizer, 
                    params, 
                    step, 
                    scaler=None,
                    lr=config.learning_rate,
                    grad_clip=config.grad_clip):

        if config.n_warmup_iters > 0 and step <= config.n_warmup_iters:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / config.n_warmup_iters, 1.0)

        if scaler is None:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
        else:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

    return optimize_fn


def get_optimizer(config, params):
    if config.optimizer == 'Adam':
        optimizer = Adam(params, 
                         lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
    elif config.optimizer == 'Adamax':
        optimizer = Adamax(params, 
                           lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(params, 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    else:
        raise NotImplementedError('Optimizer %s is not supported.' % config.optimizer)

    return optimizer


def get_data_scaler(config):
    if config.center_image and config.is_image:
        return lambda x: x * 2. - 1.  # Rescale from [0, 1] to [-1, 1]
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    if config.center_image and config.is_image:
        return lambda x: (x + 1.) / 2.  # Rescale from [-1, 1] to [0, 1]
    else:
        return lambda x: x

def get_data_scaler_ising(config):
    if (config.dataset == 'ising_2D') and config.is_image:
        # Can use torch.sign() instead of nested torch.where(),
        # but if value is exactly equal to 0, torch.sign() returns 0.abs
        # For a float64, the chance of a value being exactly zero are low and 
        # the chances of it affecting diffusion results are low.
        # The current method sets values that are equal to 0 to either -1 or +1
        # (with equal probability)
        return lambda x: torch.where(torch.where(x == 0., random.choice([-1., 1.]), x) < 0, -1., 1.) #torch.sign(x) 
    
def get_data_inverse_scaler_ising(config):
    if (config.dataset == 'ising_2D') and config.is_image:
        return lambda x: torch.where(torch.where(x == 0., random.choice([-1., 1.]), x) < torch.mean(x), -1., 1.) #torch.sign(x)

def get_data_scaler_alanine_dipeptide_25(config):
    if (config.dataset == 'alanine_dipeptide_25'):
        def scale_fn(data):
            
            train_data_mins = train_data.min(axis=0)
            train_data_maxs = train_data.max(axis=0)

            for idx in range(data.shape[1]):
                d = data[:,idx]
                min_val = train_data_mins[idx]
                max_val = train_data_maxs[idx]
                data[:,idx] = 2 * (d - min_val) / (max_val - min_val) - 1

            return data
        return scale_fn
    
def get_data_inverse_scaler_alanine_dipeptide_25(config):
    if (config.dataset == 'alanine_dipeptide_25'):
        def inverse_scale_fn(data):
            train_data = np.load('alanine_dipeptide_25.npy')
            
            train_data_mins = train_data.min(axis=0)
            train_data_maxs = train_data.max(axis=0)
                        
            for idx in range(data.shape[1]):
                d = data[:,idx]
                min_val = train_data_mins[idx]
                max_val = train_data_maxs[idx]
                data[:,idx] = 0.5 * (d + 1) * (max_val - min_val) + min_val
        
            return data
        return inverse_scale_fn

def get_data_scaler_ala_25(config):
    if (config.dataset == 'ala_25') and config.is_image:
        def scale_fn(data):

            train_data = np.load('alanine_dipeptide_25.npy')
            
            train_data_mins = train_data.min(axis=0).reshape(5,5)
            train_data_maxs = train_data.max(axis=0).reshape(5,5)     

            for xidx in range(5):
               for yidx in range(5):
                    min_val = train_data_mins[xidx,yidx]
                    max_val = train_data_maxs[xidx,yidx]
                   
                    d = data[:,:,xidx,yidx]
                    data[:,:,xidx,yidx] = 2 * (d - min_val) / (max_val - min_val) - 1
            
            return data
        return scale_fn
    
def get_data_inverse_scaler_ala_25(config):
    if (config.dataset == 'ala_25') and config.is_image:
        def inverse_scale_fn(data):
            train_data = np.load('alanine_dipeptide_25.npy')
            
            train_data_mins = train_data.min(axis=0).reshape(5,5)
            train_data_maxs = train_data.max(axis=0).reshape(5,5)     

            for xidx in range(5):
               for yidx in range(5):
                    min_val = train_data_mins[xidx,yidx]
                    max_val = train_data_maxs[xidx,yidx]
                   
                    d = data[:,:,xidx,yidx]
                    data[:,:,xidx,yidx] = 0.5 * (d + 1) * (max_val - min_val) + min_val
            return data
        return inverse_scale_fn

def get_data_scaler_ala_28(config):
    if (config.dataset == 'ala_28') and config.is_image:
        def scale_fn(data):
            train_data = np.load('alanine_dipeptide_28.npy')
            
            train_data_mins = train_data.min(axis=0)
            train_data_maxs = train_data.max(axis=0)     

            for idx in range(28):
                min_val = train_data_mins[idx]
                max_val = train_data_maxs[idx]
                
                d = data[:,idx,0,0]
                data[:,idx,0,0] = 2 * (d - min_val) / (max_val - min_val) - 1
            
            return data
        return scale_fn
    
def get_data_inverse_scaler_ala_28(config):
    if (config.dataset == 'ala_28') and config.is_image:
        def inverse_scale_fn(data):
            train_data = np.load('alanine_dipeptide_28.npy')

            train_data_mins = train_data.min(axis=0)
            train_data_maxs = train_data.max(axis=0)     

            for idx in range(28):
                min_val = train_data_mins[idx]
                max_val = train_data_maxs[idx]
                
                d = data[:,idx,0,0]
                data[:,idx,0,0] = 0.5 * (d + 1) * (max_val - min_val) + min_val
            return data
        return inverse_scale_fn

def compute_bpd_from_nll(nll, D, inverse_scaler):
    offset = 7 - inverse_scaler(-1)
    bpd = nll / (np.log(2.) * D) + offset
    return bpd


def batched_cov(x):
    covars = np.empty((x.shape[0], x.shape[2], x.shape[2]))
    for i in range(x.shape[0]):
        covars[i] = np.cov(x[i], rowvar=False)
    return covars


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def concatenate(tensor, world_size):
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)


def split_tensor(tensor, global_rank, global_size):
    if tensor.shape[0] / global_size - tensor.shape[0] // global_size > 1e-6:
        raise ValueError('Tensor is not divisible by global size.')
    return torch.chunk(tensor, global_size)[global_rank]


def set_seeds(rank, seed):
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def save_img(x, filename, figsize=None, title=None):
    figsize = figsize if figsize is not None else (6, 6)

    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid(x, nrow)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).cpu())
    if title is not None:
        plt.title(title, fontsize=20)
    plt.savefig(filename)
    plt.close()


def debug_save_img(x, filename, figsize=None, title=None):
    figsize = figsize if figsize is not None else (4,6)
    
    fig, ax = plt.subplots(2, layout='constrained')
    fig.set_size_inches(figsize)
    if title is not None:
        ax[0].set_title(title, fontsize=12)
    pic = ax[0].imshow(x.cpu().reshape(x.shape[-2], x.shape[-1]), norm=mpl.colors.CenteredNorm(), cmap='seismic')
    fig.colorbar(pic, ax=ax)
    ax[1].hist(x.cpu().flatten(), density=True)
    xabs_max = abs(max(ax[1].get_xlim(), key=abs))
    ax[1].set_xlim(xmin=-xabs_max, xmax=xabs_max)
    
    plt.savefig(filename)
    plt.close(fig)

def compute_eval_loss(config, state, eval_step_fn, valid_queue, scaler=None, test=False, step=None):
    if not test:
        n_batches = config.n_eval_batches
    else:
        n_batches = len(valid_queue)

    if n_batches < 1:
        raise ValueError('Need to evaluate at least one batch.')

    n_items = 0
    total_eval_loss = 0.0
    for i, (eval_x, _) in enumerate(valid_queue):
        if i == n_batches:
            return total_eval_loss / n_items
        else:
            if scaler is not None:
                eval_x = scaler(eval_x)
            eval_x = eval_x.cuda()

            eval_loss = eval_step_fn(state, eval_x, None, step=step)
            eval_loss = reduce_tensor(eval_loss, config.global_size)
            total_eval_loss += eval_loss * eval_x.shape[0] * config.global_size
            n_items += eval_x.shape[0] * config.global_size

    if config.global_rank == 0 and not test:
        logging.info('Does not make sense to evaluate the evaluation set more than once.')
    dist.barrier()

    return total_eval_loss / n_items


def compute_image_likelihood(config, sde, state, likelihood_fn, scaler, inverse_scaler, valid_queue, step=None, likelihood_dir=None, test=False):
    if not test:
        n_batches = config.n_likelihood_batches
    else:
        n_batches = len(valid_queue)

    if n_batches < 1:
        raise ValueError('Need to evaluate at least one batch.')

    bpds = []
    for i, (eval_x, _) in enumerate(valid_queue):
        if i == n_batches:
            return np.mean(np.asarray(bpds))
        else:
            eval_x = (torch.rand_like(eval_x) + eval_x * 255.) / 256.  # Dequantization
            eval_x = scaler(eval_x)
            eval_x = eval_x.cuda()
            if sde.is_augmented:
                eval_z = torch.randn_like(eval_x, device=eval_x.device) * np.sqrt(sde.gamma / sde.m_inv)
                eval_batch = torch.cat((eval_x, eval_z), dim=1)
            else:
                eval_batch = eval_x

            nll, _, nfe = likelihood_fn(state['model'], eval_batch)
            if sde.is_augmented:
                shape = eval_z.shape
                nll -= 0.5 * np.prod(shape[1:]) * (1 + np.log(2 * np.pi) + np.log(sde.gamma / sde.m_inv))

            bpd = compute_bpd_from_nll(nll, np.prod(eval_x.shape[1:]), inverse_scaler) 
            bpd = reduce_tensor(bpd, config.global_size)
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            bpds.extend(bpd)

            nfe = int(reduce_tensor(torch.tensor(float(nfe), device=config.device), config.global_size).detach().cpu())

            if config.global_rank == 0:
                logging.info('Batch: %d, nfe: %d, mean bpd: %6f' % (i, nfe, np.mean(np.asarray(bpds))))
            dist.barrier()

    if config.global_rank == 0 and not test:
        logging.info('Does not make sense to evaluate the evaluation set more than once.')
    dist.barrier()

    if step is not None:
        np.save(os.path.join(likelihood_dir, 'step_%d' % step), np.asarray(bpds))
    return np.mean(np.asarray(bpds))


def compute_non_image_likelihood(config, sde, state, likelihood_fn, inf_data_gen, step=None, likelihood_dir=None):
    n_batches = config.n_likelihood_batches

    if n_batches < 1:
        raise ValueError('Need to evaluate at least one batch.')

    nlls = []
    for i in range(n_batches):
        eval_x = inf_data_gen(config.dataset, config.testing_batch_size).to(config.device)
        
        if sde.is_augmented:
            eval_z = torch.randn_like(eval_x, device=eval_x.device) * np.sqrt(sde.gamma / sde.m_inv)
            eval_batch = torch.cat((eval_x, eval_z), dim=1)
        else:
            eval_batch = eval_x

        nll, _, nfe = likelihood_fn(state['model'], eval_batch)
        if sde.is_augmented:
            shape = eval_z.shape
            nll -= 0.5 * np.prod(shape[1:]) * (1 + np.log(2 * np.pi) + np.log(sde.gamma / sde.m_inv))

        nll = reduce_tensor(nll, config.global_size)
        nll = nll.detach().cpu().numpy().reshape(-1)
        nlls.extend(nll)

        nfe = int(reduce_tensor(torch.tensor(float(nfe), device=config.device), config.global_size).detach().cpu())

        if config.global_rank == 0:
            logging.info('Batch: %d, nfe: %d, mean nll: %6f' % (i, nfe, np.mean(np.asarray(nlls))))
        dist.barrier()

    if step is not None:
        np.save(os.path.join(likelihood_dir, 'step_%d' % step), np.asarray(nlls))
    return np.mean(np.asarray(nlls))


def build_beta_fn(config):
    if config.beta_type == 'linear':
        def beta_fn(t):
            return config.beta0 + config.beta1 * t
    else:
        raise NotImplementedError('Beta function %s not implemented.' % config.beta_type)

    return beta_fn


def build_beta_int_fn(config):
    if config.beta_type == 'linear':
        def beta_int_fn(t):
            return config.beta0 * t + 0.5 * config.beta1 * t**2
    else:
        raise NotImplementedError('Beta function %s not implemented.' % config.beta_type)

    return beta_int_fn


def add_dimensions(x, is_image, dim=None):
    if is_image:
        return x[:, None, None, None]
    elif dim == 1:
        return x[:, None].flatten()
    else:
        return x[:, None]


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
