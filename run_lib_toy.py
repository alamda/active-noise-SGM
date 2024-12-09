# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import logging
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import cm

from models import mlp
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from util.utils import make_dir, get_optimizer, optimization_manager, set_seeds, compute_eval_loss, compute_non_image_likelihood, broadcast_params, reduce_tensor, build_beta_fn, build_beta_int_fn, get_data_inverse_scaler, get_data_scaler, get_data_scaler_alanine_dipeptide_25, get_data_inverse_scaler_alanine_dipeptide_25
from util.checkpoint import save_checkpoint, restore_checkpoint
import losses
import sde_lib
import sampling
import likelihood
from util.toy_data import inf_data_gen
from models.utils import get_score_fn
from torchdiffeq import odeint


def train(config, workdir):
    ''' Main training script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    global_size = config.global_size

    if config.mode == 'train':
        set_seeds(global_rank, config.seed)
    elif config.mode == 'continue':
        set_seeds(global_rank, config.seed + config.cont_nbr)
    else:
        raise NotImplementedError('Mode %s is unknown.' % config.mode)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    # Setting up all necessary folders
    sample_dir = os.path.join(workdir, 'samples')
    tb_dir = os.path.join(workdir, 'tensorboard')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    likelihood_dir = os.path.join(workdir, 'likelihood')

    if global_rank == 0:
        logging.info(config)
        if config.mode == 'train':
            make_dir(sample_dir)
            make_dir(tb_dir)
            make_dir(checkpoint_dir)
            make_dir(likelihood_dir)
        writer = tensorboard.SummaryWriter(tb_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    elif config.sde == 'passive':
        sde = sde_lib.PassiveDiffusion(config, beta_fn, beta_int_fn)
    elif config.sde == 'active':
        sde = sde_lib.ActiveDiffusion(config, beta_fn, beta_int_fn)
    elif config.sde == 'chiral_active':
        sde = sde_lib.ChiralActiveDiffusion(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.sde)

    # Creating the score model
    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())  # Sync all parameters
    score_model = DDP(score_model, device_ids=[local_rank])

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    if global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, score_model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    dist.barrier()
    
    # Utility functions to map images from [0, 1] to [-1, 1] and back.
    # Used for Ising model only
    if config.dataset == 'alanine_dipeptide_25':
        scaler = get_data_scaler_alanine_dipeptide_25(config)
        inverse_scaler = get_data_inverse_scaler_alanine_dipeptide_25(config)
    else:
        scaler = get_data_scaler(config)
        inverse_scaler = get_data_inverse_scaler(config)

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    if config.mode == 'continue':
        if config.checkpoint is None:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        else:
            ckpt_path = os.path.join(checkpoint_dir, config.checkpoint)

        if global_rank == 0:
            logging.info('Loading model from path: %s' % ckpt_path)
        dist.barrier()

        state = restore_checkpoint(ckpt_path, state, device=config.device)

    num_total_iter = config.n_train_iters

    if global_rank == 0:
        logging.info('Number of total iterations: %d' % num_total_iter)
    dist.barrier()

    optimize_fn = optimization_manager(config)
    train_step_fn = losses.get_step_fn(True, optimize_fn, sde, config)

    if config.is_image: # Ising model
        sampling_shape = (config.sampling_batch_size,
                      config.image_channels,
                      config.image_size,
                      config.image_size)
    else:
        sampling_shape = (config.sampling_batch_size,
                        config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Starting training at step %d' % step)
    dist.barrier()

    if config.mode == 'continue':
        config.eval_threshold = max(step + 1, config.eval_threshold)
        config.snapshot_threshold = max(step + 1, config.snapshot_threshold)
        config.likelihood_threshold = max(
            step + 1, config.likelihood_threshold)
        config.save_threshold = max(step + 1, config.save_threshold)

    while step <= num_total_iter:
        if step % config.likelihood_freq == 0 and step >= config.likelihood_threshold:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            mean_nll = compute_non_image_likelihood(
                config, sde, state, likelihood_fn, inf_data_gen, step=step, likelihood_dir=likelihood_dir)
            ema.restore(score_model.parameters())

            if global_rank == 0:
                logging.info('Mean Nll at step: %d: %.5f' %
                             (step, mean_nll.item()))
                writer.add_scalar('mean_nll', mean_nll.item(), step)

                checkpoint_file = os.path.join(
                    checkpoint_dir, 'checkpoint_%d.pth' % step)
                if not os.path.isfile(checkpoint_file):
                    save_checkpoint(checkpoint_file, state)
            dist.barrier()

        if (step % config.snapshot_freq == 0 or step == num_total_iter) and global_rank == 0 and step >= config.snapshot_threshold:
            logging.info('Saving snapshot checkpoint.')
            save_checkpoint(os.path.join(
                checkpoint_dir, 'snapshot_checkpoint.pth'), state)

            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            x, v, nfe = sampling_fn(score_model)
            ema.restore(score_model.parameters())

            logging.info('NFE snapshot at step %d: %d' % (step, nfe))
            writer.add_scalar('nfe', nfe, step)

            this_sample_dir = os.path.join(sample_dir, 'iter_%d' % step)
            make_dir(this_sample_dir)
            
            fig, ax = plt.subplots()
            
            ax.set_aspect('equal')
            
            lim_dict = {'multigaussian_2D':
                    {'xlim': (-3, 3),
                     'ylim': (-1.5, 1.5)},
                'multigaussian_2D_close':
                    {'xlim': (-3, 3),
                     'ylim': (-1.5, 1.5)},
                'diamond':                  
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)},
                'diamond_close':            
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)},
                'swissroll':                
                    {'xlim': (-0.25, 0.25),
                     'ylim': (-0.25, 0.25)},
                'multimodal_swissroll_overlap':
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)} ,
                'multimodal_swissroll':                  
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)} ,
                'alanine_dipeptide':
                    {'xlim': (-180, 180),
                     'ylim': (-180, 180)}
                }
            
            if config.dataset == "multigaussian_1D":
                ax.hist(x.cpu().numpy().flatten(), bins=50, range=(-2.5, 2.5))
            elif config.dataset in lim_dict.keys():
                ax.set_xlim(lim_dict[config.dataset]['xlim'])
                ax.set_ylim(lim_dict[config.dataset]['ylim'])
                
                if config.dataset == "alanine_dipeptide":
                    s=None
                    alpha=0.01
                else:
                    s=3
                    alpha=0.1
                
                ax.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1],
                           alpha=alpha, c="green", edgecolor=None, s=s)
            else:
                ax.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], s=3)
            
            ax.set_title(f"iter: {step}", fontsize=20)
            
            plt.savefig(os.path.join(this_sample_dir,
                        'sample_rank_%d.png' % global_rank))
            plt.close()

            if config.sde in ('cld', 'active'):
                np.save(os.path.join(this_sample_dir, 'sample_x'), x.cpu())
                np.save(os.path.join(this_sample_dir, 'sample_v'), v.cpu())
            else:
                np.save(os.path.join(this_sample_dir, 'sample'), x.cpu())
        dist.barrier()

        if config.save_freq is not None:
            if step % config.save_freq == 0 and step >= config.save_threshold:
                if global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % step)
                    if not os.path.isfile(checkpoint_file):
                        save_checkpoint(checkpoint_file, state)
                dist.barrier()

        # Training
        start_time = time.time()

        if config.dataset == "ising_2D":
            x = inf_data_gen(config.dataset, config.training_batch_size, config).to(
                config.device)
        else:
            x = inf_data_gen(config.dataset, config.training_batch_size).to(
                config.device)
        if config.is_image: # For Ising model
            x = scaler(x)
            x = x.to(config.device)
        loss = train_step_fn(state, x)

        if step % config.log_freq == 0:
            loss = reduce_tensor(loss, global_size)
            if global_rank == 0:
                logging.info('Iter %d/%d Loss: %.4f Time: %.3f' % (step + 1,
                             config.n_train_iters, loss.item(), time.time() - start_time))
                writer.add_scalar('training_loss', loss, step)
            dist.barrier()

        step += 1

    if global_rank == 0:
        logging.info('Finished after %d iterations.' % config.n_train_iters)
        logging.info('Saving final checkpoint.')
        save_checkpoint(os.path.join(
            checkpoint_dir, 'final_checkpoint.pth'), state)
    dist.barrier()


def evaluate(config, workdir):
    ''' Main evaluation script. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    set_seeds(global_rank, config.seed + config.eval_seed)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    eval_dir = os.path.join(workdir, config.eval_folder)
    sample_dir = os.path.join(eval_dir, 'samples')
    if global_rank == 0:
        logging.info(config)
        make_dir(sample_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    elif config.sde == 'passive':
        sde = sde_lib.PassiveDiffusion(config, beta_fn, beta_int_fn)
    elif config.sde == 'active':
        sde = sde_lib.ActiveDiffusion(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.sde)

    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())
    score_model = DDP(score_model, device_ids=[local_rank])
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)
    
    # Utility functions to map images from [0, 1] to [-1, 1] and back.
    # Used for Ising model only 
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    optimize_fn = optimization_manager(config)
    eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    ckpt_path = os.path.join(checkpoint_dir, config.ckpt_file)
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    step = int(state['step'])
    if global_rank == 0:
        logging.info('Evaluating at training step %d' % step)
    dist.barrier()

    if config.eval_loss:
        eval_loss = compute_eval_loss(
            config, sde, state, eval_step_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Testing loss: %.5f" % eval_loss.item())
        dist.barrier()

    if config.eval_likelihood:
        mean_nll = compute_non_image_likelihood(
            config, sde, state, likelihood_fn, inf_data_gen)
        if global_rank == 0:
            logging.info("Mean NLL: %.5f" % mean_nll.item())
        dist.barrier()

    if config.eval_sample:
        global_size = config.global_size
        inverse_scaler = get_data_inverse_scaler(config)
        num_sampling_rounds = config.eval_sample_samples // (
            config.sampling_batch_size * global_size) + 1
        
        fig, ax = plt.subplots()
            
        ax.set_aspect('equal')
            
        lim_dict = {'multigaussian_2D':
                    {'xlim': (-3, 3),
                     'ylim': (-1.5, 1.5)},
                'multigaussian_2D_close':
                    {'xlim': (-3, 3),
                     'ylim': (-1.5, 1.5)},
                'diamond':                  
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)},
                'diamond_close':            
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)},
                'swissroll':                
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)},
                'multimodal_swissroll_overlap':
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)} ,
                'multimodal_swissroll':                  
                    {'xlim': (-1, 1),
                     'ylim': (-1, 1)} 
                }    

        for r in range(num_sampling_rounds):
            if global_rank == 0:
                logging.info('sampling -- round: %d' % r)
            dist.barrier()

            x, _, nfe = sampling_fn(score_model)
            #x = inverse_scaler(x)
            samples = x #.clamp(0.0, 1.0)

            torch.save(samples, os.path.join(
                sample_dir, 'samples_%d_%d.pth' % (r, global_rank)))
            np.save(os.path.join(sample_dir, 'nfes_%d_%d.npy' %
                    (r, global_rank)), np.array([nfe]))
        # x, _, nfe = sampling_fn(score_model)
        # logging.info('NFE: %d' % nfe)

            if config.dataset == "multigaussian_1D":
                ax.hist(x.cpu().numpy().flatten(), bins=50, range=(-2.5, 2.5))
            elif config.dataset in lim_dict.keys():
                ax.set_xlim(lim_dict[config.dataset]['xlim'])
                ax.set_ylim(lim_dict[config.dataset]['ylim'])
                
                ax.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1],
                           alpha=0.1, c="green", edgecolor=None, s=3)
            else:
                ax.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], s=3)
            
        plt.savefig(os.path.join(sample_dir,
                        'sample_rank_%d.png' % global_rank))
        plt.close()

def reverse_forces(config, workdir):
    ''' Visualization of reverse diffusion forces. '''

    local_rank = config.local_rank
    global_rank = config.global_rank
    set_seeds(global_rank, config.seed + config.eval_seed)

    torch.cuda.device(local_rank)
    config.device = torch.device('cuda:%d' % local_rank)

    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    eval_dir = os.path.join(workdir, config.eval_folder)
    sample_dir = os.path.join(eval_dir, 'samples')
    if global_rank == 0:
        logging.info(config)
        make_dir(sample_dir)
    dist.barrier()

    beta_fn = build_beta_fn(config)
    beta_int_fn = build_beta_int_fn(config)

    if config.sde == 'vpsde':
        sde = sde_lib.VPSDE(config, beta_fn, beta_int_fn)
    elif config.sde == 'cld':
        sde = sde_lib.CLD(config, beta_fn, beta_int_fn)
    elif config.sde == 'passive':
        sde = sde_lib.PassiveDiffusion(config, beta_fn, beta_int_fn)
    elif config.sde == 'active':
        sde = sde_lib.ActiveDiffusion(config, beta_fn, beta_int_fn)
    else:
        raise NotImplementedError('SDE %s is unknown.' % config.sde)

    score_model = mutils.create_model(config).to(config.device)
    broadcast_params(score_model.parameters())
    score_model = DDP(score_model, device_ids=[local_rank])
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema_rate)

    optim_params = score_model.parameters()
    optimizer = get_optimizer(config, optim_params)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    optimize_fn = optimization_manager(config)
    eval_step_fn = losses.get_step_fn(False, optimize_fn, sde, config)

    sampling_shape = (config.sampling_batch_size,
                      config.data_dim)
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, config.sampling_eps)

    likelihood_fn = likelihood.get_likelihood_fn(config, sde)

    ckpt_path = os.path.join(checkpoint_dir, config.ckpt_file)
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
           
    def probability_flow_ode(model, u, t):
        ''' 
        The "Right-Hand Side" of the ODE. 
        '''
        score_fn = get_score_fn(config, sde, model, train=False)
        rsde = sde.get_reverse_sde(score_fn, probability_flow=True)
        return rsde(u, t)[0] # return reverse drift

    def denoising_fn(model, u, t):
        score_fn = get_score_fn(config, sde, model, train=False)
        discrete_step_fn = sde.get_discrete_step_fn(
            mode='reverse', score_fn=score_fn)
        u, u_mean = discrete_step_fn(u, t, eps)
        return u_mean

    def ode_sampler(model, u=None, time_arr=None, t_start=None, t_end=None):
        with torch.no_grad():
            if u is None:
                x, v = sde.prior_sampling(sampling_shape)
                if sde.is_augmented:
                    u = torch.cat((x, v), dim=1)
                else:
                    u = x

            def ode_func(t, u):
                global nfe_counter
                nfe_counter += 1
                vec_t = torch.ones(
                    sampling_shape[0], device=u.device, dtype=torch.float64) * t
                dudt = probability_flow_ode(model, u, vec_t)
                return dudt

            global nfe_counter
            nfe_counter = 0
            
            if time_arr is None and t_start is not None and t_end is not None:
                time_tensor = torch.tensor(
                    [0, t_end-t_start], dtype=torch.float64, device=config.device)
            else:
                time_tensor = torch.tensor(
                    time_arr, dtype=torch.float64, device=config.device
                )
            solution = odeint(ode_func,
                              u,
                              time_tensor,
                              rtol=config.sampling_rtol,
                              atol=config.sampling_atol,
                              method=config.sampling_solver,
                              options=config.sampling_solver_options)
            
            if config.denoising:
                for step_idx, u in enumerate(solution):
                    u = denoising_fn(model, u, config.max_time - eps)
                    nfe_counter += 1
                    
                    solution[step_idx] = u
            
            return solution
    
    def ode_func(t, u):
        global nfe_counter
        nfe_counter += 1
        vec_t = torch.ones(
            sampling_shape[0], device=u.device, dtype=torch.float64) * t
        dudt = probability_flow_ode(score_model, u, vec_t)
        return dudt
    
    time_step = config.sampling_eps
    num_time_steps = 1000
    time_arr = np.linspace(0, config.max_time - time_step, num_time_steps)
    u=None

    score_fn = get_score_fn(config, sde, score_model, train=False)

    t_start = 0
    t_end = config.max_time
    solution = ode_sampler(score_model, u, time_arr=time_arr)

    score_mean = []
    score_std = []

    for idx, u in enumerate(solution):
        
        if idx%10 == 0: # and idx > int(num_time_steps*0.9):
            vec_t = torch.ones(sampling_shape[0], 
                               device=u.device, 
                               dtype=torch.float64) * \
                                   time_arr[idx]
    
            score_u = score_fn(u, vec_t)
            # _, diffusion = sde.sde(u, vec_t)
            
            fig, ax = plt.subplots()
            # ax.set_aspect('equal')
            time = "{:0.2f}".format(time_arr[idx])

            ax.set_title(f"{config.sde}, {config.dataset}, t={time}")
            ax.set_ylim(0,1)
            ax.set_xlim(-5, 5)
            fake_x_arr = np.linspace(-5, 5, 10000)
            fake_y_arr = np.exp(-(fake_x_arr)**2)/np.sqrt(2*np.pi)
            if sde.is_augmented:
                x, v = torch.chunk(u, 2, dim=1)
                # _, score = torch.chunk(score_u, 2, dim=1)
                # score = score.cpu().detach().numpy()
                # print(score.shape)
                XY = v.cpu().detach().numpy()
            else:
                x = u
                XY = x.cpu().detach().numpy()
            score = score_u.cpu().detach().numpy()

            # ax.scatter(x[:,0].cpu().detach().numpy(), x[:,1].cpu().detach().numpy())
            # ax.quiver(XY[:,0], XY[:,1], score[:,0], score[:,1])
            # score_x_mean = torch.mean(score_u[:,0])
            ax.plot(fake_x_arr, fake_y_arr, label="N(0,I)", color='black')
            ax.hist(np.array(score[:,0]), bins = 50, density=True, range=(-5, 5), label="x", alpha=0.5)
            ax.hist(np.array(score[:,1]), bins = 50, density=True, range=(-5, 5), label="y", alpha=0.5)
            ax.legend()
            # plt.show()
            
            time = "{:0.2f}".format(time_arr[idx])
            fname = f"{config.sde}_{config.dataset}_t{time}.png"
            
            plt.savefig(fname)
    
            plt.close()
