# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import numpy as np
from sklearn.datasets import make_swiss_roll


def inf_data_gen(dataset, batch_size):
    if dataset == 'multimodal_swissroll':
        NOISE = 0.2
        MULTIPLIER = 0.01
        OFFSETS = [[0.8, 0.8], [0.8, -0.8], [-0.8, -0.8], [-0.8, 0.8]]

        idx = np.random.multinomial(batch_size, [0.2] * 5, size=1)[0]

        sr = []
        for k in range(5):
            sr.append(make_swiss_roll(int(idx[k]), noise=NOISE)[
                      0][:, [0, 2]].astype('float32') * MULTIPLIER)

            if k > 0:
                sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

        data = np.concatenate(sr, axis=0)[np.random.permutation(batch_size)]
        return torch.from_numpy(data.astype('float32'))

    elif dataset in ('diamond', 'diamond_close'):
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
        return torch.from_numpy(data.astype('float32'))

    elif dataset == 'swissroll':
        NOISE = 0.2
        MULTIPLIER = 0.01
        OFFSETS = [[0.0, 0.0]]

        idx = np.random.multinomial(batch_size, [0.2], size=1)[0]

        sr = []
        for k in range(1):
            sr.append(make_swiss_roll(int(idx[k]), noise=NOISE)[
                      0][:, [0, 2]].astype('float32') * MULTIPLIER)

            if k > 0:
                sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

        data = np.concatenate(sr, axis=0)[np.random.permutation(batch_size)]
        return torch.from_numpy(data.astype('float32'))
    
    if dataset == 'multimodal_swissroll_overlap':
        NOISE = 0.2
        MULTIPLIER = 0.01
        OFFSETS = [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]

        idx = np.random.multinomial(batch_size, [0.2] * 5, size=1)[0]

        sr = []
        for k in range(5):
            sr.append(make_swiss_roll(int(idx[k]), noise=NOISE)[
                      0][:, [0, 2]].astype('float32') * MULTIPLIER)

            if k > 0:
                sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

        data = np.concatenate(sr, axis=0)[np.random.permutation(batch_size)]
        return torch.from_numpy(data.astype('float32'))
    
    if dataset == 'multigaussian_1D':     
        mu_list = [-1.2, 1.2]
        sigma_list = [0.5, 0.5]
        pi_list = [0.5, 0.5]

        means = np.array(mu_list)
        sigmas = np.array(sigma_list)
        weights = np.array(pi_list)/np.sum(np.array(pi_list))

        index = np.random.choice(range(len(mu_list)), 
                                 size=batch_size, 
                                 replace=True,
                                 p=weights)
        
        noise = np.random.randn(batch_size, 1)

        data = means[index] + noise.flatten() * sigmas[index]
        
        return torch.from_numpy(data.astype('float32'))
    
    if dataset in ('multigaussian_2D', 'multigaussian_2D_close'):
        if dataset == 'multigaussian_2D':
            mu_x_list = [-1.2, 1.2]
        elif dataset == 'multigaussian_2D_close':
            mu_x_list = [-1.0, 1.0]
        mu_y_list = [0., 0.]
        sigma_list = [0.5, 0.5]
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
        
        return data

    else:
        raise NotImplementedError(
            'Toy dataset %s is not implemented.' % dataset)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    dataset = "multigaussian_2D"
    batch_size = 1000
    num_bins = 50
    
    fig, ax = plt.subplots()
    
    if dataset == "multigaussian_1D":
        sample = inf_data_gen(dataset, batch_size)
        y_arr = np.zeros_like(sample)
        
        plt.hist(sample, bins=num_bins)
        
        plt.show()
    
    elif dataset == "multigaussian_2D":
        sample = inf_data_gen(dataset, batch_size)
        
        ax.hist2d(sample[:,0], sample[:,1])
        
        plt.show()