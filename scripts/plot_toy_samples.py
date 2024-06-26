import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')

from util.toy_data import inf_data_gen

if __name__=="__main__":
    
    dataset_list = ['multimodal_swissroll',
                    'diamond',
                    'diamond_close',
                    'swissroll',
                    'multimodal_swissroll_overlap',
                    'multigaussian_2D',
                    'multigaussian_2D_close']
    
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
    
    batch_size = 5120
    
    for dataset in dataset_list:
        x = inf_data_gen(dataset, batch_size)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        
        ax.set_xlim(lim_dict[dataset]['xlim'])
        ax.set_ylim(lim_dict[dataset]['ylim'])
        
        ax.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1],
                           alpha=0.1, c="green", edgecolor=None, s=3)

        plt.savefig(f'{dataset}.png')
        plt.close()