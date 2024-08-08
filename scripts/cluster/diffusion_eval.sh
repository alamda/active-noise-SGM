#!/bin/sh

#SBATCH --account=beagle3-exusers 
#SBATCH --time=01:30:00 
#SBATCH --partition=beagle3 
#SBATCH --gpus=1

source activate /home/alexandral/miniforge3/envs/pytorch

conda run -p /home/alexandral/miniforge3/envs/pytorch \
python /home/alexandral/git_repos/active-noise-SGM/main.py \
		-cc config_ising_2D_cld_train.txt \
    	--offline \
    	--root diffusion/ising_2D/cld_ising_2D_beta1.0 \
    	--workdir work_dir/dataset \
    	--dataset ising_2D \
    	--ising_lattice_temp 1.0 \
    	--mode eval \
		--eval_sample \
		--eval_sample_samples 10 \
		--eval_folder eval \
		--ckpt_file final_checkpoint.pth
