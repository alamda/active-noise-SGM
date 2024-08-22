#!/bin/bash

datasets=("ising_2D"
		  )

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=("ising_2D"
		)

sims=(
	  "passive"
	  "active"
	  "cld"
      #"vpsde"
      )

betas=(
		#"0.01"
	    #"0.5"
	    #"1.0"
	    #"1.5"
	    "2"
	    #"2.25"
	    #"2.5"
	    #"3"
		#"3.5"
		#"4"
	   )

for dir in "${dirdir[@]}"
do

for dataset in "${datasets[@]}"
do

for sim in "${sims[@]}"
do

for beta in "${betas[@]}"
do		
	root_dir="${dir}/${sim}_${dataset}_beta${beta}"

	if [ ! -d $root_dir ]
	then
        echo "starting $root_dir" >> $out_file
		mode="train"
	else
		date="$(date '+%s')"
		mode="continue --seed 0 --cont_nbr ${date}"
        echo "continuing $root_dir" >> $out_file
	fi
	
	mode="eval --eval_sample --max_sampling_time 1 --eval_sample_samples 100 --eval_folder eval_ode_1 --ckpt_file final_checkpoint.pth"

    python main.py -cc ${dir}/config_ising_2D_${sim}_train.txt \
                   --offline \
                   --debug \
                   --root $root_dir \
                   --workdir work_dir/dataset \
                   --dataset ising_2D \
                   --ising_lattice_temp $beta \
                   --mode $mode && \
    echo "$root_dir done" >> $out_file
    
done

done

done

done

