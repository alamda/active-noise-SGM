#!/bin/bash

datasets=("ising_2D"
		  )

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=("ising_2D"
		)

sims=(#"vpsde"
	  #"passive"
	  #"active"
	  #"vpsde"
	  "cld"
      )

betas=("0.01"
	   "0.1"
	   "0.5"
	   "1.0"
	   "1.5"
	   "2"
	   "2.5"
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
		mode="continue --seed 0 --cont_nbr ${date '+%s'}"
        echo "continuing $root_dir" >> $out_file
	fi
		
    python main.py -cc ${dir}/config_ising_2D_${sim}_train.txt --root $root_dir --workdir work_dir/dataset --dataset ising_2D --ising_lattice_temp $beta --mode $mode
		
	echo "$root_dir done" >> $out_file

done

done

done

done

