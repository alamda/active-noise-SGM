#!/bin/bash

datasets=("diamond_close"
          "multigaussian_2D_close"
          #"swissroll"
		  #"multimodal_swissroll"
		  "multimodal_swissroll_overlap"
		  "diamond"
		  #"multigaussian_1D"
		  "multigaussian_2D"
		  )

snapshot_freqs=(#"10"
				"100"
				"1000"
				"10000"
				#"100000"
				)

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=(#"simulations_ml"
	    #"simulations_rv1"
		"simulations_rv2"
		)

sims=("vpsde"
	  "passive"
	  "active"
	  "cld"
      )

for dir in "${dirdir[@]}"
do

for snapshot_freq in "${snapshot_freqs[@]}"
do

for dataset in "${datasets[@]}"
do

	n_train_iters="${snapshot_freq}00"

for sim in "${sims[@]}"
do
		
	root_dir="${dir}/${sim}_${dataset}_${n_train_iters}"

	if [ ! -d $root_dir ]
	then
        echo "starting $root_dir" >> $out_file
		
        python main.py -cc ${dir}/config_${sim}_train.txt --root $root_dir --workdir work_dir/dataset --dataset $dataset --snapshot_freq $snapshot_freq --n_train_iters $n_train_iters
		
		echo "$root_dir done" >> $out_file
	else
		echo "$root_dir directory exists, not running" >> $out_file
	fi

done

done

done

done
