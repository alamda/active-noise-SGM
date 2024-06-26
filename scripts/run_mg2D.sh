#!/bin/bash

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=(#"simulations_ml"
	    #"simulations_rv1"
		"simulations_rv2"
		)

sims=("passive_mg2d"
	  "vpsde_mg2d"
	  "active_mg2d"
	  "cld_mg2d"
      )

for dir in "${dirdir[@]}"
do

for sim in "${sims[@]}"
do
		
	root_dir="${dir}/${sim}"

	if [ ! -d $root_dir ]
	then
		python main.py -cc ${dir}/config_${sim}_train.txt --root $root_dir --workdir work_dir/dataset 
		
		echo "${dir}/${sim} done" >> $out_file
	else
		echo "${dir}/${sim} directory exists, not running" >> $out_file
	fi

done

done

