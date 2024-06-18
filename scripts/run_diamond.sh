#!/bin/bash

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=(#"simulations_ml"
	    #"simulations_rv1"
		"simulations_rv2"
		)

sims=(#"vpsde_multiswiss"
      #"vpsde_multiswiss_overlap"
	  #"vpsde_swiss"
	  #"cld_multiswiss"
	  #"cld_multiswiss_overlap"
	  #"cld_swiss"
	  #"passive_multiswiss"
	  #"passive_multiswiss_overlap"
	  "passive_diamond"
	  #"active_multiswiss"
	  #"active_multiswiss_overlap"
	  "active_diamond"
	  "cld_diamond"
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

