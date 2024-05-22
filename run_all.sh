#!/bin/bash

out_file="simulations/progress.out"

[ -e $out_file ] && rm $out_file

sims=("vpsde_multiswiss"
	  "vpsde_swiss"
	  "cld_multiswiss"
	  "cld_swiss"
	  "passive_multiswiss"
	  "passive_swiss"
	  "active_multiswiss"
	  "active_swiss")

for sim in "${sims[@]}"
do
		
	root_dir="simulations/${sim}"

	if [ ! -d $root_dir ]
	then
		python main.py -cc config_${sim}_train.txt --root $root_dir --workdir work_dir/dataset 
		
		echo "${sim} done" >> $out_file
	else
		echo "${sim} directory exists, not running" >> $out_file
	fi

done


