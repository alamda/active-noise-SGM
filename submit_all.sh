#!/bin/bash

dirdir=("simulations_ml"
	    "simulations_rv1"
		"simulations_rv2")

sims=("vpsde_multiswiss"
      "vpsde_multiswiss_overlap"
	  "vpsde_swiss"
	  "cld_multiswiss"
	  "cld_multiswiss_overlap"
	  "cld_swiss"
	  "passive_multiswiss"
	  "passive_multiswiss_overlap"
	  "passive_swiss"
	  "active_multiswiss"
	  "active_multiswiss_overlap"
	  "active_swiss")

for dir in "${dirdir[@]}"
do

for sim in "${sims[@]}"
do
		
	root_dir="${dir}/${sim}"

	if [ ! -d $root_dir ]
	then
		sbatch --account=pi-dfreedman --partition=schmidt-gpu --gres=gpu:1 --qos=schmidt --time=3:00:00 --wrap "conda run -n pytorch python main.py -cc ${dir}/config_${sim}_train.txt --root $root_dir --workdir work_dir/dataset"
	fi

done

done

