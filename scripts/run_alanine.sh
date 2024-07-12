#!/bin/bash

datasets=("alanine_dipeptide"
		  )

out_file="progress.out"

[ -e $out_file ] && rm $out_file

dirdir=("alanine_dipeptide"
		)

sims=(#"vpsde"
	  "passive"
	  "active"
	  "vpsde"
	  "cld"
      )

for dir in "${dirdir[@]}"
do

for dataset in "${datasets[@]}"
do

for sim in "${sims[@]}"
do
		
	root_dir="${dir}/${sim}_${dataset}"

	if [ ! -d $root_dir ]
	then
        echo "starting $root_dir" >> $out_file
		
        python main.py -cc ${dir}/config_${sim}_train.txt --root $root_dir --workdir work_dir/dataset --dataset $dataset 
		
		echo "$root_dir done" >> $out_file
	else
		echo "$root_dir directory exists, not running" >> $out_file
	fi

done

done

done

