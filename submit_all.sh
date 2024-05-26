#!/bin/bash

port=7021

root_dir="$(pwd)"
sim_dir="simulations"

[ ! -d $sim_dir ] && mkdir $sim_dir

weightings=("reweightedv2"
			#"likelihood"
			#"reweightedv1"
		   )

sdes=("vpsde"
	  "cld"
	  "passive"
	  "active"
	 )

datasets=(#"multimodal_swissroll"
		  "multimodal_swissroll_overlap"
		  #"swissroll"
		  )

taus=("0.01"
	  #"0.05"
	  "0.1"
	  #"0.25"
	  "0.5"
	  #"0.75"
	  )

cd $sim_dir

for weighting in "${weightings[@]}"
do
	for sde in "${sdes[@]}"
	do
		for dataset in "${datasets[@]}"
		do
			for tau in "${taus[@]}"
			do
				sim_sub_dir="${weighting}_${sde}_${dataset}_tau${tau}"

				[ ! -d $sim_sub_dir ] && mkdir $sim_sub_dir
				
				echo 'sbatch --account=pi-dfreedman --partition=schmidt-gpu --gres=gpu:1 --qos=schmidt --time=3:00:00 --name '${sim_sub_dir}' --wrap "echo '${sim_sub_dir}' ; conda run -n pytorch python '${root_dir}'/main.py -cc '${root_dir}'/config_toy.txt --root '${sim_sub_dir}' --workdir work_dir/dataset --master_port '${port}' --weighting '${weighting}' --sde '${sde}' --dataset '${dataset}' --tau '${tau}'"' > ${sim_sub_dir}/sbatch.sh

				((port++))
				
				sh $sim_sub_dir/sbatch.sh
			done

		done
	done
done

cd $root_dir
