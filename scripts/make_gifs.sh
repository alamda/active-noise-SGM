#!/bin/bash

sdes=("vpsde"
	  "passive"
	  "active"
	  "cld"
	  )

datasets=("diamond"
		  "diamond_close"
		  "multigaussian_2D"
		  "multigaussian_2D_close"
		  )

for dataset in "${datasets[@]}"
do

for sde in "${sdes[@]}"
do
	magick convert ${sde}_${dataset}_t*.png ${sde}_${dataset}.gif

done
done
