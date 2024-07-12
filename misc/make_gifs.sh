#!/bin/bash

sdes=(#"vpsde"
	  "passive"
	  "active"
	  #"cld"
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
	magick convert ANALYTIC_${sde}_${dataset}_t*.png ANALYTIC_${sde}_${dataset}.gif

done
done

[ ! -d gifs ] && mkdir gifs

mv *.gif gifs/
