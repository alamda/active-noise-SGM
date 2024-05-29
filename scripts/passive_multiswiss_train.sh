#!/bin/sh

root_dir="simulations/passive_multiswiss"

[ -d $root_dir ] && rm -Ir $root_dir

python main.py -cc config_passive_multiswiss_train.txt --root $root_dir --workdir work_dir/multi_swiss_roll 

