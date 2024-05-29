#!/bin/sh

root_dir="simulations/active_multiswiss"

[ -d $root_dir ] && rm -Ir $root_dir

python main.py -cc config_active_multiswiss_train.txt --root $root_dir --workdir work_dir/multi_swiss_roll 

