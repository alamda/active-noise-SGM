#!/bin/sh

root_dir="simulations/passive_swiss"

[ -d $root_dir ] && rm -Ir $root_dir

python main.py -cc config_passive_swiss_train.txt --root $root_dir --workdir work_dir/swiss_roll 

