#!/bin/sh

root_dir="simulations/active_swiss"

[ -d $root_dir ] && rm -Ir $root_dir

python main.py -cc config_active_swiss_train.txt --root $root_dir --workdir work_dir/swiss_roll 

