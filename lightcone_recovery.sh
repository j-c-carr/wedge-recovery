#!/bin/bash

PROJECTDIR=/home/jccarr/wedge-recovery/lightcone-recovery

title="z7-9_HII-DIM-128_BOX-LEN-256_full_gillet_128-boxes"
printf "\n\n-----\t Start of $title\t-----\n" >> test.log

datetime=$(date '+%Y-%m-%d-%H%M')

# Prepare virtualenv
source /home/jccarr/projects/def-acliu/jccarr/.env/bin/activate

# title datetime root_dir log_dir config_file data_file 
python3 $PROJECTDIR/bin/main.py \
        $title \
        $datetime \
        $PROJECTDIR \
        $LOGDIR \
        $PROJECTDIR/in/model_config.yml \
        scratch/datasets/z7-9_HII-DIM-128_BOX_LEN-256_full_gillet_v1.h5 \
        --sample_data_only
