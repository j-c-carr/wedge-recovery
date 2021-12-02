#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --account=def-acliu
#SBATCH --mem=47G        # memory per node
#SBATCH --time=2:30:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=jonathan.colacocarr@mail.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.7.7  
module load cuda/11.0 cudnn

# For tensorboard profiler
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64/

PROJECTDIR=/home/jccarr/wedge-recovery/coeval-wedge-recovery


TITLE="z8.5-10_test_2"
printf "\n\n-----\t Start of $title\t-----\n" >> test.log

# Prepare log directory
datetime=$(date '+%Y-%m-%d-%H%M')

# Prepare virtualenv
source /home/jccarr/projects/def-acliu/jccarr/.env/bin/activate

# title datetime root_dir log_dir config_file data_file 
time python3 $PROJECTDIR/bin/main.py \
        $TITLE \
        $datetime \
        $PROJECTDIR/out \
        $PROJECTDIR/in/model_config.yml \
        scratch/datasets/z8.5-10_HII-DIM-128_BOX_LEN-128_lite_xh_boxes.h5 \
        --predict \
        --old_model_loc scratch/model-checkpoints/2021-11-08-1108_z8.5-10_HII-DIM-128_BOX-LEN-128_dice_bintru-checkpoint.h5 \
        --save_validation_results --results_dir scratch/results
