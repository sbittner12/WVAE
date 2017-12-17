#!/bin/sh
#
#SBATCH --account=stats     # The account name for the job.
#SBATCH --job-name=midi-wavenet    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (1-4 are valid).
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=4:00:00              # The time the job will take to run.

module load cuda80/toolkit cuda80/blas cudnn/6.0_8

python /rigel/stats/users/srb2201/final_project/WVAE/train.py --data_set=$1 --max_dilation_pow=$2 --expansion_reps=$3

