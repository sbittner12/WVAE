#!/bin/sh
#
#SBATCH --account=stats     # The account name for the job.
#SBATCH --job-name=midi-wavenet    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (1-4 are valid).
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=4:00:00              # The time the job will take to run.

module load cuda80/toolkit cuda80/blas cudnn/6.0_8

python /rigel/stats/users/srb2201/final_project/midi-wavenet/train.py --max_dilation_pow=7 --expansion_reps=3 --dil_chan=$1 --res_chan=$2 --skip_chan=$3

