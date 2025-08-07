#!/bin/bash
#SBATCH --job-name=compress_1M           # create a short name for your job
#SBATCH --output=output.txt  # output file name, %A is job ID and %a is array index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint="nomig&gpu40"
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=6:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end
#SBATCH --mail-user=hywang@princeton.edu

module purge
module load anaconda3/2024.10
conda activate torch-env

export OMP_NUM_THREADS=1

python compress_train_ds.py