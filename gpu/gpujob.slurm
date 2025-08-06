#!/bin/bash
#SBATCH --job-name=compress_1M           # create a short name for your job
#SBATCH --output=output.txt  # output file name, %A is job ID and %a is array index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig          # mig is for small task trial
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=2:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end
#SBATCH --mail-user=hywang@princeton.edu

module purge
module load anaconda/version
conda activate pytorch_env

export OMP_NUM_THREADS=1

python code.py