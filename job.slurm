#!/bin/bash
#SBATCH --job-name=dataset           # create a short name for your job
#SBATCH --output=output.txt  # output file name, %A is job ID and %a is array index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=9G         # memory per cpu-core (4G is default)
#SBATCH --array=380-419
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=hywang@princeton.edu

module purge
module load anaconda3/2024.10
conda activate torch-env

export OMP_NUM_THREADS=1

cd scaling_law/
python trainds.py