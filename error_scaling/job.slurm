#!/bin/bash
#SBATCH --job-name=m5-80-479           # create a short name for your job
#SBATCH --output=output.txt  # output file name, %A is job ID and %a is array index
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=300M         # memory per cpu-core (4G is default)
#SBATCH --array=80-479                # job array $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
#SBATCH --time=14:50:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end
#SBATCH --mail-user=hywang@princeton.edu

module purge
module load anaconda3/2024.10
conda activate torch-env

export OMP_NUM_THREADS=4

python error_scaling.py --task-index ${SLURM_ARRAY_TASK_ID} --out error_list.csv