#!/bin/bash
#SBATCH --job-name=scap_cpu_profiler-torch
#SBATCH -o /usr/users/%u/%x-%A-%a.log
#SBATCH -t 06:00:00                  # TODO: estimate the time you will need
#SBATCH -w amp078
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-user=hgronen@gwdg.de # TODO: change this to your mailaddress!

# Prepare the environment.
module load python/3.9
module load anaconda3
source activate /scratch1/users/hgronen/.conda/scap

# Printing out some info.
echo "Job name: ${SLURM_JOB_NAME}"
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

echo "logdir"
echo /scratch1/users/hgronen/torch-log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}

# Run the script.
python -u train.py -l /scratch1/users/hgronen/torch-log/${SLURM_JOB_NAME}_${SLURM_JOB_ID} -t False -p True -d False
