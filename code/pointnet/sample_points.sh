#!/bin/bash
#SBATCH --job-name=sample_points
#SBATCH -o /usr/users/%u/%x-%A-%a.log
#SBATCH -t 00:05:00                  # TODO: estimate the time you will need
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
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env


# Run the script.
python -u sample_points.py
