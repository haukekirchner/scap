#!/bin/bash
#SBATCH --job-name=train-resnet
#SBATCH -p gpu                       # request gpu node for the training
#SBATCH -t 00:10:00                  # TODO: estimate the time you will need
#SBATCH -G gtx1080                   # requesting specific GPU, run sinfo -p gpu --format=%N,%G # to see what is available
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-user=hgronen@gwdg.de # TODO: change this to your mailaddress!

# Prepare the environment.
module load python/3.9
module load anaconda3
module load cuda
source activate test_resnet

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script.
python -u train_resnet.py
