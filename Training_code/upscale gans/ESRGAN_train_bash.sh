#!/bin/bash
# This line is required to inform the Linux
#command line to parse the script using
#the bash shell

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for
#various combinations
#SBATCH -N 1
#SBATCH -c 4

# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH -p "ug-gpu-small"
#SBATCH --qos="short"
#SBATCH --gres=gpu
#SBATCH -t 02-00:00:00
#SBATCH -e ESRGAN_%j.err
#SBATCH -o ESRGAN_%j.out
#SBATCH --mem=28g
#SBATCH --job-name esrgan

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"
# Source the bash profile (required to use the module command)
source /home2/vcqf59/master_cloud/bin/activate
module load cuda/11.2-cudnn8.1.0


PROGRAM="ESRGAN_train.py"
python -u ${PROGRAM}
BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"