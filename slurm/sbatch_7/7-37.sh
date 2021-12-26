#!/bin/bash
#SBATCH --chdir /home/rajic/xode/ruslan
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 92160
#SBATCH --time 24:00:00
#SBATCH --account cs503
#SBATCH --reservation courses
#SBATCH --gres=gpu:1
#SBATCH -q gpu
#SBATCH -o ./slurm_logs/slurm-sbatch_7-37-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Modules
module purge
module load gcc/8.4.0-cuda
module load cuda/10.1
module load parallel

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ruslan

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth --num_episodes 1000 --num_episode_sample 1000 --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name ddppo --dataset_split train --seed 72 -vc Motion_Blur -vs 3 --pyrobot_noise_multiplier 0.0
echo FINISHED at $(date)
