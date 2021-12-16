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
#SBATCH -o ./slurm_logs/slurm-sbatch_1-12-%j.out

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
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name ruslan --dataset_split val -pn_controller Proportional -pn_multiplier 0.0
echo FINISHED at $(date)
