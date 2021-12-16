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
#SBATCH -o ./slurm_logs/slurm-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Modules
module purge
module load gcc/8.4.0-cuda
module load cuda/10.1

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ruslan

# Run configuration
export CUDA_VISIBLE_DEVICES=0
export AGENT_ARGS="agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on"
export CHALLENGE_CONFIG_FILE=config_files/challenge_pointnav2021.local.rgbd.GPU.yaml
export AGENT_NAME_AND_SPLIT="--agent_name ruslan --dataset_split val_mini"
COMMANDS=(
    "python $AGENT_ARGS $AGENT_NAME_AND_SPLIT --challenge_config_file $CHALLENGE_CONFIG_FILE --visual_corruption Defocus_Blur --visual_severity 3 --num_episodes 30"
#    "python $AGENT_ARGS $AGENT_NAME_AND_SPLIT --challenge_config_file $CHALLENGE_CONFIG_FILE --visual_corruption Defocus_Blur --visual_severity 5 --num_episodes 30"
)

# Run
date
printf "Run configured and environment setup. Time to srun\n\n"
for  (( i=0; i<${#COMMANDS[@]}; i++ ));
do
#     ${COMMANDS[$i]} &>> ./slurm_logs/slurm-$SLURM_JOBID.out.$i &
#     srun --output ./slurm_logs/slurm-$SLURM_JOBID.out.$i ${COMMANDS[$i]} &
#     srun ${COMMANDS[$i]} &> ./slurm_logs/slurm-$SLURM_JOBID.out.$i &
#     (srun ${COMMANDS[$i]} &> ./slurm_logs/slurm-$SLURM_JOBID.out.$i) &
#     sleep 3
#     srun ${COMMANDS[$i]} &
#     srun -n 1 -N 1 --exclusive ${COMMANDS[$i]} &
     srun ${COMMANDS[$i]}
done
wait
echo FINISHED at $(date)
