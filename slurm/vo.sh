#!/bin/bash
#SBATCH --chdir /home/rajic/xode/vo
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 10
#SBATCH --mem 40960
#SBATCH --time 12:00:00
#SBATCH --account cs503
#SBATCH --reservation courses
#SBATCH --gres=gpu:1

echo "Hello world!"

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)
#######################
##~~~~~~~~~~~~~~~~~~~##

source venv/bin/activate
module purge
module load gcc/8.4.0-cuda
module load cuda/10.1



config_file="configs/point_nav_habitat_challenge_2020.yaml"
config_file_template="configs/point_nav_habitat_challenge_2020_TEMPLATE.yaml"
export MAGNUM_LOG="quiet"
export MAGNUM_GPU_VALIDATION=ON
GLOG_minloglevel=2

datasets=(val_mini val train)
# datasets=(val_mini train)
# datasets=(val)
for dataset in "${datasets[@]}"; do
    cat "$config_file_template" | sed "s/\btrain\b/$dataset/g" > "${config_file}"
    cat "$config_file"

    export POINTNAV_VO_ROOT=$PWD

    export NUMBA_NUM_THREADS=1 && \
    export NUMBA_THREADING_LAYER=workqueue && \
    python ${POINTNAV_VO_ROOT}/launch.py \
        --repo-path ${POINTNAV_VO_ROOT} \
        --n_gpus 1 \
        --task-type rl \
        --noise 1 \
        --run-type eval \
        --addr 127.0.1.1 \
        --port 8338

    date
    printf "JURE I MATE\n\n\n\n\n"
done

##~~~~~~~~~~~~~~~~~~~##
#######################
echo FINISHED at $(date)
