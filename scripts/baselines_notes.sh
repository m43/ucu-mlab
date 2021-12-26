#########
# NOTES #
############################
# It's not really a script #
############################
exit

#########################################################
################ HABITAT-LAB BASELINES ##################
#########################################################

## Hello world - does the Habitat environment work?
cd /home/rajic/xode/habitat-lab

python examples/example.py


## Train + eval of PPO nn model for pointnav
cd /home/rajic/xode/habitat-lab

python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train

python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type eval


## Habitat-lab baselines
cd /home/rajic/xode/habitat-lab

wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/habitat_baselines_v2.zip
unzip habitat_baselines_v2.zip
python -u habitat_baselines/agents/ppo_agents.py --model-path gibson-rgbd-best.pth --task-config configs/tasks/pointnav_rgbd.yaml --input-type rgbd

python -u -m habitat_baselines.run \
    --exp-config ../habitat-challenge/configs/ddppo_pointnav.yaml \
    --run-type eval \
    EVAL_CKPT_PATH_DIR $PATH_TO_CHECKPOINT \
    TASK_CONFIG.DATASET.SPLIT val


################################################################
############### Habitat Challenge 2021 baselines ###############
################################################################
cd /home/rajic/xode/habitat-challenge

wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth
mkdir checkpoints
mv ddppo_pointnav_habitat2021_challenge_baseline_v1.pth checkpoints/

vim configs/ddppo_pointnav.yaml
# BASE_TASK_CONFIG_PATH: "configs/challenge_objectnav2021.local.rgbd.yaml" #configs/challenge_pointnav2021.local.rgbd.yaml"
# -->
# BASE_TASK_CONFIG_PATH: "configs/challenge_pointnav2021.local.rgbd.yaml"

vim configs/challenge_pointnav2021.local.rgbd.yaml
# DATASET:
#   TYPE: PointNav-v1
#   SPLIT: val_mini
#   DATA_PATH: habitat-challenge-data/pointgoal_gibson_v2/{split}/{split}.json.gz
#   SCENES_DIR: "habitat-challenge-data/data/scene_datasets/"
# -->
# DATASET:
#   TYPE: PointNav-v1
#   SPLIT: val_mini
#   DATA_PATH: data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz
#   SCENES_DIR: data/scene_datasets/
mkdir data
ln -s /home/rajic/xode/ruslan/data/datasets data/datasets
ln -s /home/rajic/xode/ruslan/data/scene_datasets data/scene_datasets

#to run on val_mini
vim configs/rl/ddppo_pointnav.yaml
# EVAL.SPLIT
# val --> val_mini

# Btw, I think that this is the blind agent, below is with --model-path checkpoint.pth
python -u -m habitat_baselines.run \
    --exp-config ../habitat-challenge/configs/ddppo_pointnav.yaml \
    --run-type eval \
    EVAL_CKPT_PATH_DIR checkpoints/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth

# Can we run this without the trainer stuff?
# Can we have the agent+benchmark setup that Ruslan used?
# Yes! It's needed for submitting the code! Awesome!

export CHALLENGE_CONFIG_FILE=/home/rajic/xode/ruslan/config_files/challenge_pointnav2021.local.rgbd.CPU.yaml
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path checkpoints/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth

################################################################
############### Plug Baseline into Ruslan's code ###############
################################################################
# copy ddppo_agents.py
# adapt ddppo_agents.py to use our benchmark and parser
# copy the ddppo_pointnav.yaml config

# get the agent
wget --output-document saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth  https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth

# clean settings
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth --agent_name ddppo --dataset_split val_mini

# lets add some noise
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth --agent_name ddppo --dataset_split val_mini

########################
#########  VO  #########
########################

export CUDA_LAUNCH_BLOCKING=1 &&
export PYTHONPATH=/scratch/izar/rajic/vo:$PYTHONPATH &&
python -u -m torch.distributed.launch
 --nproc_per_node=1
 --master_addr 127.0.1.1
 --master_port 8338
 --use_env
 /scratch/izar/rajic/vo/pointnav_vo/run.py
 --task-type rl
 --noise 1
 --exp-config /scratch/izar/rajic/vo/configs/rl/ddppo_pointnav.yaml
 --run-type eval
 --n-gpu 1
 --cur-time 20211214_222538319058


# export PYTHONPATH=/scratch/izar/rajic/vo:$PYTHONPATH &&
# python /scratch/izar/rajic/vo/pointnav_vo/run.py
#  --task-type rl
#  --noise 1
#  --exp-config /scratch/izar/rajic/vo/configs/rl/ddppo_pointnav.yaml
#  --run-type eval
#  --n-gpu 1
#  --cur-time 20211214_222538319058


export PYTHONPATH=/scratch/izar/rajic/vo:$PYTHONPATH && pointnav_vo/run.py --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 20211214_222538319058

#to run on val_mini
vim configs/rl/ddppo_pointnav.yaml
# EVAL.SPLIT
# val --> val_mini