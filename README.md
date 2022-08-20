# Robustness of Embodied Point Navigation Agents: UCU Mlab agent

[Frano Rajic](https://m43.github.io/)

[`Project Website`](https://m43.github.io/projects/embodied-ai-robustness/) | [`Paper`](https://www.youtube.com/watch?v=dQw4w9WgXcQ) | [**`>> Code [UCU Mlab] <<`**](https://github.com/m43/ucu-mlab) | [`Code [VO2021]`](https://github.com/m43/vo2021)

This repository contains the evaluation code for reproducing the benchmark results for the UCU Mlab agent and the baselines. The codebase of the UCU Mlab agent is taken from [rpartsey/pointgoal-navigation](https://github.com/rpartsey/pointgoal-navigation) at the version that was submitted to the [Habitat Challenge 2021](https://aihabitat.org/challenge/2021/).

## Set-up

Start by cloning the repository:
```bash
git clone https://github.com/m43/ucu-mlab.git
cd ucu-mlab
```

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment using the provided environment setup script, adapted for our local machine setup. Depending on your local machine, you might want to remove the `module purge` and `module load ...` lines in the script, since we used the to prepare the cluster machines we worked with. The script might take a long time to run as `habitat-sim` must be built. The script will print out verbose logs about what bash command was run (`set -o xtrace`) and will stop if an error is encountered (`set -e`). To run the environment setup script:
```bash
bash scripts/setup_cluster_conda_gpu.sh
```

The environment setup script will automatically download the Gibson dataset split data (a bunch of compressed `.json` files defining the splits of the Gibson dataset) with `gdown`. However, the Gibson dataset (~10GB) itself needs to be downloaded separately. To download (and link) the Gibson dataset, you could do the following:
```bash
# Optionally: create (or move) to a folder where you usually store datasets
mkdir -p /home/frano/data
cd /home/frano/data

# Sign agreement and download gibson_habitat_trainval.zip: https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform
# wget link/to/gibson_habitat_trainval.zip
unzip gibson_habitat_trainval.zip

cd /get/back/to/the/cloned/code/repo # cd -
mkdir -p ./dataset
mkdir -p ./data

# Link the Gibson dataset correctly
ln -s /home/frano/data data/scene_datasets

# Verify that everything was linked correctly:
tree -L 1 /home/frano/data/gibson/
# /home/frano/data/gibson/
# ├── Ackermanville.glb
# ├── Ackermanville.navmesh
# ├── Adairsville.glb
# ├── Adairsville.navmesh
# ├── Adrian.glb
# ├── Adrian.navmesh
# ├── Airport.glb
# ├── Airport.navmesh
# ├── Albertville.glb
# ...
# └── Yscloskey.navmesh
tree data/ -L 5
data/
# ├── datasets
# │   └── pointnav
# │       └── gibson
# │           ├── gibson_quality_ratings.csv
# │           └── v2
# │               ├── train
# │               ├── val
# │               └── val_mini
# └── scene_datasets -> /home/frano/data/
```

Finally, download the UCU Mlab agent checkpoints from [this link](https://drive.google.com/drive/folders/1TLniS5FxwDxvoDWiAsshohY3rZDF2WVz?usp=sharing) to `saved`. For the DD-PPO agent, download the checkpoint from [this link](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth) and place it into `saved`. For example:
``` bash
gdown --folder 1TLniS5FxwDxvoDWiAsshohY3rZDF2WVz --output saved
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth -O saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth

tree saved/
# saved/
# ├── best_checkpoint_064e.pt                                # UCU Mlab: The visual odometry checkpoint
# ├── config.yaml                                            # UCU Mlab: The visual odometry config
# ├── ddppo_pointnav_habitat2021_challenge_baseline_v1.pth   # DD-PPO baseline checkpoint, provided by Habitat
# └── pointnav2021_gt_loc_depth_ckpt.345.pth                 # UCU Mlab: The navigation policy checkpoint
```

## Results reproduction

Activate the created environment, and then follow the instructions for specific agents:
```bash
# module purge
# module load gcc/8.4.0-cuda cuda/10.1
conda activate ruslan
```

### UCU Mlab agent

To reproduce the Color Jitter visual corruption results on the validation subset (row 13 of Table 1 of the paper), run the following:
```bash
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name ruslan --dataset_split val --color_jitter
```

This run configuration can be found in `slurm/sbatch_1/03.sh`. For other run configurations, consult the set of SLURM scripts in `slurm/sbatch_1`, `slurm/sbatch_2`, `slurm/sbatch_4`, and `slurm_10`. Alternatively, consult the `eval.sh` script to see how all the possible corruption settings can be run.

### Random agent

To reproduce the Color Jitter visual corruption results (row 13 of Table 1 of the paper), run the following:
```bash
python random_agent.py --evaluation local --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name random_agent --dataset_split val --color_jitter
```

This run configuraiton can be found in `slurm/sbatch_3/3-01.sh`. For other run configurations, consult the set of SLURM scripts in `slurm/sbatch_3` and `slurm/sbatch_10`.

### DD-PPO agent

To reproduce the Color Jitter visual corruption results (row 13 of Table 1 of the paper), run the following:
```bash
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name ddppo --dataset_split val --seed 72 --color_jitter
```

This run configuration can be found in `slurm/sbatch_6/6-01.sh`. For other run configurations, consult the set of SLURM scripts in `slurm/sbatch_6` and `slurm/sbatch_10`.

## Citing
If you find our work useful, please consider citing:
```BibTeX
[WIP]: Will be added once the Proceedings of the ECCV 2022 Workshops are published.
```
