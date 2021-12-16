import os
from collections import namedtuple

from utils.util import ensure_dir


def fill_template(i, gpus, cpus, command, sbatch_id):
    return f"""#!/bin/bash
#SBATCH --chdir /home/rajic/xode/ruslan
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {cpus}
#SBATCH --mem 92160
#SBATCH --time 24:00:00
#SBATCH --account cs503
#SBATCH --reservation courses
#SBATCH --gres=gpu:{gpus}
#SBATCH -q gpu
#SBATCH -o ./slurm_logs/slurm-{sbatch_id}-{i}-%j.out

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
printf "Run configured and environment setup. Gonna run now.\\n\\n"
{command}
echo FINISHED at $(date)
"""


Task = namedtuple("Task", "agent_name agent_args dataset_split corruptions")
visual_corruptions = ["Defocus_Blur", "Lighting", "Speckle_Noise", "Spatter", "Motion_Blur"]

RUSLAN_NAME = "ruslan"
RUSLAN_ARGS = "agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on"

RANDOM_AGENT_NAME = "random_agent"
RANDOM_AGENT_ARGS = "random_agent.py --evaluation local"

sbatch1 = {
    "id": "sbatch_1",
    "configurations_to_run": [
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", ""),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", "--visual_corruption Defocus_Blur --visual_severity 5"),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", "--color_jitter"),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", "--random_shift"),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", "-hfov 50"),
            "cpus": 20,
            "gpus": 1
        },
        *[
            {
                "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", f"-pn_robot LoCoBot-Lite -pn_multiplier {pnm}"),
                "cpus": 20, "gpus": 1
            } for pnm in [0.0, 0.5]
        ],
        *[{"task":
               Task(RUSLAN_NAME, RUSLAN_ARGS, "val", f"--pyrobot_controller {pc} --pyrobot_noise_multiplier {pnm}"),
           "cpus": 20, "gpus": 1} for pc in ["Movebase", "ILQR"] for pnm in [0.0, 0.5]],
        *[{"task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", f"-pn_controller Proportional -pn_multiplier {pnm}"),
           "cpus": 20, "gpus": 1} for pnm in [0.0, 0.5, 1.0]],
        *[
            {
                "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", f"-vc {vc} -vs {vs} --pyrobot_noise_multiplier {pnm}"),
                "cpus": 20,
                "gpus": 1
            } for vc in visual_corruptions for pnm in [0.0, 0.5, 1.0] for vs in [3, 5]
        ],
    ]
}
sbatch2 = {
    "id": "sbatch_2",
    "configurations_to_run": [
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", "--color_jitter"),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", "--random_shift"),
            "cpus": 20,
            "gpus": 1
        },
        {
            "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", "-hfov 50"),
            "cpus": 20,
            "gpus": 1
        },
        *[
            {
                "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", f"-pn_robot LoCoBot-Lite -pn_multiplier {pnm}"),
                "cpus": 20, "gpus": 1
            } for pnm in [0.0, 0.5]
        ],
        # *[{"task":
        #        Task(RUSLAN_NAME, RUSLAN_ARGS, "train", f"--pyrobot_controller {pc} --pyrobot_noise_multiplier {pnm}"),
        #    "cpus": 20, "gpus": 1} for pc in ["Movebase", "ILQR"] for pnm in [0.0, 0.5]],
        # *[{"task": Task(RUSLAN_NAME, RUSLAN_ARGS, "val", f"-pn_controller Proportional -pn_multiplier {pnm}"),
        #    "cpus": 20, "gpus": 1} for pnm in [0.0, 0.5, 1.0]],
        *[
            {
                "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", f"-vc {vc} -vs {vs} --pyrobot_noise_multiplier {pnm}"),
                "cpus": 20,
                "gpus": 1
            } for vc in visual_corruptions for pnm in [0.5] for vs in [3, 5]
        ],
        # *[
        #     {
        #         "task": Task(RUSLAN_NAME, RUSLAN_ARGS, "train", f"-vc {vc} -vs {vs} --pyrobot_noise_multiplier {pnm}"),
        #         "cpus": 20,
        #         "gpus": 1
        #     } for vc in visual_corruptions for pnm in [0.0, 1.0] for vs in [3, 5]
        # ],
    ]
}


def sbatch_all_corruptions(sbatch_id, agent_name, agent_args, dataset_split, random_episodes=None):
    if random_episodes is not None:
        assert random_episodes > 0
        agent_args += f" --num_episodes {random_episodes} --num_episode_sample {random_episodes}"
    return {
        "id": sbatch_id,
        "configurations_to_run": [
            {
                "task": Task(agent_name, agent_args, dataset_split, "--color_jitter"),
                "cpus": 20,
                "gpus": 1
            },
            {
                "task": Task(agent_name, agent_args, dataset_split, "--random_shift"),
                "cpus": 20,
                "gpus": 1
            },
            {
                "task": Task(agent_name, agent_args, dataset_split, "-hfov 50"),
                "cpus": 20,
                "gpus": 1
            },
            *[
                {
                    "task": Task(agent_name, agent_args, dataset_split, f"-pn_robot LoCoBot-Lite -pn_multiplier {pnm}"),
                    "cpus": 20, "gpus": 1
                } for pnm in [0.0, 0.5]
            ],
            *[{"task":
                   Task(agent_name, agent_args, dataset_split,
                        f"--pyrobot_controller {pc} --pyrobot_noise_multiplier {pnm}"),
               "cpus": 20, "gpus": 1} for pc in ["Movebase", "ILQR"] for pnm in [0.0, 0.5]],
            *[{"task": Task(agent_name, agent_args, dataset_split, f"-pn_controller Proportional -pn_multiplier {pnm}"),
               "cpus": 20, "gpus": 1} for pnm in [0.0, 0.5, 1.0]],
            *[
                {
                    "task": Task(agent_name, agent_args, dataset_split,
                                 f"-vc {vc} -vs {vs} --pyrobot_noise_multiplier {pnm}"),
                    "cpus": 20,
                    "gpus": 1
                } for vc in visual_corruptions for pnm in [0.0, 0.5, 1.0] for vs in [3, 5]
            ],
        ]
    }


# SBATCH = sbatch1
# SBATCH = sbatch2
# SBATCH = sbatch_all_corruptions("sbatch_3", RANDOM_AGENT_NAME, RANDOM_AGENT_ARGS, "val")
SBATCH = sbatch_all_corruptions("sbatch_4", RUSLAN_NAME, RUSLAN_ARGS, "train", random_episodes=1000)
# SBATCH = sbatch_all_corruptions("sbatch_5", RANDOM_AGENT_NAME, RANDOM_AGENT_ARGS, "train", random_episodes=1000)
if __name__ == '__main__':
    OUTPUT_FOLDER = f"./{SBATCH['id']}"
    ensure_dir(OUTPUT_FOLDER)

    for i, config in enumerate(SBATCH["configurations_to_run"]):
        i += 1
        command = f'python {config["task"].agent_args}' \
                  f' --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml' \
                  f' --agent_name {config["task"].agent_name}' \
                  f' --dataset_split {config["task"].dataset_split}' \
                  f' --seed 72' \
                  f' {config["task"].corruptions}'

        script_path = os.path.join(OUTPUT_FOLDER, f"{SBATCH['id'].split('_')[-1]}-{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(
                fill_template(
                    i=i,
                    gpus=config["gpus"],
                    cpus=config["cpus"],
                    command=command,
                    sbatch_id=SBATCH["id"]
                ))
