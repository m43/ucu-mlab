import ast
import os

import pandas as pd

from habitat import get_config
from utils.util import get_str_formatted_time, project_path

# logs_dir = "/home/user72/Desktop/logs/"
logs_dir = "logs"

ouput_csv = "logs/logs_3.csv"
timestamp = get_str_formatted_time()


def load_run(run_id, run_dir_path):
    episode_metrics_path = os.path.join(run_dir_path, "episode_metrics.csv")
    if not os.path.exists(episode_metrics_path):
        print(f"Skipping {run_dir} because no episode_metrics were found."
              f"\nAbsolute path to run:\n{os.path.abspath(run_dir_path)}")
        return None
    print(f"+1")

    episode_metrics = pd.read_csv(episode_metrics_path)

    with open(os.path.join(run_dir_path, "args.txt"), "r") as f:
        args = ast.literal_eval(f.read())

    # get_config wants a .yaml file, and I also need to convert hfov to int
    # copyfile(os.path.join(run_dir_path, "config.txt"), os.path.join(run_dir_path, "config.yaml"))
    with open(os.path.join(run_dir_path, "config.txt"), "r") as f:
        task_config_txt = f.read()
    task_config_txt = task_config_txt.replace("HFOV: 50.0", "HFOV: 50")  # hacky
    with open(os.path.join(run_dir_path, "config.yaml"), "w") as f:
        f.write(task_config_txt)
    task_config = get_config(os.path.join(run_dir_path, "config.yaml"))

    video_paths = [
        os.path.abspath(os.path.join(run_dir_path, v))
        for v in os.listdir(run_dir_path)
        if v.endswith("mp4")
    ]

    run = {
        "run_id": run_id,
        "args": args,
        "task_config": task_config,
        "episode_metrics": episode_metrics,
        "video_paths": video_paths,
    }
    return run


def get_corruptions_id(args_dict):
    viz_corruptions = []
    if args_dict["habitat_rgb_noise_intensity"] != 0.1:
        viz_corruptions.append(f"habitatrgbnoise={args_dict['pyrobot_noise_multiplier']}")
    if args_dict["visual_corruption"] and args_dict["visual_severity"] != 0:
        viz_corruptions.append(f"{args_dict['visual_corruption']}={args_dict['visual_severity']}")
    if args_dict['color_jitter']:
        viz_corruptions += ["colorjitter"]
    if args_dict['random_crop']:
        viz_corruptions += [f"radnomcrop={args_dict['crop_width']}x{args_dict['crop_height']}"]
    if args_dict['random_shift']:
        viz_corruptions += [f"randomshift"]
    if args_dict['habitat_rgb_hfov'] != 70:
        viz_corruptions += [f"hfov={args_dict['habitat_rgb_hfov']}"]

    dyn_corruptions = []
    if args_dict["pyrobot_controller_spec"] != "Proportional":
        dyn_corruptions.append(f"pc={args_dict['pyrobot_controller_spec']}")
    if args_dict["pyrobot_robot_spec"] != "LoCoBot":
        dyn_corruptions.append(f"pr={args_dict['pyrobot_robot_spec']}")
    if args_dict["pyrobot_noise_multiplier"] != 0.5:
        dyn_corruptions.append(f"pnm={args_dict['pyrobot_noise_multiplier']}")

    return "+".join(viz_corruptions + dyn_corruptions)


all_results = pd.DataFrame(columns={
    "run_id", "videos", "agent_name", "dataset_split", "episodes", "corruptions",
    "episode", "spl", "softspl", "distance_to_goal", "success"
})

for agent_dir in os.listdir(logs_dir):
    agent_dir_path = os.path.join(logs_dir, agent_dir)
    if not os.path.isdir(agent_dir_path):
        continue
    for corruption_dir in os.listdir(agent_dir_path):
        corruption_dir_path = os.path.join(agent_dir_path, corruption_dir)
        if not os.path.isdir(corruption_dir_path):
            continue
        for run_dir in os.listdir(corruption_dir_path):
            run_dir_path = os.path.join(corruption_dir_path, run_dir)
            if not os.path.isdir(run_dir_path):
                continue

            run = load_run(run_dir, run_dir_path)
            if run is None:
                continue
            run["episode_metrics"]["run_id"] = run["run_id"]
            run["episode_metrics"]["episodes"] = len(run["episode_metrics"])
            run["episode_metrics"]["videos"] = len(run["video_paths"])
            run["episode_metrics"]["agent_name"] = run["args"]["agent_name"]
            run["episode_metrics"]["dataset_split"] = run["args"]["dataset_split"]
            run["episode_metrics"]["dataset_split"] = run["args"]["dataset_split"]
            run["episode_metrics"]["corruptions"] = get_corruptions_id(run["args"])

            all_results = pd.concat([all_results, run["episode_metrics"]])

results = all_results.copy()
results = results.groupby(["run_id", "videos", "agent_name", "dataset_split", "corruptions", "episodes"],
                          as_index=False).mean()
results = results.sort_values(["agent_name", "dataset_split", "corruptions", "episodes", "videos"])
results.set_index("run_id")

logs_csv_path_1 = os.path.abspath(ouput_csv)
results.to_csv(logs_csv_path_1)

logs_csv_path_2 = os.path.join(project_path, f"logs_{timestamp}.csv")
results.to_csv(logs_csv_path_2)

print(results)
print(logs_csv_path_1)
print(logs_csv_path_2)
