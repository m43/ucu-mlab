invalid_runs_ids = [
    "2021.12.19_03.09.57.751658__ddppo__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=0.0",
    "2021.12.19_03.09.57.969529__ddppo__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=1.5",
    "2021.12.19_03.09.57.819957__ddppo__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=2.0",
    "2021.12.16_03.22.54.007379__ddppo__val_mini____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.16_02.57.07.263570__ddppo__val_mini____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.16_03.23.07.328630__ddppo__val_mini__+colorjitter__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.16_20.15.36.412036__ddppo__val_mini__+colorjitter__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.16_03.20.16.911154__ddppo__val_mini____pc=Proportional_pr=LoCoBot_pnm=0.0__habitatrgbnoise=0.0",
    "2021.12.19_03.09.58.360729__random_agent__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=0.0",
    "2021.12.19_03.09.58.090556__random_agent__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=1.5",
    "2021.12.19_03.09.58.626903__random_agent__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=2.0",
    "2021.12.14_04.35.14.126583__random_agent__val___Spatter=3__pc=Proportional_pr=LoCoBot_pnm=1.0__habitatrgbnoise=0.1",
    "2021.12.14_03.35.42.249381__random_agent__val___Spatter=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.14_04.35.39.114404__random_agent__val___Spatter=5__pc=Proportional_pr=LoCoBot_pnm=1.0__habitatrgbnoise=0.1",
    "2021.12.14_01.58.52.411895__random_agent__val___Speckle_Noise=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.14_01.27.35.223443__random_agent__val___Speckle_Noise=3__pc=Proportional_pr=LoCoBot_pnm=0.0__habitatrgbnoise=0.1",
    "2021.12.14_03.09.14.309838__random_agent__val___Speckle_Noise=3__pc=Proportional_pr=LoCoBot_pnm=1.0__habitatrgbnoise=0.1",
    "2021.12.14_02.08.52.556139__random_agent__val___Speckle_Noise=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.14_01.36.33.946068__random_agent__val___Speckle_Noise=5__pc=Proportional_pr=LoCoBot_pnm=0.0__habitatrgbnoise=0.1",
    "2021.12.14_03.09.14.309838__random_agent__val___Speckle_Noise=5__pc=Proportional_pr=LoCoBot_pnm=1.0__habitatrgbnoise=0.1",
    "2021.12.13_02.47.28.728206__ruslan__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.10.14.061425__ruslan__train___Defocus_Blur=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.14_06.08.48.327693__ruslan__train___Defocus_Blur=3__pc=Proportional_pr=LoCoBot_pnm=0.0__habitatrgbnoise=0.1",
    "2021.12.13_03.17.17.871056__ruslan__train___Defocus_Blur=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_02.47.28.728059__ruslan__train___Defocus_Blur=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.17.19.115937__ruslan__train___Lighting=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.18.24.568746__ruslan__train___Lighting=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.19.23.731206__ruslan__train___Motion_Blur=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.19.01.488706__ruslan__train___Spatter=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.19.12.120277__ruslan__train___Spatter=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.18.32.647278__ruslan__train___Speckle_Noise=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.19.01.489136__ruslan__train___Speckle_Noise=5__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.00.11.183495__ruslan__train__+colorjitter__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.19_03.09.58.392035__ruslan__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=0.0",
    "2021.12.19_03.09.58.122676__ruslan__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=1.5",
    "2021.12.19_03.09.58.658153__ruslan__train____pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1_depthnoise=2.0",
    "2021.12.14_13.37.28.547662__ruslan__train__+hfov=50.0__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.02.52.928661__ruslan__train__+hfov=50.0__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.05.57.217655__ruslan__train____pc=Proportional_pr=LoCoBot-Lite_pnm=0.0__habitatrgbnoise=0.1",
    "2021.12.13_03.00.12.188226__ruslan__train__+randomshift__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.09.03.045064__ruslan__train____pc=Proportional_pr=LoCoBot-Lite_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.18_13.22.08.352936__vo__val__+hfov=50__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.13_03.19.24.272630__ruslan__train___Motion_Blur=3__pc=Proportional_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "2021.12.18_13.55.04.730792__vo__val____pc=Movebase_pr=LoCoBot_pnm=0.5__habitatrgbnoise=0.1",
    "",
    "",
    "",
    "",
]

import ast
import os

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

from habitat import get_config
from utils.util import get_str_formatted_time, project_path

# logs_dir = "/home/user72/Desktop/logs/logs"
logs_dir = "/home/user72/Desktop/logs/logs_11"
# logs_dir = "logs"

ouput_csv = "logs/logs_11.csv"
timestamp = get_str_formatted_time()


def load_run(run_id, run_dir_path, vo_format=False):
    if vo_format:
        episode_metrics = pd.DataFrame()
        for metric_name, metric_dir in [
            ("distance_to_goal", "tb/eval_metrics_distance_to_goal"),
            ("softspl", "tb/eval_metrics_softspl"),
            ("spl", "tb/eval_metrics_spl"),
            # ("length", "tb/eval_metrics_len"),
            ("success", "tb/eval_metrics_success"),
        ]:
            metric_path = os.path.join(run_dir_path, metric_dir)
            if not os.path.exists(metric_path):
                print(f"Skipping {run_dir} because no {metric_dir} was found."
                      f"\nAbsolute path to run:\n{os.path.abspath(run_dir_path)}")
                return None
            assert len(os.listdir(metric_path)) == 1
            event_path = os.path.join(metric_path, os.listdir(metric_path)[0])
            event_value = float(list(summary_iterator(event_path))[1].summary.value[0].simple_value)
            episode_metrics[metric_name] = [event_value]
        print(f"+1")
    else:
        episode_metrics_path = os.path.join(run_dir_path, "episode_metrics.csv")
        if not os.path.exists(episode_metrics_path):
            print(f"Skipping {run_dir} because no episode_metrics were found."
                  f"\nAbsolute path to run:\n{os.path.abspath(run_dir_path)}")
            return None
        episode_metrics = pd.read_csv(episode_metrics_path)
        print(f"+1")

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
        viz_corruptions.append(f"HC Noise = {args_dict['pyrobot_noise_multiplier']}")
    if "depth_noise_multiplier" in args_dict and args_dict["depth_noise_multiplier"] != 1.0:
        viz_corruptions.append(f"Depth Noise = {args_dict['depth_noise_multiplier']}")
    if args_dict["visual_corruption"] and args_dict["visual_severity"] != 0:
        viz_corruptions.append(f"{args_dict['visual_corruption'].replace('_', ' ')} (S={args_dict['visual_severity']})")
    if args_dict['color_jitter']:
        viz_corruptions += ["Color Jitter"]
    if args_dict['random_crop']:
        viz_corruptions += [f"Random Crop ({args_dict['crop_width']}x{args_dict['crop_height']})"]
    if args_dict['random_shift']:
        viz_corruptions += [f"Random Shift"]
    if args_dict['habitat_rgb_hfov'] != 70:
        viz_corruptions += [f"Lower HFOV ({int(args_dict['habitat_rgb_hfov'])}°)"]

    dyn_corruptions = []
    if args_dict["pyrobot_controller_spec"] != "Proportional":
        dyn_corruptions.append(f"controller={args_dict['pyrobot_controller_spec']}")
    if args_dict["pyrobot_robot_spec"] != "LoCoBot":
        dyn_corruptions.append(f"robot={args_dict['pyrobot_robot_spec']}")
    if args_dict["pyrobot_noise_multiplier"] != 0.5:
        dyn_corruptions.append(f"noise_multiplier={args_dict['pyrobot_noise_multiplier']}")

    if viz_corruptions:
        viz_corruptions_str = " ".join(viz_corruptions)
    if dyn_corruptions:
        dyn_corruptions_str = f"PyRobot ({' '.join(dyn_corruptions)})"

    if len(viz_corruptions) == 0 and len(dyn_corruptions) == 0:
        cid = "[Clean]"
    elif len(viz_corruptions) > 0 and len(dyn_corruptions) == 0:
        cid = f"[VIZ] {viz_corruptions_str}"
    elif len(viz_corruptions) == 0 and len(dyn_corruptions) > 0:
        cid = f"[DYN] {dyn_corruptions_str}"
    else:
        cid = f"[VIZ+DYN] {viz_corruptions_str} + {dyn_corruptions_str}"

    return cid


def get_short_corruptions_id(args_dict):
    viz_corruptions = []
    if args_dict["habitat_rgb_noise_intensity"] != 0.1:
        viz_corruptions.append(f"HCN={args_dict['pyrobot_noise_multiplier']}")
    if "depth_noise_multiplier" in args_dict and args_dict["depth_noise_multiplier"] != 1.0:
        viz_corruptions.append(f"DN={args_dict['depth_noise_multiplier']}")
    if args_dict["visual_corruption"] and args_dict["visual_severity"] != 0:
        vc_str = "".join([f"{s[0]}" for s in args_dict['visual_corruption'].split('_')])
        viz_corruptions.append(f"{vc_str}({args_dict['visual_severity']})")
    if args_dict['color_jitter']:
        viz_corruptions += ["CJ"]
    if args_dict['random_crop']:
        viz_corruptions += [f"RC"]
    if args_dict['random_shift']:
        viz_corruptions += [f"RS"]
    if args_dict['habitat_rgb_hfov'] != 70:
        viz_corruptions += [f"HFOV"]

    if viz_corruptions:
        viz_corruptions_str = ",".join(viz_corruptions)

    dyn_corruptions = []
    if args_dict["pyrobot_controller_spec"] != "Proportional":
        pc_str = args_dict['pyrobot_controller_spec'] if args_dict["pyrobot_controller_spec"] == "ILQR" else "MB"
        dyn_corruptions.append(f"PC={pc_str}")
    if args_dict["pyrobot_robot_spec"] != "LoCoBot":
        dyn_corruptions.append(f"PR=Lite")
    if args_dict["pyrobot_noise_multiplier"] != 0.5:
        dyn_corruptions.append(f"PNM={args_dict['pyrobot_noise_multiplier']}")

    if dyn_corruptions:
        dyn_corruptions_str = ','.join(dyn_corruptions)

    if len(viz_corruptions) == 0 and len(dyn_corruptions) == 0:
        cid = "[Clean]"
    elif len(viz_corruptions) > 0 and len(dyn_corruptions) == 0:
        cid = f"[V] {viz_corruptions_str}"
    elif len(viz_corruptions) == 0 and len(dyn_corruptions) > 0:
        cid = f"[D] {dyn_corruptions_str}"
    else:
        cid = f"[VD] {dyn_corruptions_str}+{viz_corruptions_str}"

    return cid


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

            run = load_run(run_dir, run_dir_path, "vo" in agent_dir)
            if run is None:
                print("Run was None, continue...")
                continue
            run["episode_metrics"]["run_id"] = run["run_id"]
            run["episode_metrics"]["episodes"] = len(run["episode_metrics"])
            run["episode_metrics"]["videos"] = len(run["video_paths"])
            run["episode_metrics"]["Agent"] = run["args"]["agent_name"]
            run["episode_metrics"]["Dataset Split"] = run["args"]["dataset_split"]
            run["episode_metrics"]["Corruptions (Long)"] = get_corruptions_id(run["args"])
            run["episode_metrics"]["Corruptions"] = get_short_corruptions_id(run["args"])

            all_results = pd.concat([all_results, run["episode_metrics"]])

all_results = all_results[~all_results.run_id.isin(invalid_runs_ids)]

results = all_results.copy()
results = results.groupby(
    ["run_id", "videos", "Agent", "Dataset Split", "Corruptions (Long)", "episodes", "Corruptions"],
    as_index=False).mean()
results = results.sort_values(["Agent", "Dataset Split", "Corruptions", "episodes", "videos"])
results.set_index("run_id")

results = results[
    ["run_id", "episodes", "videos", "Agent", "Dataset Split", "Corruptions",
     "success", "distance_to_goal", "softspl", "spl", "Corruptions (Long)"
     ]]

logs_csv_path_1 = os.path.abspath(ouput_csv)
results.to_csv(logs_csv_path_1)

logs_csv_path_2 = os.path.join(project_path, f"logs/logs_{timestamp}.csv")
results.to_csv(logs_csv_path_2)

print(results)
print(logs_csv_path_1)
print(logs_csv_path_2)
