import os
from collections import defaultdict
from typing import Dict, Optional, List

import imageio
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from util import ensure_dir, get_str_formatted_time


class MyBenchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
            self, config_paths: Optional[str] = None, eval_remote: bool = False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        print(config_env)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

        self.run_id = f"{config_env.RUN_ID}__{get_str_formatted_time()}"
        self.log_folder = os.path.join(config_env.LOG_FOLDER, self.run_id)
        self.video_log_interval = config_env.VIDEO_LOG_INTERVAL
        ensure_dir(self.log_folder)

    def remote_evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        raise NotImplementedError("No soup for you")

    def local_evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        metrics_to_log = ["distance_to_goal", "success", "spl", "softspl"]
        # print(f"self._env.task._config={self._env.task._config}")
        # print(f"self._env.sim.habitat_config={self._env.sim.habitat_config}")
        # print(f"self._env._config={self._env._config}")
        top_down_map_measure = TopDownMap(self._env.sim, self._env.task._config.TOP_DOWN_MAP)

        viridis_cmap = plt.get_cmap("viridis")
        normalizer = Normalize(vmin=None, vmax=None)
        # normalizer = Normalize(
        #     vmin=self._env._config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
        #     vmax=self._env._config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        # )

        # csv_log_line_buffer = []
        csv_log_file = os.path.join(self.log_folder, "episode_metrics.csv")
        with open(csv_log_file, "a") as csv_file:
            csv_file.write(','.join(["episode"] + metrics_to_log))
            csv_file.write("\n")

        pbar = tqdm.tqdm(total=num_episodes, )
        count_episodes = 0
        while count_episodes < num_episodes:
            log_video_for_episode = (count_episodes == num_episodes - 1) or (
                    count_episodes % self.video_log_interval) == 0

            agent.reset()
            observations = self._env.reset()
            if log_video_for_episode:
                top_down_map_measure.reset_metric(self._env.current_episode, self._env.task)

            logger.info("Agent stepping around inside environment.")
            images = []
            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

                if log_video_for_episode:
                    rgb = observations["rgb"]
                    if type(rgb) == torch.Tensor:
                        rgb = rgb.cpu().numpy().astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)

                    depth = observations["depth"]
                    if type(depth) == torch.Tensor:
                        depth = depth.cpu().numpy().squeeze(axis=-1)
                    else:
                        depth = depth.squeeze(axis=-1)
                    # depth = (depth * 255 / 10)
                    depth = normalizer(depth)
                    depth = viridis_cmap(depth, alpha=None, bytes=True)[:, :, :3]  # exclude 'a' from 'rgba'

                    top_down_map_measure.update_metric(self._env.current_episode, action)
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                        top_down_map_measure.get_metric(),  # self._env.get_metrics()["top_down_map"],
                        rgb.shape[0]
                    )
                    output_im = np.concatenate((rgb, depth, top_down_map), axis=1)
                    images.append(output_im)

            logger.info(f"Episode {count_episodes} finished")
            if log_video_for_episode:
                self.images_to_video(images, self.log_folder, f"{count_episodes:04d}")
            metrics = self._env.get_metrics()
            print(f"metrics for count_episodes={count_episodes}")
            csv_log_line = f"{count_episodes}"
            for m in metrics_to_log:
                v = metrics[m]
                agg_metrics[m] += v
                logger.info("{}: {}".format(m, v))
                csv_log_line += f",{v}"
            # csv_log_line_buffer.append(csv_log_line)
            with open(csv_log_file, "a") as csv_file:
                csv_file.write(f"{csv_log_line}\n")
                # for csv_log_line in csv_log_line_buffer:
                #     csv_file.write(f"{csv_log_line}\n")
                # csv_log_line_buffer = []

            pbar.update()
            count_episodes += 1

        avg_metrics = {k: agg_metrics[k] / count_episodes for k in metrics_to_log}

        return avg_metrics

    @staticmethod
    def images_to_video(
            images: List[np.ndarray],
            output_dir: str,
            video_name: str,
            fps: int = 10,
            quality: Optional[float] = 5,
            **kwargs,
    ):

        r"""Calls imageio to run FFMPEG on a list of images. For more info on
        parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
        Args:
            images: The list of images. Images should be HxWx3 in RGB order.
            output_dir: The folder to put the video in.
            video_name: The name for the video.
            fps: Frames per second for the video. Not all values work with FFMPEG,
                use at your own risk.
            quality: Default is 5. Uses variable bit rate. Highest quality is 10,
                lowest is 0.  Set to None to prevent variable bitrate flags to
                FFMPEG so you can manually specify them using output_params
                instead. Specifying a fixed bitrate using ‘bitrate’ disables
                this parameter.
        """
        assert 0 <= quality <= 10
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
        writer = imageio.get_writer(
            os.path.join(output_dir, video_name),
            fps=fps,
            quality=quality,
            **kwargs,
        )
        logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
        for im in images:
            writer.append_data(im)
        writer.close()

    def evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)


class MyChallenge(MyBenchmark):
    def __init__(self, eval_remote=False):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(config_paths, eval_remote=eval_remote)

    def submit(self, agent):
        metrics = super().evaluate(agent)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
