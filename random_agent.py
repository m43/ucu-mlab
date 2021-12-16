import os

import numpy

import habitat
from corruptions.parser import get_corruptions_parser, apply_corruptions_to_config, get_runid_and_logfolder
from habitat_extensions.sensors.noise_models.gaussian_noise_model_torch import GaussianNoiseModelTorch
from my_benchmark import MyChallenge


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config, steps_to_make_percentage=1.0):
        assert 0.0 <= steps_to_make_percentage <= 1.0
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS[1:]
        self._STOP_ACTION = task_config.TASK.POSSIBLE_ACTIONS[0]
        self._STEP_COUNT = 0
        self._STEPS_TO_MAKE = int(task_config.ENVIRONMENT.MAX_EPISODE_STEPS * steps_to_make_percentage)

    def reset(self):
        self._STEP_COUNT = 0

    def act(self, observations):
        self._STEP_COUNT += 1
        if self._STEP_COUNT == self._STEPS_TO_MAKE:
            action = {"action": self._STOP_ACTION}
        else:
            action = {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}
        return action


def main():
    _ = GaussianNoiseModelTorch()
    parser = get_corruptions_parser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()

    if args.challenge_config_file:
        config_paths = args.challenge_config_file
    else:
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    task_config = habitat.get_config(config_paths)
    apply_corruptions_to_config(args, task_config)
    args.run_id, args.log_folder = get_runid_and_logfolder(args, task_config)

    # agent = RandomAgent(task_config=task_config, steps_to_make_percentage=0.125)
    agent = RandomAgent(task_config=task_config)

    if args.evaluation == "local":
        challenge = MyChallenge(task_config, eval_remote=False, **args.__dict__)
        challenge._env.seed(task_config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent, args.num_episodes)


if __name__ == "__main__":
    main()
