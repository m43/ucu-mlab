import argparse
import os

import numpy

import habitat
from habitat_extensions.sensors.noise_models.gaussian_noise_model_torch import GaussianNoiseModelTorch
from my_benchmark import MyChallenge

class RandomAgentV1(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}

class RandomAgentV2(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS[1:]
        self._STOP_ACTION = task_config.TASK.POSSIBLE_ACTIONS[0]
        self._STEP_COUNT = 0
        self._MAX_STEPS = task_config.ENVIRONMENT.MAX_EPISODE_STEPS

    def reset(self):
        self._STEP_COUNT = 0

    def act(self, observations):
        self._STEP_COUNT += 1
        if self._STEP_COUNT == self._MAX_STEPS:
            action = {"action": self._STOP_ACTION}
        else:
            action = {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}
        return action


def main():
    _ = GaussianNoiseModelTorch()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.RANDOM_SEED = args.seed
    config.freeze()

    agent = RandomAgentV1(task_config=config)
    # agent = RandomAgentV2(task_config=config)

    if args.evaluation == "local":
        challenge = MyChallenge(eval_remote=False)
        challenge._env.seed(config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
