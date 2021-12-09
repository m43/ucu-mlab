import argparse


def get_corruptions_parser():
    parser = argparse.ArgumentParser(
        description="corruptions", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-hni",
        "--habitat_rgb_noise_intensity",
        default=0.1,
        type=float,
        required=False,
        help="Intensity of RGB noise introduced by habitat (in the 2021 challenge it was GaussianNoiseModel with "
             "intensity 0.1). This allows the noise to be disabled by setting the intensity to 0.0, or reinforcing "
             "its magnitute by making the intensity higher.",
    )

    parser.add_argument(
        "-hfov",
        "--habitat_rgb_hfov",
        default=70,
        type=float,
        required=False,
        help="Habitat RGB sensor horizontal field of view (in the 2021 challenge, the default was 70)",
    )

    # Defocus_Blur Lighting Speckle_Noise Spatter Motion_Blur
    parser.add_argument(
        "-vc",
        "--visual_corruption",
        default=None,
        type=str,
        required=False,
        help="Visual corruption to be applied to egocentric RGB observation",
    )

    parser.add_argument(
        "-vs",
        "--visual_severity",
        default=0,
        type=int,
        required=False,
        help="Severity of visual corruption to be applied",
    )

    # parser.add_argument(
    #     "-dcr",
    #     "--dyn_corr_mode",
    #     dest="dyn_corr_mode",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply dynamics corruptions",
    # )
    # parser.set_defaults(dyn_corr_mode=False)
    #
    # parser.add_argument(
    #     "-mf",
    #     "--motor_failure",
    #     dest="motor_failure",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply motor failure as the dynamics corruption",
    # )
    # parser.set_defaults(motor_failure=False)
    #
    # parser.add_argument(
    #     "-ctr",
    #     "--const_translate",
    #     dest="const_translate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply constant translation bias as the dynamics corruption",
    # )
    # parser.set_defaults(const_translate=False)
    #
    # parser.add_argument(
    #     "-crt",
    #     "--const_rotate",
    #     dest="const_rotate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply constant rotation bias as the dynamics corruption",
    # )
    # parser.set_defaults(const_rotate=False)
    #
    # parser.add_argument(
    #     "-str",
    #     "--stoch_translate",
    #     dest="stoch_translate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply stochastic translation bias as the dynamics corruption",
    # )
    # parser.set_defaults(stoch_translate=False)
    #
    # parser.add_argument(
    #     "-srt",
    #     "--stoch_rotate",
    #     dest="stoch_rotate",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply stochastic rotation bias as the dynamics corruption",
    # )
    # parser.set_defaults(stoch_rotate=False)
    #
    # parser.add_argument(
    #     "-dr",
    #     "--drift",
    #     dest="drift",
    #     required=False,
    #     action="store_true",
    #     help="Whether to apply drift in translation as the dynamics corruption",
    # )
    # parser.set_defaults(drift=False)
    #
    # parser.add_argument(
    #     "-dr_deg",
    #     "--drift_degrees",
    #     default=1.15,
    #     type=float,
    #     required=False,
    #     help="Drift angle for the motion-drift dynamics corruption",
    # )

    parser.add_argument(
        "-irc",
        "--random_crop",
        dest="random_crop",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.add_argument(
        "-cw",
        "--crop_width",
        type=int,
        required=False,
        help="Specify if random crop width is to be applied to the egocentric observations",
    )
    parser.add_argument(
        "-ch",
        "--crop_height",
        type=int,
        required=False,
        help="Specify if random crop height is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_crop=False)

    parser.add_argument(
        "-icj",
        "--color_jitter",
        dest="color_jitter",
        required=False,
        action="store_true",
        help="Specify if random crop is to be applied to the egocentric observations",
    )
    parser.set_defaults(color_jitter=False)

    parser.add_argument(
        "-irs",
        "--random_shift",
        dest="random_shift",
        required=False,
        action="store_true",
        help="Specify if random shift is to be applied to the egocentric observations",
    )
    parser.set_defaults(random_shift=False)

    # LoCoBot, LoCoBot-Lite
    parser.add_argument(
        "-pn_robot",
        "--pyrobot_robot_spec",
        default="LoCoBot",
        type=str,
        required=False,
        help="Which robot specification to use for PyRobot (LoCoBot, LoCoBot-Lite)",
    )

    # ILQR, Proportional, ILQR
    parser.add_argument(
        "-pn_controller",
        "--pyrobot_controller_spec",
        default="Proportional",
        type=str,
        required=False,
        help="Which PyRobot controller specification to use (ILQR, Proportional, ILQR)",
    )

    parser.add_argument(
        "-pn_multiplier",
        "--pyrobot_noise_multiplier",
        default=0.5,
        type=float,
        required=False,
        help="PyRobot noise magnitude multiplier",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "-ne",
        "--num_episodes",
        default=None,
        type=int,
        required=False,
        help="Number of episodes to run the benchmark for, or None.",
    )
    parser.add_argument(
        "-config",
        "--challenge_config_file",
        default=None,
        type=str,
        required=False,
        help="Habitat config that, if specified, overwrites the environmental variable CHALLENGE_CONFIG_FILE",
    )
    parser.add_argument(
        "-an",
        "--agent_name",
        default=None,
        type=str,
        required=True,
        help="Agent name, used for loging",
    )
    parser.add_argument(
        "-ds",
        "--dataset_split",
        default=None,
        type=str,
        required=True,
        help="Which dataset split to use (train, val, val_mini)",
    )
    return parser


def apply_corruptions_to_config(args, task_config):
    task_config.defrost()
    task_config.RANDOM_SEED = args.seed
    task_config.SIMULATOR.RGB_SENSOR.HFOV = args.habitat_rgb_hfov
    task_config.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = args.habitat_rgb_noise_intensity
    task_config.SIMULATOR.NOISE_MODEL.ROBOT = args.pyrobot_robot_spec
    task_config.SIMULATOR.NOISE_MODEL.CONTROLLER = args.pyrobot_controller_spec
    task_config.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = args.pyrobot_noise_multiplier
    task_config.freeze()
