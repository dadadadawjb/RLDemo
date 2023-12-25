import argparse


def config_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', 
                        choices=['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4', 
                                 'Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Ant-v2'], 
                        help='name of the environment to run')
    parser.add_argument('--improve', action='store_true', help='whether use improved algorithm')
    parser.add_argument('--test', action='store_true', help='whether test')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run')
    parser.add_argument('--debug', action='store_true', help='whether show some debug info')
    parser.add_argument('--gui', action='store_true', help='whether show gui')
    args = parser.parse_args()
    return args
