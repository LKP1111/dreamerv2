import argparse
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym, random
from IFactor.utils.wrapper import OneHotAction
from myenv.robodesk.robodesk import RoboDesk, RoboDeskWithTV
from IFactor.utils.wrapper import RoboDeskImageWrapper
from IFactor.training.config import RoboDeskConfig
from IFactor.training.evaluator import Evaluator


def main(args):
    env_name = 'robodesk'
    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}'.format(env_name), '{}'.format(exp_id))
    model_dir = os.path.join(result_dir, 'models')
    gif_dir = os.path.join(result_dir, 'visualization')
    # dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')

    print('using :', device)

    env = RoboDeskWithTV(
        task="tv_green_hue",
        action_repeat=2,
        episode_length=1000,
        distractors='all',
        tv_video_file_pattern=os.path.expanduser("~/.driving_car/*.mp4")
    )
    env.seed(args.seed)
    env = RoboDeskImageWrapper(env)
    test_env = RoboDeskWithTV(
        task="tv_green_hue",
        action_repeat=2,
        episode_length=1000,
        distractors='all',
        tv_video_file_pattern=os.path.expanduser("~/.driving_car/*.mp4")
    )
    test_env.seed(args.seed + 1)
    test_env = RoboDeskImageWrapper(test_env)
    # eval_env = RoboDesk()
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    print(obs_shape, action_size)

    config = RoboDeskConfig(
        env=env_name,
        seed=args.seed,
        obs_shape=obs_shape,
        action_size=action_size,
        model_dir=model_dir,
        gif_dir=gif_dir,
    )

    config_dict = config.__dict__
    # gpu_tracker.track()
    # trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)
    best_score = 0
    best_model = None
    model_name = os.path.join(model_dir, 'models_best/models_best.pth')
    # model_name = '/home/liuyr/thl/interprl/results/robodesk/21/models/models_1000000/models_1000000.pth'
    evaluator.load_model(config, model_name)
    # eval_score = evaluator.eval_agent(test_env, evaluator.RSSM, evaluator.ObsEncoder, evaluator.ObsDecoder, evaluator.ActionModel, 0)
    # print('eval score: ', eval_score)
    evaluator.eval_visualize(env, model_name, interval=50, frame_save=5, visualize_episode=10, random=False)
    # for f in sorted(os.listdir(model_dir)):
    #     eval_score = evaluator.eval_saved_agent(env,  model_name)
    #     if eval_score > best_score:
    #         print(f'..Best model: {f}, best_score:{eval_score}')
    #         best_score=eval_score
    #         best_model = f


if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='robodesk', help='env name')
    parser.add_argument('--eval_episode', type=int, default=4, help='number of episodes to eval')
    parser.add_argument("--id", type=str, default='21', help='Experiment ID')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--noise', action='store_true', help='noise in the dynamics')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--distractor', action='store_true', help='distractor in the input observation')
    parser.add_argument('--disentangle', action='store_true')
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    parser.add_argument('--no-distractor', dest='distractor', action='store_false')
    parser.add_argument('--no-disentangle', dest='disentangle', action='store_false')
    parser.set_defaults(noise=True)
    parser.set_defaults(distractor=True)
    parser.set_defaults(disentangle=True)
    args = parser.parse_args()
    main(args)
