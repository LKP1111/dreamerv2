import warnings

warnings.filterwarnings("ignore")
import wandb
import argparse
import os, sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
import gym
from IFactor.utils.wrapper import RoboDeskImageWrapper
from myenv.cartpole import CartPoleWorldEnv
from IFactor.utils.visualize import Visualizer
from IFactor.training.config import DMCConfig, TestDMCConfig
from IFactor.training.dtrainer import Trainer
from IFactor.training.evaluator import Evaluator

from myenv.dmc2gym import make_dmc_env
from absl import logging

# from gpu_mem_track import MemTracker

logging.set_verbosity(logging.FATAL)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MUJOCO_GL'] = 'egl'


# gpu_tracker = MemTracker()


def main(args):
    wandb.login()
    domain_name = args.domain_name
    if domain_name == "cheetah":
        task_name = "run"
    elif domain_name == "walker":
        task_name = "walk"
    elif domain_name == "reacher":
        task_name = "easy"
    else:
        raise NotImplementedError

    variant = args.variant
    config_class = TestDMCConfig
    # if variant == "video_background_camera_jitter":
    #     config_class = JittorTestDMCConfig
    # elif variant == "video_background":
    #     config_class = VideoTestDMCConfig
    # elif variant == "video_background_noisy_sensor":
    #     config_class = NoisyTestDMCConfig
    # elif variant == "noiseless":
    #     config_class = NoiselessTestDMCConfig
    # else:
    #     raise NotImplementedError

    exp_id = args.id

    # '''make dir for saving results'''
    # result_dir = os.path.join('results', domain_name, task_name, variant, exp_id)
    # model_dir = os.path.join(result_dir, 'models')
    # gif_dir = os.path.join(result_dir, 'visualization')
    # # dir to save learnt models
    # os.makedirs(model_dir, exist_ok=True)

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

    env = make_dmc_env(
        domain_name=domain_name,
        task_name=task_name,
        variant=variant,
        max_episode_length=1000,
        action_repeat=2,
        seed=args.seed
    )
    test_env = make_dmc_env(
        domain_name=domain_name,
        task_name=task_name,
        variant=variant,
        max_episode_length=1000,
        action_repeat=2,
        seed=args.seed
    )

    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    print(obs_shape, action_size)

    config = config_class(
        domain_name=domain_name,
        task_name=task_name,
        variant=variant,
        seed=args.seed,
        obs_shape=obs_shape,
        action_size=action_size
        )

    config_dict = config.__dict__
    # gpu_tracker.track()
    if args.domain_name == "cheetah":
        model_dir = "/home/frank/Projects/interprl/results/cheetah/run/video_background_camera_jitter/12/models/models_best"
    elif args.domain_name == "walker":
        model_dir = "/home/frank/Projects/interprl/results/walker/walk/video_background_noisy_sensor/1/models/models_best"
    elif args.domain_name == "reacher":
        model_dir = "/home/lamda5/liuyr/research/interprl/results/reacher/easy/video_background/1/models/models_best/"
        test_decoder_path = "/home/lamda5/liuyr/research/interprl/results/reacher/easy/video_background/1/models/eval/300001.pth"
        pic_dir = "/home/lamda5/liuyr/research/interprl/results/reacher/easy/video_background/1/models/eval/visualization"
    model_name = os.path.join(model_dir, 'models_best.pth')
    model_save_dir = os.path.join(model_dir, 'eval')
    os.makedirs(model_save_dir, exist_ok=True)
    # "/home/lamda5/liuyr/research/interprl/results/reacher/easy/video_background/1/models/models_best/models_best.pth"
    # gpu_tracker.track()
    # trainer._print_summary()
    trainer = Trainer(config, device)
    trainer.load_model(model_name)
    trainer.init_extra_decoder()
    trainer.load_test_decoder(test_decoder_path)
    visualizer = Visualizer(config)
    obs, score = env.reset(), 0
    done = False
    prev_rssmstate = trainer.RSSM._init_rssm_state(1)
    prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
    episode_actor_ent = []
    scores = []
    best_mean_score = 0
    print(f"Enter training iteration")
    episode_num = 5
    interval = 6
    pic_num = 6
    for e in range(episode_num):
        step = 0
        current_save_frame = 0
        frame_dict = {"obs": [], "1": [], "12": [], "3": [], "4": [], "34": []}
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if obs.dtype == np.uint8:
                    obs_tensor = obs_tensor.div(255).sub_(0.5)
                embed = trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(trainer.device))
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                if step % interval == 0 and current_save_frame < pic_num:
                    deter_dict = trainer.RSSM.get_deter_state_dict(posterior_rssm_state)
                    stoch_dict = trainer.RSSM.get_stoch_state_dict(posterior_rssm_state)
                    input_decoder1 = torch.cat([deter_dict['s1'], stoch_dict['s1']], dim=-1)
                    input_decoder12 = torch.cat([deter_dict['s1'], deter_dict['s2'], stoch_dict['s1'], stoch_dict['s2']], dim=-1)
                    input_decoder3 = torch.cat([deter_dict['s3'], stoch_dict['s3']], dim=-1)
                    input_decoder4 = torch.cat([deter_dict['s4'], stoch_dict['s4']], dim=-1)
                    input_decoder34 = torch.cat([deter_dict['s3'], deter_dict['s4'], stoch_dict['s3'], stoch_dict['s4']], dim=-1)
                    
                    obs_1 = trainer.TestObsDecoder1(input_decoder1.detach()).mean
                    obs_12 = trainer.TestObsDecoder12(input_decoder12.detach()).mean
                    obs_3 = trainer.TestObsDecoder3(input_decoder3.detach()).mean
                    obs_4 = trainer.TestObsDecoder4(input_decoder4.detach()).mean
                    obs_34 = trainer.TestObsDecoder34(input_decoder34.detach()).mean
                    frame_dict["obs"].append(visualizer.obs_to_image(obs_tensor.cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    frame_dict["1"].append(visualizer.obs_to_image(obs_1.squeeze().cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    frame_dict["12"].append(visualizer.obs_to_image(obs_12.squeeze().cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    frame_dict["3"].append(visualizer.obs_to_image(obs_3.squeeze().cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    frame_dict["4"].append(visualizer.obs_to_image(obs_4.squeeze().cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    frame_dict["34"].append(visualizer.obs_to_image(obs_34.squeeze().cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
                    current_save_frame += 1
                
                # model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                asr_state = trainer.RSSM.get_asr_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(asr_state)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            step += 1
            score += rew

            if done:
                visualizer.output_picture_dmc(pic_dir, e, frame_dict, kind=['obs', '1', '12', '3', '4', '34'])
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
                break
            else:
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    # evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    # python test/cartpole_run.py --noise False --distractor True --id 4
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default='cheetah')
    parser.add_argument("--variant", type=str, default='video_background_camera_jitter')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')

    args = parser.parse_args()
    main(args)

# --variant video_background_camera_jitter
# --variant video_background_noisy_sensor
# --variant video_background

# CUDA_VISIBLE_DEVICES

# python script/train/d_dmc.py --domain_name cheetah --variant video_background
# python script/train/d_dmc.py --domain_name walker --variant video_background_camera_jitter --seed 2 --id 2
# python script/train/d_dmc.py --domain_name reacher --variant video_background
