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

    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', domain_name, task_name, variant, exp_id)
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
        action_size=action_size,
        model_dir=model_dir,
        gif_dir=gif_dir,
    )

    config_dict = config.__dict__
    # gpu_tracker.track()
    if args.domain_name == "cheetah":
        model_dir = "anonymized_path/results/cheetah/run/video_background_camera_jitter/12/models/models_best"
    elif args.domain_name == "walker":
        model_dir = "anonymized_path/results/walker/walk/video_background_noisy_sensor/1/models/models_best"
    elif args.domain_name == "reacher":
        model_dir = "anonymized_path/results/reacher/easy/video_background/1/models/models_best/"
        test_decoder_path = "anonymized_path/results/reacher/easy/video_background/1/models/models_best"
    model_name = os.path.join(model_dir, 'models_best.pth')
    model_save_dir = os.path.join(model_dir, 'eval')
    os.makedirs(model_save_dir, exist_ok=True)
    # gpu_tracker.track()
    # trainer._print_summary()
    trainer = Trainer(config, device)
    trainer.load_model(model_name)
    trainer.init_extra_decoder()
    # trainer.load_test_decoder(test_decoder_path)

    with wandb.init(project='DMC_test', entity='yuren', config=config_dict):
        # if 1 == 1:
        """training loop"""
        print('...training...')
        train_metrics = {}
        eval_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        print(f"Enter training iteration")
        for iter in range(1, trainer.config.train_steps + 2):
            if iter % trainer.config.train_every == 1:
                train_metrics = trainer.train_batch_test_decoder(train_metrics)
                wandb.log(train_metrics, step=iter)
            if iter % trainer.config.save_every == 1:
                save_path = os.path.join(model_save_dir, f'{iter}.pth')
                model_dir = trainer.save_test_decoder(save_path)
                print('save_model succeeds')
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if obs.dtype == np.uint8:
                    obs_tensor = obs_tensor.div(255).sub_(0.5)
                embed = trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(trainer.device))
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                # model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                asr_state = trainer.RSSM.get_asr_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(asr_state)
                action_ent = torch.mean(-action_dist.log_prob(action)).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                wandb.log(train_metrics, step=iter)
                print(score)
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    # evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default='cheetah')
    parser.add_argument("--variant", type=str, default='video_background_camera_jitter')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')

    args = parser.parse_args()
    main(args)