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
from gym import logger, spaces
from IFactor.utils.wrapper import RoboDeskImageWrapper
from IFactor.training.config import RoboDeskDiscreteConfig
from IFactor.training.dtrainer import Trainer
from IFactor.training.evaluator import Evaluator
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from myenv.robodesk.robodesk import RoboDesk, RoboDeskWithTV
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor

from absl import logging

from gpu_mem_track import MemTracker

logging.set_verbosity(logging.FATAL)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MUJOCO_GL'] = 'egl'

class EvaluateCallback(BaseCallback):
    def __init__(self, eval_env, model_dir, eval_freq=10000):
        super(EvaluateCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -float('inf')
        self.model_dir = model_dir
    def _on_step(self) -> bool:
        # Implement your evaluation logic here
        # return super(EvaluateCallback, self)._on_step()
        # print(self.n_calls)
        if self.n_calls % self.eval_freq == 1:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
            print(f"Eval reward at {self.num_timesteps}: {mean_reward:.2f} +/- {std_reward:.2f}")
            self.logger.record('eval_reward', mean_reward)
            # Save the model if it has the best evaluation value so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.model_dir, 'best_model.zip'))
        return True

    # def on_step_end(self) -> bool:
    #     if self.n_calls % self.eval_freq == 1:
    #         mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
    #         print(f"Eval reward at {self.num_timesteps}: {mean_reward:.2f} +/- {std_reward:.2f}")
    #         self.logger.record('eval_reward', mean_reward)
    #         # Save the model if it has the best evaluation value so far
    #         if mean_reward > self.best_mean_reward:
    #             self.best_mean_reward = mean_reward
    #             self.model.save(os.path.join(self.model_dir, 'best_model.zip'))
    #     return True

# gpu_tracker = MemTracker()
class RobodeskWithWorldModel(gym.Env):
    def __init__(self, args, type='s1') -> None:
        # type ['s1', 's12', 's123', 's1234']
        self.env = RoboDeskImageWrapper(RoboDeskWithTV(
            task="tv_green_hue",
            action_repeat=2,
            episode_length=1000,
            distractors='all',
            tv_video_file_pattern=os.path.expanduser("~/.driving_car/*.mp4")
        ))
        self.type = type
        obs_shape = self.env.observation_space.shape
        action_size = self.env.action_space.shape[0]
        self.action_space = self.env.action_space
        if self.type == 's1':
            size = 140
        elif self.type == 's12':
            size = 280
        elif self.type == 's123':
            size = 350
        elif self.type == 's34':
            size = 140
        else:
            size = 420
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[size])
        result_dir = os.path.join('results', 'robodesk', '{}'.format(1000))
        model_dir = os.path.join(result_dir, 'models')
        gif_dir = os.path.join(result_dir, 'visualization')
        # dir to save learnt models
        os.makedirs(model_dir, exist_ok=True)
        config = RoboDeskDiscreteConfig(
            env='robodesk',
            seed=args.seed,
            obs_shape=obs_shape,
            action_size=action_size,
            model_dir=model_dir,
            gif_dir=gif_dir,
            )
        if torch.cuda.is_available() and args.device:
            device = torch.device('cuda')
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            device = torch.device('cpu')

        print('using :', device)
        config_dict = config.__dict__
        # gpu_tracker.track()
        self.trainer = Trainer(config, device)
        # gpu_tracker.track()
        # trainer._print_summary()
        last_model_name = "anonymized_path/results/robodesk/21/models/models_best/models_best.pth"
        save_dict = torch.load(os.path.join(model_dir, last_model_name))
        self.trainer.load_save_dict(save_dict)

    def reset(self):
        obs, score = self.env.reset(), 0
        done = False
        self.prev_rssmstate = self.trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, self.trainer.action_size).to(self.trainer.device)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            if obs.dtype == np.uint8:
                obs_tensor = obs_tensor.div(255).sub_(0.5)
            embed = self.trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(self.trainer.device))
            _, posterior_rssm_state = self.trainer.RSSM.rssm_observe(embed, prev_action, not done, self.prev_rssmstate)
            self.prev_rssmstate = posterior_rssm_state
            state = self.get_state(posterior_rssm_state)
            return state.cpu().numpy()
    
    def step(self, action):
        next_obs, rew, done, _ = self.env.step(action)
        prev_action = torch.tensor(action).unsqueeze(0).to(self.trainer.device)
        with torch.no_grad():
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
            if next_obs.dtype == np.uint8:
                obs_tensor = obs_tensor.div(255).sub_(0.5)
            embed = self.trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(self.trainer.device))
            _, posterior_rssm_state = self.trainer.RSSM.rssm_observe(embed, prev_action, True, self.prev_rssmstate)
            self.prev_rssmstate = posterior_rssm_state
            state = self.get_state(posterior_rssm_state)
            return state.cpu().numpy(), rew, done, {}
    
    def render(self, mode='human'):
        pass
    
    def get_state(self, posterior_rssm_state):
        stoch_dict = self.trainer.RSSM.get_stoch_state_dict(posterior_rssm_state)
        deter_dict = self.trainer.RSSM.get_deter_state_dict(posterior_rssm_state)
        if self.type == 's1':
            return torch.cat([deter_dict['s1'], stoch_dict['s1']], dim=-1).squeeze()
        elif self.type == 's12':
            return torch.cat([deter_dict['s1'], deter_dict['s2'], stoch_dict['s1'], stoch_dict['s2']], dim=-1).squeeze()
        elif self.type == 's123':
            return torch.cat([deter_dict['s1'], deter_dict['s2'], deter_dict['s3'], stoch_dict['s1'], stoch_dict['s2'], stoch_dict['s3']], dim=-1).squeeze()
        elif self.type == 's34':
            return torch.cat([deter_dict['s3'], deter_dict['s4'], stoch_dict['s3'], stoch_dict['s4']], dim=-1).squeeze()
        elif self.type == 's1234':
            return torch.cat([deter_dict['s1'], deter_dict['s2'], deter_dict['s3'], deter_dict['s4'], stoch_dict['s1'], stoch_dict['s2'], stoch_dict['s3'], stoch_dict['s4']], dim=-1).squeeze()
        else:
            raise NotImplementedError

def main(args):
    env = RobodeskWithWorldModel(args, type=args.type)
    test_env = RobodeskWithWorldModel(args, type=args.type)
    vec_env = VecMonitor(DummyVecEnv([lambda: env]))
    test_env = VecMonitor(DummyVecEnv([lambda: test_env]))
    policy_kwargs = dict(activation_fn=torch.nn.ELU,
                     net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
    result_dir = os.path.join(f'anonymized_path/{args.type}/seed_{args.seed}')
    td_dir = os.path.join(result_dir, 'tb')
    os.makedirs(td_dir, exist_ok=True)
    # Create the SAC agent
    model = SAC("MlpPolicy", vec_env, learning_rate=0.0002, policy_kwargs=policy_kwargs, tensorboard_log=td_dir, verbose=1)
    
    
    callback = EvaluateCallback(test_env, model_dir=result_dir)
    # Train the agent
    model.learn(total_timesteps=1000001, callback=callback)

    # # Step 4: Evaluate and visualize the learned policy
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()


if __name__ == "__main__":
    # python test/cartpole_run.py --noise False --distractor True --id 4
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='cartpole', help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument("--type", type=str, default='s1', help='type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--resume', action='store_true', help='resume')

    args = parser.parse_args()
    main(args)
