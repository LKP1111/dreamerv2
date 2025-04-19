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
from dreamerv2.utils.wrapper import RoboDeskImageWrapper

"""ifactor"""
from dreamerv2.training.dtrainer_ifactor import Trainer
from dreamerv2.training.config import RoboDeskConfigIFactor
# from dreamerv2.training.evaluator_ifactor import Evaluator

"""dreamer_v2"""
# from dreamerv2.training.trainer_v2 import Trainer
# from dreamerv2.training.config import RoboDeskConfigV2
from dreamerv2.training.evaluator_v2 import Evaluator

from myenv.robodesk.robodesk import RoboDesk, RoboDeskWithTV, RoboDeskBase
from absl import logging

from gpu_mem_track import MemTracker

logging.set_verbosity(logging.FATAL)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MUJOCO_GL'] = 'egl'


# gpu_tracker = MemTracker()


def main(args):
    wandb.login()
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

    # env = RoboDeskWithTV(
    #     task="tv_green_hue",
    #     action_repeat=2,
    #     episode_length=1000,
    #     distractors='all',
    #     tv_video_file_pattern=os.path.expanduser("~/.driving_car/*.mp4")
    # )
    env = RoboDesk(
        task='push_green', reward='dense', action_repeat=2,
        episode_length=500, image_size=64
    )
    env.seed(args.seed)
    env = RoboDeskImageWrapper(env)
    test_env = RoboDesk(
        task='push_green', reward='dense', action_repeat=2,
        episode_length=500, image_size=64
    )
    # test_env = RoboDeskWithTV(
    #     task="tv_green_hue",
    #     action_repeat=2,
    #     episode_length=1000,
    #     distractors='all',
    #     tv_video_file_pattern=os.path.expanduser("~/.driving_car/*.mp4")
    # )
    test_env.seed(args.seed + 1)
    test_env = RoboDeskImageWrapper(test_env)
    # eval_env = RoboDesk()
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    print(obs_shape, action_size)

    config = RoboDeskConfigIFactor(
        env=env_name,
        seed=args.seed,
        obs_shape=obs_shape,
        action_size=action_size,
        model_dir=model_dir,
        gif_dir=gif_dir,
    )

    config_dict = config.__dict__
    # gpu_tracker.track()
    trainer = Trainer(config, device)
    # gpu_tracker.track()
    # trainer._print_summary()
    resume_step = 0
    if args.resume:
        resume_step = trainer.resume_training(model_dir, 200000)

    evaluator = Evaluator(config, device)

    best_eval_score = -np.inf
    # with wandb.init(project='RoboDesk', entity='yuren', config=config_dict):
    with wandb.init(entity="aaaa112-1",
                    project='test',
                    config=config_dict):
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
        best_save_path = os.path.join(model_dir, 'models_best.pth')
        print(f"Enter training iteration")
        for iter in range(1 + resume_step, trainer.config.train_steps + 2 + resume_step):
            if iter % trainer.config.train_every == 1:
                train_metrics = trainer.train_batch(train_metrics)
                wandb.log(train_metrics, step=iter)
            """
            cur (hard_update):
                replay_ratio = 100 / 1000 = 1 / 10
                critic_update_freq = 1000 iter = 100 update
                critic_update_frac = 1
                critic_update_scale = 1 per 1000 iter
                
            v3 (soft_update): 
                replay_ratio = 1 / 8
                critic_update_freq = 1 update = 8 iter
                critic_update_frac = 0.02
                critic_update_scale = 0.02 per 8 iter = 1 per 400 iter
            
            k soft_update < 1 hard_update ???
            """
            if iter % trainer.config.slow_target_update == 0:
                trainer.update_target()
            if iter % trainer.config.save_every == 0:
                model_dir = trainer.save_model(iter)
            # TODO
            # if iter % trainer.config.eval_every == 1:
            #     eval_score = evaluator.eval_agent(test_env, trainer.RSSM, trainer.ObsEncoder, trainer.ObsDecoder,
            #                                       trainer.ActionModel, iter)
            #     eval_metrics["eval_rewards"] = eval_score
            #     wandb.log(eval_metrics, step=iter)
            #     if eval_score > best_mean_score:
            #         best_mean_score = eval_score
            #         trainer.save_model(iter, best=True)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if obs.dtype == np.uint8:
                    obs_tensor = obs_tensor.div(255).sub_(0.5)
                embed = trainer.ObsEncoder(obs_tensor.unsqueeze(0).to(trainer.device))
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                # model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                asr_state = trainer.RSSM.get_asr_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(asr_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(-action_dist.log_prob(action)).item()
                episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] = np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                """avg on 20 scores"""
                if len(scores) > 20:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average > best_mean_score:
                        best_mean_score = current_average
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)

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
    evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    # python test/cartpole_run.py --noise False --distractor True --id 4
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='robodesk', help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--resume', action='store_true', help='resume')

    args = parser.parse_args()
    main(args)
