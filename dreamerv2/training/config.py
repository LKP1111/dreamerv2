import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict

@dataclass
class RoboDeskConfigV2():
    '''default HPs that are known to work for Cartpole envs'''
    harmony: bool = True

    # env desc
    env: str = "robodesk"
    seed: int = 0
    obs_shape: Tuple = (3, 64, 64)
    action_size: int = 2
    pixel: bool = True
    action_repeat: int = 2
    time_limit: int = 1000

    # buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # training desc, replay_ratio = 1 / 10
    train_steps: int = int(1e6)
    train_every: int = 1000
    collect_intervals: int = 100
    batch_size: int = 50
    seq_len: int = 50
    eval_every = int(1e4)
    eval_episode: int = 5
    eval_render: bool = False
    visualize_episode: int = 3
    save_every: int = int(2e5)
    seed_episodes: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'
    seed_steps: int = 500

    # latent space desc
    rssm_type: str = 'continuous'
    embedding_size: int = 1024
    rssm_node_size: int = 400
    rssm_info: Dict = field(
        default_factory=lambda: {'deter_size': 400, 'stoch_size': 20, 'class_size': 20, 'category_size': 20,
                                 'min_std': 0.1})

    # objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 15
    lr: Dict = field(default_factory=lambda: {'model': 3e-4, 'actor': 8e-05, 'critic': 8e-5})
    loss_scale: Dict = field(default_factory=lambda: {'kl': 1.0, 'reward': 1.0, 'discount': 1.0})  # important!!!
    kl: Dict = field(default_factory=lambda: {'use_kl_balance': True, 'kl_balance_scale': 0.8, 'use_free_nats': False,
                                              'free_nats': 0.0})
    use_slow_target: float = True
    slow_target_update: int = 1000
    slow_target_fraction: float = 1.0

    # actor critic
    actor: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'min_std': 1e-4, 'init_std': 2,
                                 'mean_scale': 3, 'activation': nn.ELU})
    critic: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    expl: Dict = field(
        default_factory=lambda: {'train_noise': 0.4, 'eval_noise': 0.0, 'expl_min': 0.05, 'expl_decay': 10000.0,
                                 'expl_type': 'add_noise'})  # epsilon_greedy or add_noise or no
    actor_grad: str = "dynamics"  # "'reinforce' # or dynamics
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    # learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda: {})  # no use
    obs_decoder: Dict = field(default_factory=lambda: {})  # no use
    # obs_encoder: Dict = field(
    #     default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU, 'kernel': 4,
    #                              'depth': 32})
    # obs_decoder: Dict = field(
    #     default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU, 'kernel': 4,
    #                              'depth': 32})
    reward: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    discount: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'binary', 'activation': nn.ELU, 'use': True})

@dataclass
class RoboDeskConfigIFactor():
    '''default HPs that are known to work for Cartpole envs'''
    # env desc
    env: str = "robodesk"
    seed: int = 0
    obs_shape: Tuple = (3, 64, 64)
    action_size: int = 2
    pixel: bool = True
    action_repeat: int = 2
    time_limit: int = 1000

    # train_policy_start: int = 0
    # train_policy_every: int = 10
    # collect_policy_intervals: int = 10

    # Algorithm desc
    disentangle: bool = True

    # buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    # training desc
    train_steps: int = int(1e6)
    train_every: int = 1000
    collect_intervals: int = 100
    batch_size: int = 50
    seq_len: int = 50
    eval_every = int(1e4)
    eval_episode: int = 5
    eval_render: bool = False
    visualize_episode: int = 3
    save_every: int = int(2e5)
    seed_episodes: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'
    seed_steps: int = 500

    # latent space desc
    rssm_type: str = 'continuous'
    embedding_size: int = 1024
    rssm_node_size: int = 400
    # performance not good
    rssm_info: Dict = field(
        default_factory=lambda: {'deter_size_s1': 120, 'deter_size_s2': 40, 'deter_size_s3': 40, 'deter_size_s4': 40,
                                 'stoch_size_s1': 20, 'stoch_size_s2': 10, 'stoch_size_s3': 10, 'stoch_size_s4': 10,
                                 'class_size': 20, 'category_size_s1': 30, 'category_size_s2': 10,
                                 'category_size_s3': 10, 'category_size_s4': 10, 'min_std': 0.1})

    # objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 15
    lr: Dict = field(default_factory=lambda: {'model': 6e-4, 'actor': 8e-05, 'critic': 8e-5})
    loss_scale: Dict = field(
        default_factory=lambda: {'kl_s1': 2, 'kl_s2': 2, 'kl_s3': 0.25, 'kl_s4': 0.25, 'reward': 5.0, 'discount': 10.0,
                                 'aux_reward_1': 0.1, 'aux_reward_2': 0.1, 'aux_action_1': 0.1, 'aux_action_2': 0.1})
    kl: Dict = field(default_factory=lambda: {'use_kl_balance': True, 'kl_balance_scale': 0.8, 'use_free_nats': False,
                                              'free_nats': 0.0})
    use_slow_target: float = True
    slow_target_update: int = 1000
    slow_target_fraction: float = 1.0

    # actor critic
    actor: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'min_std': 1e-4, 'init_std': 2,
                                 'mean_scale': 3, 'activation': nn.ELU})
    critic: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    expl: Dict = field(
        default_factory=lambda: {'train_noise': 0.4, 'eval_noise': 0.0, 'expl_min': 0.05, 'expl_decay': 10000.0,
                                 'expl_type': 'add_noise'})  # epsilon_greedy or add_noise or no
    actor_grad: str = "dynamics"  # "'reinforce' # or dynamics
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    # learnt world-models desc
    # obs_encoder: Dict = field(
    #     default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU, 'kernel': 4,
    #                              'depth': 32})
    # obs_decoder: Dict = field(
    #     default_factory=lambda: {'layers': 2, 'node_size': 100, 'dist': 'normal', 'activation': nn.ELU, 'kernel': 4,
    #                              'depth': 32})
    reward: Dict = field(
        default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    aux_reward_1: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    aux_reward_2: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'normal', 'activation': nn.ELU})
    aux_action_1: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'categorical', 'activation': nn.ELU})
    aux_action_2: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'categorical', 'activation': nn.ELU})

    action: Dict = field(default_factory=lambda: {'layers': 4, 'node_size': 400, 'dist': None, 'activation': nn.ELU})

    discount: Dict = field(
        default_factory=lambda: {'layers': 3, 'node_size': 400, 'dist': 'binary', 'activation': nn.ELU, 'use': True})


# Following HPs are not a result of detailed tuning.   

@dataclass
class MinAtarConfig():
    '''default HPs that are known to work for MinAtar envs '''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = True
    action_repeat: int = 1
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc, replay_ratio = 1 / 10
    train_steps: int = int(5e6)
    train_every: int = 50                                  #reduce this to potentially improve sample requirements
    collect_intervals: int = 5 
    batch_size: int = 50 
    seq_len: int = 50
    eval_episode: int = 4
    eval_render: bool = True
    save_every: int = int(1e5)
    seed_steps: int = 4000
    model_dir: int = 'results'
    gif_dir: int = 'results'
    
    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 200
    rssm_node_size: int = 200
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':200, 'stoch_size':20, 'class_size':20, 'category_size':20, 'min_std':0.1})
    
    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':0.1, 'reward':1.0, 'discount':5.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 100
    slow_target_fraction: float = 1.00

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':7000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':3, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})


@dataclass
class MiniGridConfig():
    '''default HPs that are known to work for MiniGrid envs'''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = True
    action_repeat: int = 1
    time_limit: int = 1000
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(1e6)
    train_every: int = 5
    collect_intervals: int = 5
    batch_size: int = 50
    seq_len: int = 8
    eval_episode: int = 5
    eval_render: bool = False
    save_every: int = int(5e4)
    seed_episodes: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'

    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':100, 'stoch_size':256, 'class_size':16, 'category_size':16, 'min_std':0.1})

    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 8
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':1, 'reward':1.0, 'discount':10.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':10000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':2, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':2, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    
