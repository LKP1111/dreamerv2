import minatar
import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name) 
        self.minimal_actions = self.env.minimal_action_set()
        h,w,c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c,h,w))

    def reset(self):
        self.env.reset()
        """(c, h, w) (3, 10, 10) 这里 minatar 不是 grayscale"""
        return self.env.state().transpose(2, 0, 1)
    
    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed='None'):
        self.env = minatar.Environment(self.env_name, random_seed=seed)
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


class MyCartPoleWrapper(gym.Wrapper):
    def __init__(self):
        # env = gym.make('CartPole-v1', render_mode="human")
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        super(MyCartPoleWrapper, self).__init__(env)
        self.env = env
        self.n_step = 0

    def reset(self, seed=None):
        self.n_step = 0
        state, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed=seed)
        return state

    def step(self, action) -> object:
        state, reward, terminated, truncated, info = self.env.step(action)
        self.n_step += 1
        done = terminated or truncated
        return state, reward, done, info


class MyBreakoutWrapper(gym.Wrapper):
    """
    We modify the Atari environment to accelerate the training with some tricks:
        Episode termination: Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and co. since it helps value estimation.
        Frame skipping: Return only every `skip`-th frame.
        Observation resize: Warp frames from 210x160 to 84x84 as done in the Nature paper and later work.
        Frame Stacking: Stack k last frames. Returns lazy array, which is much more memory efficient.
    """

    def __init__(self):
        self.env = gym.make("ALE/Breakout-v5",
                            render_mode="rgb_array",
                            obs_type="grayscale",
                            frameskip=4,
                            full_action_space=False)
        self.env.action_space.seed(seed=1)
        self.env.unwrapped.reset(seed=1)
        self.max_episode_steps = self.env._max_episode_steps if hasattr(self.env, '_max_episode_steps') else 1e5
        super(MyBreakoutWrapper, self).__init__(self.env)
        # self.env.seed(config.env_seed)
        self.num_stack = 1
        self.obs_type = "grayscale"
        self.frames = deque([], maxlen=1)
        # self.image_size = [210, 160] if config.img_size is None else config.img_size
        self.image_size = [64, 64]
        self.noop_max = 30
        self.lifes = self.env.unwrapped.ale.lives()
        self.was_real_done = True
        self.grayscale, self.rgb = False, False
        if self.obs_type == "rgb":
            self.rgb = True
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.image_size[0], self.image_size[1], 3 * self.num_stack), dtype=np.uint8)
        elif self.obs_type == "grayscale":
            self.grayscale = True
            # self.observation_space = gym.spaces.Box(
            #     low=0, high=255, shape=(self.image_size[0], self.image_size[1], self.num_stack), dtype=np.uint8)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.num_stack, self.image_size[0], self.image_size[1]), dtype=np.uint8)
        else:  # ram type
            self.observation_space = self.env.observation_space
        # assert self.env.unwrapped.get_action_meanings()[0] == "NOOP"
        # assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(self.env.unwrapped.get_action_meanings()) >= 3
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self._render_mode = self.render_mode
        self._episode_step = 0

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed):
        info = {}
        if self.was_real_done:
            self.env.reset()
            # Execute NoOp actions
            num_noops = np.random.randint(0, self.noop_max)
            for _ in range(num_noops):
                obs, _, done, _, _ = self.env.step(0)
                if done:
                    self.env.reset(seed=seed)
            # try to fire
            obs, _, done, _, _ = self.env.step(1)
            if done:
                obs = self.env.reset()
            # stack reset observations
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

            self._episode_step = 0
        else:
            obs, _, done, _, _ = self.env.step(0)
            for _ in range(self.num_stack):
                self.frames.append(self.observation(obs))

        self.lifes = self.env.ale.lives()
        self.was_real_done = False
        return self._get_obs()

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        self.frames.append(self.observation(observation))
        lives = self.env.ale.lives()
        # avoid environment bug
        if self.max_episode_steps is not None:
            if self._episode_step >= self.max_episode_steps:
                terminated = True
        self.was_real_done = terminated
        if (lives < self.lifes) and (lives > 0):
            terminated = True
        truncated = self.was_real_done
        self.lifes = lives
        self._episode_step += 1
        return self._get_obs(), self.reward(reward), terminated or truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames))[:]

    def observation(self, frame):
        if self.grayscale:
            return np.expand_dims(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA), -1).transpose(2, 0, 1)
        elif self.rgb:
            return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            return frame

    def reward(self, reward):
        return np.sign(reward)


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


"""下面只是针对特定 minatar 环境的 obs_wrapper, 对观测进行处理, 让环境变成 POMDP"""
class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class asterixPOMDP(gym.ObservationWrapper):
    '''index 2 (trail) is removed, which gives ball's direction'''
    def __init__(self, env):
        super(asterixPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class freewayPOMDP(gym.ObservationWrapper):
    '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''
    def __init__(self, env):
        super(freewayPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-5,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1]], axis=0)    

class space_invadersPOMDP(gym.ObservationWrapper):
    '''index 2-3 (trail) are removed, which gives aliens' direction'''
    def __init__(self, env):
        super(space_invadersPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-2,h,w))
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)

class seaquestPOMDP(gym.ObservationWrapper):
    '''index 3 (trail) is removed, which gives enemy and driver's direction'''
    def __init__(self, env):
        super(seaquestPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
        
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6], observation[7], observation[8], observation[9]], axis=0)    

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self, seed):
        return self.env.reset(seed=seed)
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
