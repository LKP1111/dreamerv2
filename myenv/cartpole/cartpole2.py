"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import pdb
import math
from typing import Optional, Union

import numpy as np

import cv2, os

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from .utils import NumPyRNGWrapper
import logging
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

class CartPoleWorldEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,
        case: int = 1,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        noises: float = 0,
        polelen: float = 0.5,
        mode: Optional[str] = "rgb_array",
        reward_type: str = "dense",
        image_size: int = 64,
        episode_length = 200,
        seed: Optional[int] = None,
        state_obs_noise = False,
        distractor = True,
        full_state = False,):
        self.case = case
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.noises = noises
        self.green_step = 5
        self.total_mass = self.masspole + self.masscart
        self.length = polelen  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.reward_type = reward_type
        self.image_size = image_size
        self.episode_length = episode_length
        self.tau = 0.02  # seconds between state updates
        self.frictioncart = 5e-4  # AA Added cart friction
        self.friction_multiplier = 1.5
        self.friction_add = 0.1
        self.frictionpole = 2e-6  # AA Added cart friction
        self.distractor_terminated = False
        self.kinematics_integrator = "euler"
        self.num_steps = 0
        self.np_rng = NumPyRNGWrapper(seed)
        self.state_obs_noise = True if state_obs_noise else False
        self.distractor = distractor
        self.full_state = full_state
        # Angle at which to fail the episode
        self.theta_threshold= 45
        self.x_threshold= 5
        self.theta_threshold_radians = self.theta_threshold * 2 * math.pi / 360
        # self.x_threshold = 2.4

        if case < 4:
            self.force_mag = 10.0 * (1 + self.addnoise(case))
            self.case = 1
        else:
            self.force_mag = 10.0
            self.case = case

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_size = 8
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(0, 255, shape=[3, 64, 64], dtype=np.uint8)
        self.mode = mode

        # self.screen_width = 600
        self.screen_width = 400
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.distractor_state = None
        self.steps_beyond_terminated = None
        self.action_force_dict = {0: - self.force_mag, 1: - self.force_mag, 2: -2 * self.force_mag, 3: -2 * self.force_mag, 4: self.force_mag, 5: self.force_mag, 6: 2 * self.force_mag, 7: 2 * self.force_mag}
        self.action_green_dict = {0: - self.green_step, 1: self.green_step, 2: - self.green_step, 3: self.green_step, 4: - self.green_step, 5: self.green_step, 6: - self.green_step, 7: self.green_step}

    def step_kernel(self, state, action, terminal, friction_multiplier):
        if terminal:
            return (state[0], 0, state[2], 0)
        err_msg = f"{action!r} ({type(action)}) invalid"
        # print(self.action_space)
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = state
        force = self.action_force_dict[action]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (
        #     force + self.polemass_length * theta_dot**2 * sintheta
        # ) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta - friction_multiplier * self.frictioncart * np.sign(
            x_dot)) / self.total_mass
        # AA Added pole friction
        thetaacc = (self.gravity * sintheta - costheta * temp - friction_multiplier * self.frictionpole * theta_dot / self.polemass_length
                    ) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        noise = self.addnoise(self.case)

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = (theta + self.tau * theta_dot) * (1 + noise)
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = (theta + self.tau * theta_dot) * (1 + noise)
        state_noise = (self.np_rng.normal(0.0, 0.002), self.np_rng.normal(0.0, 0.02), self.np_rng.normal(0.0, 0.005), self.np_rng.normal(0.0, 0.03)) if self.state_obs_noise else (0, 0, 0, 0)
        new_state = (x + state_noise[0], x_dot + state_noise[1], theta + state_noise[2], theta_dot + state_noise[3])
        return new_state

    def step(self, action):  
        # self.friction_multiplier = np.clip(self.np_rng.choice([-1, 1]) * 0.1 + self.friction_multiplier, 1, 2)
        assert(self.friction_multiplier <=2.01 and self.friction_multiplier >= 0.99)
        self.friction_multiplier += self.friction_add
        self.friction_multiplier = np.clip(self.friction_multiplier, 1, 2)
        if self.friction_multiplier >= 1.95:
            self.friction_add *= -1
        if self.friction_multiplier <= 1.05:
            self.friction_add *= -1
        # print(self.friction_multiplier)
        # print(f"action_cart: {action_cart}, action_green: {action_green}")
        self.state = self.step_kernel(self.state, action, False, self.friction_multiplier)
        next_greeness = self.greeness + self.action_green_dict[action]
        self.greeness = np.clip(next_greeness, 0, 100)
        self.num_steps += 1
        (x, x_dot, theta, theta_dot) = self.state
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        ) or bool(self.num_steps >= self.episode_length)
        if self.reward_type == "dense":
            # reward = (1 - (x ** 2) / 5 - (theta ** 2) / 3) + (self.friction_multiplier  - 1) ** 2 / 10
            r1, r2, r3, r4, r5 = (x ** 2) / 5, (x_dot ** 2)/50, (theta ** 2) / 3, (theta_dot ** 2)/100, (self.friction_multiplier  - 1) / 10
            reward = max(1 - r1 - r2 - r3 - r4 + r5, 0)
            if not terminated:
                pass
            elif self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
                # reward = 1
                # reward = 2 * math.cos(theta) - 1
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward = 0
        elif self.reward_type == "sparse":
            if not terminated:
                reward = 1
            elif self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
                reward = 1
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward = 0
        else:
            raise NotImplementedError
        # distractor: fake state:
        random_action = self.np_rng.choice(self.action_size)
        (distractor_x, distractor_x_dot, distractor_theta, distractor_theta_dot) = self.step_kernel(self.distractor_state, random_action, self.distractor_terminated, 1)
        self.distractor_terminated = bool(
            distractor_x < -self.x_threshold * 0.5
            or distractor_x > self.x_threshold * 0.5
            or distractor_theta < -self.theta_threshold_radians
            or distractor_theta > self.theta_threshold_radians
        )
        distractor_x = np.clip(distractor_x, -self.x_threshold, self.x_threshold)
        distractor_theta = np.clip(distractor_theta, -self.theta_threshold_radians, self.theta_threshold_radians)
        self.distractor_state = (distractor_x, distractor_x_dot, distractor_theta, distractor_theta_dot)
        obs = self.render()
        obs = cv2.resize(obs, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        # obs = cv2.cvtColor(obs,cv2.COLOR_GRAY2BGR)
        obs = np.transpose(obs, axes=[2, 0, 1])
        obs_dict = {'image': obs, 'state': np.array(self.state, dtype=np.float32), 'fake_state': np.array(self.distractor_state), 'greeness': np.array(self.greeness, dtype=np.float32), 's1': np.array([self.state]), 's2': np.array([self.friction_multiplier]), 's3': np.array(self.greeness, dtype=np.float32), 's4': self.distractor_state}
        if self.full_state:
            return obs_dict, reward, terminated, {}
            # return obs_dict, reward, False, {}
        else:
            return obs_dict['image'], reward, terminated, {}
            # return obs_dict['image'], reward, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_rng.uniform(low=low, high=high, size=(4,))
        self.distractor_state = self.np_rng.uniform(low=low, high=high, size=(4,))
        self.greeness = self.np_rng.uniform(low=40, high=60, size=(1, ))
        self.steps_beyond_terminated = None
        self.friction_add = self.np_rng.choice([-0.1, 0.1])
        # print(self.friction_add)
        self.num_steps = 0
        obs = self.render()
        obs = cv2.resize(obs, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        obs = np.transpose(obs, axes=[2, 0, 1])
        # s1: [0], s2: [state[0], state[2]], s3: [state[1], state[3]], s4: distractor_state
        # obs_dict = {'image': obs, 'state': np.array(self.state, dtype=np.float32), 'fake_state': np.array(self.distractor_state), 'greeness': np.array(self.greeness, dtype=np.float32), 's1': np.array([self.state[1], self.state[3]]), 's2': np.array([self.state[0], self.state[2]]), 's3': np.array(self.greeness, dtype=np.float32), 's4': np.array(self.distractor_state)}
        obs_dict = {'image': obs, 'state': np.array(self.state, dtype=np.float32), 'fake_state': np.array(self.distractor_state), 'greeness': np.array(self.greeness, dtype=np.float32), 's1': np.array([self.state]), 's2': np.array([self.friction_multiplier]), 's3': np.array(self.greeness, dtype=np.float32), 's4': self.distractor_state}
        if self.full_state:
            return_obs = obs_dict
        else:
            return_obs = obs_dict['image']
        if not return_info:
            return return_obs
        else:
            return return_obs, {}

    def seed(self, seed=None):
        self.np_rng.seed(seed)

    def get_random_state(self):
        return self.np_rng.get_random_state()

    def set_random_state(self, random_state):
        self.np_rng.set_random_state(random_state)

    def render(self):
        assert self.mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        # polelen = scale * (2 * self.length)
        polelen = 125 * 2 * self.length
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state
        distractor_x = self.distractor_state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        cartxd = distractor_x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 70  # TOP OF CART
        cartyd = 250
        cart_coords_basic = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords_basic]
        cartd_coords = [(c[0] + cartxd, c[1] + cartyd) for c in cart_coords_basic]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        if self.distractor:
            gfxdraw.aapolygon(self.surf, cartd_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cartd_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        poled_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord_1 = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord_1 = (coord_1[0] + cartx, coord_1[1] + carty + axleoffset)
            pole_coords.append(coord_1)
            dcoord = pygame.math.Vector2(coord).rotate_rad(-distractor_x[2])
            dcoord = (dcoord[0] + cartxd, dcoord[1] + cartyd + axleoffset)
            poled_coords.append(dcoord)

        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        if self.distractor:
            gfxdraw.aapolygon(self.surf, poled_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, poled_coords, (202, 152, 101))

        l_green, r_green, t_green, b_green = 50, 150, 40, 10
        green_poly = [(l_green, b_green), (l_green, t_green), (r_green, t_green), (r_green, b_green)]
        gfxdraw.aapolygon(self.surf, green_poly, (0, int(self.greeness * 2.55), 0))
        gfxdraw.filled_polygon(self.surf, green_poly, (0, int(self.greeness * 2.55), 0))
        
        l_red, r_red, t_red, b_red = 250, 350, 40, 10
        red_poly = [(l_red, b_red), (l_red, t_red), (r_red, t_red), (r_red, b_red)]
        gfxdraw.aapolygon(self.surf, red_poly, (int((self.friction_multiplier-1) * 255), 0, 0))
        gfxdraw.filled_polygon(self.surf, red_poly, (int((self.friction_multiplier-1) * 255), 0, 0))
        try:
            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            # gfxdraw.aacircle(
            #     self.surf,
            #     int(self.screen_width / 2.0),
            #     int(25),
            #     int(15),
            #     (0, int(self.greeness * 2.55), 0),
            # )
            # gfxdraw.filled_circle(
            #     self.surf,
            #     int(self.screen_width / 2.0),
            #     int(25),
            #     int(15),
            #     (0, int(self.greeness * 2.55), 0),
            # )
            if self.distractor:
                gfxdraw.aacircle(
                    self.surf,
                    int(cartxd),
                    int(cartyd + axleoffset),
                    int(polewidth / 2),
                    (129, 132, 203),
                )
                gfxdraw.filled_circle(
                    self.surf,
                    int(cartxd),
                    int(cartyd + axleoffset),
                    int(polewidth / 2),
                    (129, 132, 203),
                )
        except Exception:
            print(f"Error: cartx: {cartx}, {carty}, {cartxd}, {cartyd}, {axleoffset}")
            logging.error(f"Error: cartx: {cartx}, {carty}, {cartxd}, {cartyd}")
            pdb.set_trace()

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        if self.distractor:
            gfxdraw.hline(self.surf, 0, self.screen_width, cartyd, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen), dtype='uint8'), axes=(1, 0, 2)
            )

        elif self.mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen), dtype='uint8'), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def set_noises_level(self, x):
        self.noises = x

    def addnoise(self, x):
        return {
            1: 0,
            2: self.np_rng.uniform(low=-0.05, high=0.05, size=(1,)),  # 5% actuator noise
            3: self.np_rng.uniform(low=-0.10, high=0.10, size=(1,)),  # 10% actuator noise
            4: self.np_rng.uniform(low=-0.05, high=0.05, size=(1,)),  # 5% sensor noise
            5: self.np_rng.uniform(low=-0.10, high=0.10, size=(1,)),  # 10% sensor noise
            6: self.np_rng.normal(loc=0, scale=np.sqrt(0.10), size=(1,)),  # 0.1 var sensor noise
            7: self.np_rng.normal(loc=0, scale=np.sqrt(0.20), size=(1,)),  # 0.2 var sensor noise
        }.get(x, 1)

    # def sample_one_hot_action(self):
    #     actions = 2
    #     index = self.np_rng.choice(actions)
    #     reference = np.zeros(actions, dtype=np.float32)
    #     reference[index] = 1.0
    #     return reference


if __name__ == '__main__':
    # os.environ['SDL_VIDEODRIVER'] = 'dummy'
    # env_name = 'CartPoleWorld-v01'
    env = CartPoleWorldEnv(full_state=True, mode='human')
    # print(env.gravity)
    env.reset()
    length = 200
    total_reward = 0
    for i in range(length):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # print('state, action, reward, done: ', state['state'], action, reward, done)
        env.render()
        total_reward += reward
        if done:
            break
