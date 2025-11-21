import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from .quadrotor_utils import Quadrotor, TrajectoryGenerator, run_single_episode
from scipy import linalg
import time

from .quadrotor_env import QuadrotorEnv

class QuadrotorFixedEnv(QuadrotorEnv):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, waypoints=None, total_time=None, render_mode=None, control_motors=True,
                 normalized_actions=True, fully_observable=True, boundary_length=5,
                 time_per_waypoint=0.15625, add_takeoff_waypoint=False):
        '''
        Initializes the quadrotor environment. Race track is a single lissajous curves with many twists and turns. Reference trajectory is 18 seconds, or 1800 steps. The only randomized parameter is this height of the curve.
        Waypoints:
            2D list or numpy array of goal positional coordinates for the drone.
        total_time:
            total time for drone to fly through waypoints. Used for creating smooth trajectories. Defaults to 4 seconds per waypoint.
        render_mode:
            Optional argument to display quadrotor path. Defaults to None
        control_motors:
            Flag to switch control inputs from net torques and forces to the angular velocity of each rotor. Defaults to false.
        normalized_actions:
            Flag for if actions passed to the environment are normalized. (Either motor commands or net torques and z-force). Defaults to True.
        fully_observable:
            Flag for if the state is fully observable. If False the state is assumed to be noisy measurements of (x, y, z, yaw).
        boundary_length:
            Boundary side length for arena centered at the origin. Defaults to 5 meters.
        default_time_per_waypoint: float
            Time per waypoint for quadrotor.
        add_takeoff_waypoint: bool
            If True, automatically add takeoff waypoint at index 0. If False, use waypoints as-is.
        '''
        super().__init__(waypoints=waypoints, total_time=total_time, render_mode=render_mode, control_motors=control_motors,
                         normalized_actions=normalized_actions, fully_observable=fully_observable, boundary_length=boundary_length,
                         time_per_waypoint=time_per_waypoint, add_takeoff_waypoint=add_takeoff_waypoint)


    @staticmethod
    def add_args(parser):
        parser.add_argument('--no_control_motors', action='store_false', dest='control_motors', default=True)
        parser.add_argument('--no_normalized_actions', action='store_false', dest='normalized_actions', default=True)
        parser.add_argument('--partially_observable', action='store_false', dest='fully_observable', default=True)
        parser.add_argument('--boundary_length', type=int, default=5)
        parser.add_argument('--total_time', type=float, default=None)
        parser.add_argument('--time_per_waypoint', type=float, default=0.15625)

    
    @staticmethod
    def get_env_kwargs(args):
        kwargs = {'control_motors': args.control_motors,
                  'normalized_actions': args.normalized_actions,
                  'fully_observable': args.fully_observable,
                  'boundary_length': args.boundary_length,
                  'total_time': args.total_time,
                  'time_per_waypoint': args.time_per_waypoint
                  }

        return kwargs

    def generate_lissajous_waypoints(self, n_waypoints=64):
        """
        Generate waypoints following a 3D Lissajous curve pattern.
        https://en.wikipedia.org/wiki/Lissajous_curve

        Lissajous parameterization:
        x(t) = alpha*sin(t)
        y(t) = beta*sin(n*t + phi)
        z(t) = gamma*sin(m*t + psi) + z_offset
        
        Args:
            n_waypoints: Number of waypoints to generate
            
        Returns:
            numpy array of [x, y, z, yaw] waypoints
        """
        # y oscillations
        n = 2
        # z oscillations
        m = 4

        z_offset = np.random.uniform(2.5, 3.25)
        phi = np.pi / 4
        psi = np.pi / 4
        
        alpha = 2
        beta = 2
        gamma = 1.1
        
        waypoints = np.zeros((n_waypoints, 4))
        
        for i in range(n_waypoints):
            t = 2 * np.pi * i / (n_waypoints - 1)
            
            x = alpha * np.sin(t)
            y = beta * np.sin(n * t + phi)
            z = gamma * np.sin(m * t + psi) + z_offset
            
            waypoints[i, 0] = x
            waypoints[i, 1] = y
            waypoints[i, 2] = z
            waypoints[i, 3] = 0.0
            
        return waypoints

if __name__ == '__main__':
    show_gui = True
    fully_observable = True
    # Use waypoints=None to trigger the new default randomized Lissajous path behavior
    waypoints = None
    num_waypoints = 64

    time_per_waypoint = 10.0 / num_waypoints

    reference_trajectory_time = num_waypoints * time_per_waypoint

    if show_gui:
        render_mode = 'human'
    else:
        render_mode = None
    
    add_takeoff_waypoint = False
    env = QuadrotorFixedEnv(waypoints=waypoints, total_time=reference_trajectory_time, control_motors=True, normalized_actions=True,
                       render_mode=render_mode, fully_observable=fully_observable,
                       add_takeoff_waypoint=add_takeoff_waypoint)

    model = lambda obs: env.action_space.sample()
    if not show_gui:
        num_episodes = 100
        reward = 0
        for i in range(num_episodes):
            reward += run_single_episode(model, env)
        print("Avg Episode Reward: {}".format(reward / num_episodes))
    else:
        run_single_episode(model, env)